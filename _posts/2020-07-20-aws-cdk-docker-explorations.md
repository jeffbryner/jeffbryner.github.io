---
title:  "Taking AWS CDK for a spin for deploying docker containers"
tags: [infosec, aws, docker, containers, python]
author: Jeff
---

## Docker

If you are like me (or anyone?) you use docker in your development workflow. Nothing better than a fresh docker-compose.yml environment to see how your code will actually operate in today's world of kubernetes, ECS, etc.

## The Holy Grail
Having an easy way to take a local docker construct and make it 'production' in a cloud environment has been a long pursuit and there are a variety of technologies that may get you there. Everyone seems to have their favorite.

I took some time recently to check out the current state of the state as it relates to easily deploying containers to AWS. Here's where I settled.

### Docker ecs context
Docker [recently announced](https://www.docker.com/blog/from-docker-straight-to-aws/) a neat plugin for their 'edge' version of docker desktop that allows you to take your docker-compose.yml file and deploy a [sample app consisting of a custom flask program and a stock redis container](https://github.com/docker/ecs-plugin/tree/master/example) to amazon's Fargate Container service in like 3 command line steps!

This is super cool, and a great way for developers to make the leap from their local desktop to a cloud environment without making many changes to their app.

### AWS CDK
[Released in 2018]( https://aws.amazon.com/blogs/developer/aws-cdk-developer-preview/) Amazon's Cloud Development Kit is an interesting opportunity to 'meet you where you are' and bring your python, typescript, java, javascript, or C## skills to bear when deploying cloud infrastructure.

It has a [really, really deep API](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-construct-library.html) covering all aspects of AWS and allows you high or low level access to AWS services to patch them together in whatever way your app needs.

## Taking AWS CDK for a spin

While the docker workflow is neat and works right out of the box it doesn't seem like it would survive to a production CI/CD workflow so I decided to take the docker sample and see what it would take to get it working via AWS CDK.

It's not intuitive and while there are examples, they don't quite get you there so to save the next person on this journey. Here's what works (so far).

[Here's the github repo if you'd like to follow along in a real deployment](https://github.com/jeffbryner/aws-cdk-example-deployment). It includes setting up a pipenv environment with all the libraries you'll need, etc.


## The (simple) app
The [app in the docker sample](https://github.com/docker/ecs-plugin/tree/master/example) is a relatively simple combination of two containers. A 'frontend' flask app with a template talking to a 'backend' redis service. Simple on the surface, but it goes beyond the [hello world examples](https://github.com/aws-samples/aws-cdk-examples) in the stock [CDK examples using only one container](https://github.com/aws-samples/aws-cdk-examples/blob/master/python/ecs/fargate-load-balanced-service/app.py).

Still, look at how easy it is to declare an ECS cluster with an app from a docker image

```python
fargate_service = ecs_patterns.NetworkLoadBalancedFargateService(
    self, "FargateService",
    cluster=cluster,
    task_image_options={
        'image': ecs.ContainerImage.from_registry("amazon/amazon-ecs-sample")
    }
)
```

This construct plus a ```cdk deploy``` is almost all you need to get an app live provisioned via cloud formation.


## The rub
The complexity comes when you have more than one container as part of your app and they need to live in a common space and be able to communicate together. Now you need more than one ECS Task, DNS resolution, etc. The docker ecs context handles this very well and allows you to reference containers by name ('flask' or 'redis') and auto-provisions security group rules, etc. The AWS CDK needs a bit of prodding to arrive at the same functionality.

## AWS CDK with all the fixins
So how do we get a replica of the docker experience with AWS CDK?

[Here's the final code again in github](https://github.com/jeffbryner/aws-cdk-example-deployment). But to step through the details, here's what we need:

- A VPC
- An ECS Fargate cluster
- A cloud map service discovery instance
- A way to build docker containers
- ECS tasks per container
- Security group rules to allow comms between containers

### VPC
Easy enough:
``` python
vpc = ec2.Vpc(self, "SampleVPC", max_azs=2)  # default is all AZs in region
```

### ECS cluster
```python
cluster = ecs.Cluster(self, "ServiceCluster", vpc=vpc)
```

### Service Discovery
The ability for one service to reference another starts with the instantiation of a cloud map for our cluster:
```python
cluster.add_default_cloud_map_namespace(name="service.local")
```

### Frontend Task
OK, now that we've declared our cluster lets start at the front end python/flask app declaration:

``` python
frontend_asset = DockerImageAsset(
    self, "frontend", directory="./frontend", file="Dockerfile"
)
frontend_task = ecs.FargateTaskDefinition(
    self, "frontend-task", cpu=512, memory_limit_mib=2048,
)
frontend_task.add_container(
    "frontend",
    image=ecs.ContainerImage.from_docker_image_asset(frontend_asset),
    essential=True,
    environment={"LOCALDOMAIN": "service.local"},
    logging=ecs.LogDrivers.aws_logs(
        stream_prefix="FrontendContainer",
        log_retention=logs.RetentionDays.ONE_WEEK,
    ),
).add_port_mappings(ecs.PortMapping(container_port=5000, host_port=5000))
```
You can see we are accomplishing a couple things here that docker did automatically. We build an 'asset' from a local Dockerfile, make a task and attach the container to the task with the proper port mapping and service discovery label and logging construct.

Adding a LOCALDOMAIN to the environment of the container allows us to reference the domain when building the container to properly resolve 'backend' as 'backend.service.local' within the service discovery namespace.

This happens in the entrypoint.sh script of the docker build:
```bash
#! /bin/sh

if [ "${LOCALDOMAIN}" != ""  ]; then echo "search ${LOCALDOMAIN}" >> /etc/resolv.conf; fi
exec "$@"
```

This way, the 'frontend' container can use 'backend' as the name for the container it wants to connect to just like on our docker desktop compose environment!

### Backend Task
This is less complicated, just a stock redis build

``` python
backend_task = ecs.FargateTaskDefinition(
    self, "backend-task", cpu=512, memory_limit_mib=2048,
)
backend_task.add_container(
    "backend",
    image=ecs.ContainerImage.from_registry("redis:alpine"),
    essential=True,
    logging=ecs.LogDrivers.aws_logs(
        stream_prefix="BackendContainer",
        log_retention=logs.RetentionDays.ONE_WEEK,
    ),
).add_port_mappings(ecs.PortMapping(container_port=6379, host_port=6379))
```

A relatively simple addition of the redis:apline docker image to our environment.

### Services
OK, now that we have the tasks/containers, we can configure them as ECS services:

```python
frontend_service = ecs_patterns.NetworkLoadBalancedFargateService(
    self,
    id="frontend-service",
    service_name="frontend",
    cluster=cluster,  # Required
    cloud_map_options=ecs.CloudMapOptions(name="frontend"),
    cpu=512,  # Default is 256
    desired_count=2,  # Default is 1
    task_definition=frontend_task,
    memory_limit_mib=2048,  # Default is 512
    listener_port=80,
    public_load_balancer=True,
)

frontend_service.service.connections.allow_from_any_ipv4(
    ec2.Port.tcp(5000), "flask inbound"
)
```

The ecs_patterns api is great for quick calls to build all the underlying subnets, routing tables, load balancers, etc that are required to support a higher level construct like a docker container/service.

Key here is the use of the cloud_map_options parameter with the name of the service to register it in our service discovery. Without this the service gets a randomly assigned name and you'll have to do DNS gymnastics to get them to talk.

Augmenting the inbound security group rules to allow the load balancer to connect to our flask container on the default port 5000 is also key since the default security group build contains no ingress rules.

Here's the backend service:

```python
backend_service = ecs_patterns.NetworkLoadBalancedFargateService(
    self,
    id="backend-service",
    service_name="backend",
    cluster=cluster,  # Required
    cloud_map_options=ecs.CloudMapOptions(name="backend"),
    cpu=512,  # Default is 256
    desired_count=2,  # Default is 1
    task_definition=backend_task,
    memory_limit_mib=2048,  # Default is 512
    listener_port=6379,
    public_load_balancer=False,
)

backend_service.service.connections.allow_from(
    frontend_service.service, ec2.Port.tcp(6379)
)
```

To be clear, this is in no way a properly clustered redis instance. It's just a couple instances that will be out of sync with each other but will be accessible to the frontend container.

The important bit here is to allow connections from the frontend service to the backend service over the port we are using for redis: 6379. Doing it from the ```frontend_service.service``` object allows AWS to create a security group ACL that is limited to the security group for the frontend service.

## Deployment
If you've made it this far with a valid environment, deployment is as simple as a

```bash
cdk deploy
```

### Compile and validate
It will evaluate your python for validity, create a cloud formation template, evaluate it for security concerns, prompt you for allowing any security changes:

![evaluate for security concerns](/assets/aws-cdk-docker-explorations/security_verification.png)

### Realtime cloud formation
When you execute the cloudformation deployment you'll get realtime updates on the progress:
![build underway](/assets/aws-cdk-docker-explorations/build_underway.png)

### Stack output
If all goes well, you'll be rewarded with output of your load balanced endpoints that you can copy and paste into firefox to view your sample service!

![stack output](/assets/aws-cdk-docker-explorations/stack_output.png)

### Voila
The final webpage is a simple interactive service sending timestamps to redis and out to the flask template page:
![final webpage](/assets/aws-cdk-docker-explorations/final_service.png)

Enjoy!













