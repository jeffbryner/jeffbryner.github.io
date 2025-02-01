---
title:  "Taking 6 python AI agent frameworks for a spin"
tags: [infosec, gcp, AI, agents, python]
author: Jeff
---

## Agentic AI?
AI has made a lot of advances recently, most notably the ability to call tools and link LLMs together to form agents. [Nvidia describes this well](https://blogs.nvidia.com/blog/what-is-agentic-ai/): 

```
Agentic AI uses soagnosticated reasoning and iterative planning to solve complex, multi-step problems.
```

## A tour
I've been using AI for simple one-off tasks like summarizing, labeling, etc and wanted to take some of the agentic frameworks for a tour to see what they are capable of and how well they work for more advanced tasks. 

In this post I'll cover a starting viewpoint of: 

- [langchain](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent)
- [pydantic](https://github.com/pydantic/pydantic-ai/tree/main)
- [google]() ([google-cloud-aiplatform](https://cloud.google.com/python/docs/reference/aiplatform/latest) vs [python-genai](https://github.com/googleapis/python-genai) )
- [llama](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)
- [agno](https://docs.agno.com/introduction)
- [semantic kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/samples/getting_started_with_agents)

My work these days is mostly in google cloud, so I used vertexAI and Gemini as the LLM, and python notebooks for the code environment. 

## langchain
Langchain is enourmous, with coverage and libraries that seem to cover everything so it's a bit difficult to orient to figure out where to start. I stumbled on [this agent notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/agent_executor.ipynb) that became a shortcut for me. 

### langchain setup 
Here's some starter code to get started with langchain in vertex ai and create a simple LLM
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from langchain_google_vertexai import ChatVertexAI, VertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate
)

VERTEX_PROJECT_ID = "my-project-id-in-gcp"
model="gemini-1.5-flash"

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 1,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 4096,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
}

llm = ChatVertexAI(
    model=model,
    temperature=1,
    max_tokens=4096,
    max_retries=5,
    location="us-central1",
    project=VERTEX_PROJECT_ID,
    safety_settings=safety_settings,
    streaming=True,
    
)
```

### langchain with tools
LLMs by themselves aren't much use in an agent setting. They can work on the knowlege they've been trained on but can't access real world knowlege like what day it is, or access external systems to figure out the weather, etc. For this reason it's important that a framework offers a way to make tools available to a LLM to give it more capability. 

Langchain does this via fairly straightforward code.  We create some tools and bind them to a LLM: 
```python
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)

```
To get the LLM to use the tools without an agent framework is tedious. You need to invoke the LLM, see if it needs to use tools, then resolve the tool calls: 

```python
from langchain_core.messages import HumanMessage

query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

print(ai_msg.tool_calls)

messages.append(ai_msg)

```
```markdown
### output
[{'name': 'multiply', 'args': {'a': 3.0, 'b': 12.0}, 'id': 'ce045e34-cbd7-4a33-9c91-707e5ed5b878', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11.0, 'b': 49.0}, 'id': '22b23471-ec22-4d63-bff1-f03f03f727dc', 'type': 'tool_call'}]

```
You can see the LLM figured out it should use tools, what the parameters were and setup how to invoke, but didn't use the functions. To use the functions and get the result, we invoke again: 

```python
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

messages
```
```markdown
### output

[HumanMessage(content='What is 3 * 12? Also, what is 11 + 49?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='\n', additional_kwargs={'function_call': {'name': 'multiplyadd', 'arguments': '{"a": 3.0, "b": 12.0}{"a": 11.0, "b": 49.0}'}}, response_metadata={'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE'}], 'finish_reason': 'STOP'}, id='run-412efe60-2f51-481c-9d13-91c28d349785-0', tool_calls=[{'name': 'multiply', 'args': {'a': 3.0, 'b': 12.0}, 'id': 'ce045e34-cbd7-4a33-9c91-707e5ed5b878', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11.0, 'b': 49.0}, 'id': '22b23471-ec22-4d63-bff1-f03f03f727dc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 42, 'output_tokens': 7, 'total_tokens': 49}),
 ToolMessage(content='36', name='multiply', tool_call_id='ce045e34-cbd7-4a33-9c91-707e5ed5b878'),
 ToolMessage(content='60', name='add', tool_call_id='22b23471-ec22-4d63-bff1-f03f03f727dc')]
```
Yuck right? I appreciate the safety settings in Gemini but they are verbose. You can see the tool calls and the results. 

Lastly we call the llm again with the new information so it can generate a result: 
```python
llm_with_tools.invoke(messages)
```
```markdown
### output

AIMessage(content='3 * 12 is 36. 11 + 49 is 60. \n', additional_kwargs={}, response_metadata={'safety_ratings': [snipped.for.brevity], 'finish_reason': 'STOP'}, id='run-fc63e92b-fc62-4c6f-b5fb-c7f88b04c1b3-0', usage_metadata={'input_tokens': 57, 'output_tokens': 24, 'total_tokens': 81})
```

So that's calling LLM with tools without an agent framework in langchain. 

##langchain agent
Using an agent is much friendlier when invoking tools. You setup a prompt with a scratchpad for the agent and invoke.

```python
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(  input_variables=[], 
                                    template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate( prompt=PromptTemplate(
            input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')]
)
question='What is 3 * 32? Also, what is 11 + 49?'
agent = create_tool_calling_agent(llm, tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"input": question})
```
```markdown
### output

{'input': 'What is 3 * 32? Also, what is 11 + 49?',
 'output': '3 * 32 is 96. 11 + 49 is 60. \n'}
```

If you want to see the underlying steps: 
```python
for step in agent_executor.stream({"input": question}):
    print(f"{step['messages']}\n\n")
```
```markdown
### output

[AIMessageChunk(content='', additional_kwargs={'function_call': {'name': 'multiply', 'arguments': '{"a": 3.0, "b": 32.0}'}}, response_metadata={'safety_ratings': [snipped.for.brevity], 'finish_reason': 'STOP'}, id='run-c58c557f-e380-4924-8d9a-ef6420c7fdc7', tool_calls=[{'name': 'multiply', 'args': {'a': 3.0, 'b': 32.0}, 'id': '084a05fc-f4b4-4f85-b933-93ca55a7f42b', 'type': 'tool_call'}], usage_metadata={'input_tokens': 47, 'output_tokens': 3, 'total_tokens': 50}, tool_call_chunks=[{'name': 'multiply', 'args': '{"a": 3.0, "b": 32.0}', 'id': '084a05fc-f4b4-4f85-b933-93ca55a7f42b', 'index': None, 'type': 'tool_call_chunk'}])]


[FunctionMessage(content='96', additional_kwargs={}, response_metadata={}, name='multiply')]


[AIMessageChunk(content='', additional_kwargs={'function_call': {'name': 'add', 'arguments': '{"a": 11.0, "b": 49.0}'}}, response_metadata={'safety_ratings': [snipped.for.brevity], 'finish_reason': 'STOP'}, id='run-99faaa9a-aadf-432d-8e04-d47cf7adb75b', tool_calls=[{'name': 'add', 'args': {'a': 11.0, 'b': 49.0}, 'id': '319829a3-9274-41dd-b06b-b23ea8552673', 'type': 'tool_call'}], usage_metadata={'input_tokens': 54, 'output_tokens': 3, 'total_tokens': 57}, tool_call_chunks=[{'name': 'add', 'args': '{"a": 11.0, "b": 49.0}', 'id': '319829a3-9274-41dd-b06b-b23ea8552673', 'index': None, 'type': 'tool_call_chunk'}])]


[FunctionMessage(content='60', additional_kwargs={}, response_metadata={}, name='add')]


[AIMessage(content='3 * 32 is 96. 11 + 49 is 60. \n', additional_kwargs={}, response_metadata={})]
```

## pydantic ai
Pydantic itself is at the core of many AI frameworks these days to give structured types to functions, classes, output, etc. Their AI framework is friendly and simple to use. 

```python
!pip install pydantic-ai

import nest_asyncio
nest_asyncio.apply() #needed for jupyter notebook environment
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.vertexai import VertexAIModel
from typing import List
# create model and agent
model = VertexAIModel('gemini-1.5-flash')
agent = Agent(model,
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.data)
print(result.all_messages())
```
```markdown
### output

"Hello, world!" is a traditional introductory program in computer programming. 

[ModelRequest(parts=[SystemPromptPart(content='Be concise, reply with one sentence.', dynamic_ref=None, part_kind='system-prompt'), UserPromptPart(content='Where does "hello world" come from?', timestamp=datetime.datetime(2025, 1, 21, 19, 54, 7, 629789, tzinfo=datetime.timezone.utc), part_kind='user-prompt')], kind='request'), ModelResponse(parts=[TextPart(content='"Hello, world!" is a traditional introductory program in computer programming. \n', part_kind='text')], timestamp=datetime.datetime(2025, 1, 21, 19, 54, 9, 621057, tzinfo=datetime.timezone.utc), kind='response')]
```

To add a tool to an agent is straightforward
```python
@agent.tool_plain
def multiply_numbers(numbers: List[int]) -> int:
  """
  Calculates the product of all numbers in an array.

  Args:
      numbers: An array of numbers to be multiplied.

  Returns:
      The product of all the numbers. If the array is empty, returns 1.
  """

  if not numbers:  # Handle empty array
      return 1

  product = 1
  for num in numbers:
      product *= num

  return product

result = agent.run_sync('What is 12*112 * 4?')  
print(result.data)
print(result.all_messages())
```
```markdown
### output

The product of 12, 112, and 4 is 5376. 

[ModelRequest(parts=[SystemPromptPart(content='Be concise, reply with one sentence.', dynamic_ref=None, part_kind='system-prompt'), UserPromptPart(content='What is 12*112 * 4?', timestamp=datetime.datetime(2025, 1, 21, 19, 59, 55, 853801, tzinfo=datetime.timezone.utc), part_kind='user-prompt')], kind='request'), ModelResponse(parts=[ToolCallPart(tool_name='multiply_numbers', args=ArgsDict(args_dict={'numbers': [12, 112, 4]}), tool_call_id=None, part_kind='tool-call')], timestamp=datetime.datetime(2025, 1, 21, 19, 59, 56, 753382, tzinfo=datetime.timezone.utc), kind='response'), ModelRequest(parts=[ToolReturnPart(tool_name='multiply_numbers', content=5376, tool_call_id=None, timestamp=datetime.datetime(2025, 1, 21, 19, 59, 56, 764839, tzinfo=datetime.timezone.utc), part_kind='tool-return')], kind='request'), ModelResponse(parts=[TextPart(content='The product of 12, 112, and 4 is 5376. \n', part_kind='text')], timestamp=datetime.datetime(2025, 1, 21, 19, 59, 57, 265541, tzinfo=datetime.timezone.utc), kind='response')]
```
The biggest drawback I found in this frameworks is that there doesn't appear to be a way to set safety settings for Vertex/Gemini. Without this you are using the default settings which are conservative and will eventually end up with blocked requests. I believe this is meant to be addressed [model by model](https://github.com/pydantic/pydantic-ai/issues/71)

It is simple to create multi-agent interplay like this joke telling symphony

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

joke_selection_agent = Agent(  
    model,
    system_prompt=(
        'Use the `joke_factory` to generate some jokes, then choose the best. '
        'You must return just a single joke.'
    ),
)
joke_generation_agent = Agent(model, result_type=list[str])  


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    logger.info(type(ctx.deps))
    r = await joke_generation_agent.run(  
        f"Please generate {ctx.deps['count']} jokes about {ctx.deps['subject']}.",
        usage=ctx.usage,  
    )
    return r.data  


result = joke_selection_agent.run_sync(
    'Tell me a joke.',
    deps={'count':5,'subject':'dogs'},
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4096),
)
print(result.data)
print(result.usage())
```
```markdown
### output

Why don't dogs play poker? Because they keep getting 🐶  ace 🐶  of spades! 

Usage(requests=3, request_tokens=199, response_tokens=131, total_tokens=330, details=None)

```
Not exactly SNL, but it's great that this framework supplies an easy way to tailor the usage (tokens, requests) along with each agent. Without controls like this it is easy to overrun quotas. 

Pydantic also goes much further with [graphs to control agent action](https://ai.pydantic.dev/graph/) which I did not explore.

## Google
Google (confusingly) has multiple SDKs for interacting with Gemini in Vertex and they operate differently. 

### google-cloud-aiplatform

[I followed this guide](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#python-from-function) for using the google-cloud-aiplatform python SDK and using it for tool calling. 


```python
!pip install --upgrade google-cloud-aiplatform

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    ChatSession,
    Part,
    Content,
    FunctionDeclaration,
    Tool,
    ToolConfig,
    GenerationConfig
)

VERTEX_PROJECT_ID = "my-gcp-project-id"

vertexai.init(project=VERTEX_PROJECT_ID, location="us-central1")

#setup the tool
from typing import List

# Define a function. Could be a local function or you can import the requests library to call an API
def multiply_numbers(numbers: List[int]) -> int:
  """
  Calculates the product of all numbers in an array.

  Args:
      numbers: An array of numbers to be multiplied.

  Returns:
      The product of all the numbers. If the array is empty, returns 1.
  """

  if not numbers:  # Handle empty array
      return 1

  product = 1
  for num in numbers:
      product *= num

  return product

multiply_number_func = FunctionDeclaration.from_func(multiply_numbers)
tool = Tool(
    function_declarations=[multiply_number_func],
)

user_prompt_content = Content(
    role="user",
    parts=[
        Part.from_text("What is 3 times 6 times 4?"),
    ],
)

#give it a system instruction
system_instruction=[
    "You are a helpful math expert.",
    "Your mission is to help people learn math.",
]

generation_model = GenerativeModel("gemini-1.5-pro",system_instruction=system_instruction)
 

response = generation_model.generate_content(
    user_prompt_content,
    generation_config=GenerationConfig(temperature=0),
    tools=[tool]
)
logger.info(response.candidates)
```
```markdown
### output

INFO:root:[content {
  role: "model"
  parts {
    function_call {
      name: "multiply_numbers"
      args {
        fields {
          key: "numbers"
          value {
            list_value {
              values {
                number_value: 3
              }
              values {
                number_value: 6
              }
              values {
                number_value: 4
              }
            }
          }
        }
      }
    }
  }
}
avg_logprobs: -0.024972693994641304
finish_reason: STOP
safety_ratings...
]
```
I didn't go further with this framework since tool calling is manual without further integration. [Google appears to be building reasoning engines with langchain to bring this framework closer to an agent.](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)

### python-genai
The [python-genai](https://github.com/googleapis/python-genai) sdk also supports tool calling, but in an automated way more like the other agenic frameworks. 

```python 
#https://googleapis.github.io/python-genai/
!pip install google-genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import datetime
from typing import List


from google import genai
from google.genai import types
PROJECT_ID = "my-gcp-project-id"

# init for Vertex AI API
client = genai.Client(
    vertexai=True, project=PROJECT_ID, location="us-central1"
)

# tools

def multiply_numbers(numbers: List[int]) -> int:
  """
  Calculates the product of all numbers in an array.
  Args:
      numbers: An array of numbers to be multiplied.
  Returns:
      The product of all the numbers. If the array is empty, returns 1.
  """

  if not numbers:  # Handle empty array
      return 1

  product = 1
  for num in numbers:
      product *= num

  return product
    
def day_of_week() -> str:
    """Get the current day of the week.
    Example:
        {{time.dayOfWeek}} => Sunday
    """
    now = datetime.datetime.now()
    return now.strftime("%A")

response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents="What day is it today?",
    config=types.GenerateContentConfig(
        tools=[day_of_week, multiply_numbers],
    ),
)
print(response.text)

```
```markdown
### output

Today is Sunday.
```
I like that tool definition is straightfoward and only requires declaring normal python function. The library does not appear to offer agent interaction like pydantic. 

## llama
I used [this guide to take a brief tour of llama](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/). 

```python
!pip install --upgrade google-cloud-aiplatform  llama-index llama_index-llms-vertex
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from google.cloud import aiplatform
from typing import List
from llama_index.llms.vertex import Vertex
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
import datetime

aiplatform.init(project=PROJECT_ID, location=REGION)

PROJECT_ID = "my-gcp-project-id"
REGION="us-central1"

# tools
def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def day_of_week() -> str:
    """Get the current day of the week.

    Example:
        {{time.dayOfWeek}} => Sunday
    """
    now = datetime.datetime.now()
    return now.strftime("%A")

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
day_of_week = FunctionTool.from_defaults(fn=day_of_week)
tools = [multiply_tool, add_tool, day_of_week]

# 
vertex_gemini = Vertex(
    model="gemini-1.5-pro",
    temperature=0,
    system_prompt="you are a french math wizard, use french as much as possible",
    additional_kwargs={},
)

Settings.llm=vertex_gemini
agent = FunctionCallingAgent.from_tools(
    tools,
    llm=Settings.llm,
    verbose=False,
    allow_parallel_tool_calls=True,
)
response = agent.query("tell me the answer to this math problem: What is (121 + 2) * 5?")
print(str(response))
```
```markdown
### output

You are right! 
```

Trying the 'function calling agent' always led to this sort of output for me. It would not tell me the answer, but would congratulate me on being right? 

I had better luck with the react agent

```python

re_agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=False)
response = re_agent.chat("tell me the answer to this math problem: What is (121 + 2) * 5?")
print(str(response))
```
```markdown
### output

The answer is 615.
```
Setting verbose=True would show me the thinking and the tool use

```python
re_agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)
response = re_agent.chat("what day of the week is tomorrow?")
print(str(response))
```
```markdown
### output

> Running step 2c16b54d-eab0-4695-85c3-75036ca28bba. Step input: what day of the week is tomorrow?
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: day_of_week
Action Input: {}
Observation: Monday
> Running step 10737206-8a58-4e3a-a967-a25cdc01990a. Step input: None
Thought: Today is Monday so tomorrow is Tuesday. I can answer without using any more tools. I'll use the user's language to answer
Answer: Tomorrow is Tuesday.
Tomorrow is Tuesday.
```
Neat! I appreciate how easy it is to create and use tools, even if one method doesn't appear to work as well for results. I also appreciated seeing the thought process rather than having to interpret messages. 


## [agno](https://docs.agno.com/introduction)

(Note: agnodata just changed their name to agno. )
Agno wasn't on my radar, but I stumbled onto it via a [youtube video showing how you can use it and duckdb to create a streaming stats chat app](https://youtu.be/sVBFPNW_GGc?feature=shared). 

It's a feature-rich framework offering agents, storage for memory, tools and much more. 

```python
!pip install agno duckduckgo-search duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from agno.agent import Agent, AgentMemory
from agno.models.vertexai import Gemini
from agno.tools.duckdb import DuckDbTools
from textwrap import dedent

from agno.memory.db.sqlite import SqliteMemoryDb
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.memory.classifier import MemoryClassifier
from agno.memory.summarizer import MemorySummarizer
from agno.memory.manager import MemoryManager
from agno.utils.pprint import pprint_run_response


import json

from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)

model="gemini-1.5-flash"
generation_config = GenerationConfig(
    temperature=0,
    top_p=0.1,
    top_k=1,
    max_output_tokens=4096,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
#simple agent
agent=Agent(model=Gemini(id=model,
                         generation_config=generation_config,
                        safety_settings=safety_settings,),
            telemetry=False)
response=agent.run("what is the best day of the week?")
pprint_run_response(response,markdown=True)
```
```markdown
### output

┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃ what is the best day of the week?                                            ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (1.5s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃ There's no single "best" day of the week! It's all about personal preference ┃
┃ and what you enjoy.                                                          ┃
┃                                                                              ┃
┃ Here's a breakdown of why people might love different days:                  ┃
┃                                                                              ┃
┃ * **Monday:** A fresh start, a chance to tackle new goals.                   ┃
┃ * **Tuesday:** The hump day, but also a chance to get into the groove of the ┃
┃ week.                                                                        ┃
┃ * **Wednesday:** The middle ground, a chance to reflect and recharge.        ┃
┃ * **Thursday:** Almost the weekend, a sense of anticipation.                 ┃
┃ * **Friday:** The day of freedom, time to relax and unwind.                  ┃
┃ * **Saturday:** The day for fun, socializing, and exploring.                 ┃
┃ * **Sunday:** A day for rest, reflection, and spending time with loved ones. ┃
┃                                                                              ┃
┃ Ultimately, the best day of the week is the one that brings you the most joy ┃
┃ and fulfillment. 😊  
┃                                                                              ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

It's great that it can initiate a vertexAI model with little fuss and accept safety settings, etc. 

You can give it tools and easily use them to assist with answers: 

```python
from agno.tools.duckduckgo import DuckDuckGoTools

assistant = Agent(, tools=[DuckDuckGoTools()], show_tool_calls=True,debug_mode=True)
assistant.print_response("Whats happening with Pete Carroll?", markdown=True,stream=True)
```
```markdown
### output

╭──────────┬───────────────────────────────────────────────────────────────────╮
│ Message  │ Whats happening with Pete Carroll?                                │
├──────────┼───────────────────────────────────────────────────────────────────┤
│ Response │                                                                   │
│ (1.8s)   │  • Running: duckduckgo_search(query=Pete Carroll)                 │
│          │                                                                   │
│          │ It looks like Pete Carroll is returning to the NFL as the head    │
│          │ coach of the Las Vegas Raiders! He's agreed to a three-year deal  │
│          │ with a fourth-year                                                │
```

What really amazed me was the combination of using duckDb tools with an agent. Here's a mini program using an open source csv file to answer movie questions. 

```python

duckdb_tools = DuckDbTools(create_tables=False, export_tables=False, summarize_tables=False)
duckdb_tools.create_table_from_path(
    path="https://agnodata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv", table="movies"
)

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[duckdb_tools],
    markdown=True,
    show_tool_calls=True,
    instructions="You are a movie expert with instant access to movie ratings",
    additional_context=dedent("""\
    You have access to the following tables:
    - movies: Contains information about movies from IMDB.
    It has columns for the Title, Genre, Description, Revenue (Millions), Metascore and Actors
    """),
    debug_mode=True
)
agent.print_response("What movie had the highest metascore? What is it's description? Who were the actors?", stream=True)
```

```markdown
### output

INFO     Running: CREATE TABLE IF NOT EXISTS 'movies' AS SELECT * FROM          
         'https://agnodata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv'
INFO     Running: SELECT Title, Description, Actors FROM movies ORDER BY        
         Metascore DESC LIMIT 1                                                 
▰▰▰▱▱▱▱ Thinking...

INFO:httpx:HTTP Request: POST https://api.agnodata.com/v1/telemetry/agent/run/create "HTTP/1.1 200 OK"

┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃ What movie had the highest metascore? What is it's description? Who were the ┃
┃ actors?                                                                      ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (2.8s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃                                                                              ┃
┃  • Running: run_query(query=SELECT Title, Description, Actors FROM movies    ┃
┃    ORDER BY Metascore DESC LIMIT 1)                                          ┃
┃                                                                              ┃
┃ The movie with the highest metascore is Boyhood.                             ┃
┃                                                                              ┃
┃ Its description is: The life of Mason, from early childhood to his arrival   ┃
┃ at college.                                                                  ┃
┃                                                                              ┃
┃ The actors are: Ellar Coltrane, Patricia Arquette, Ethan Hawke, and Elijah   ┃
┃ Smith.                                                                       ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

```

This was my favorite of all the frameworks and I'll definitely be exploring more of it's features like memory/knowlege stores and multi agent workflows. 

## [semantic kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/samples/getting_started_with_agents)

I was most excited to try this one as it seemed like it was built to facilitate agent interactions. 

I got some simple agent and tool calling to work, but not the multi agent chat features. 

```python 
# ran into a numpy compatibility issue, selected a specific version
!pip install --upgrade semantic-kernel pandas numpy==1.26.4

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# it is big! 
import asyncio
from functools import reduce
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.google.vertex_ai import VertexAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

VERTEX_PROJECT_ID = "my-gcp-project-id"
model="gemini-1.5-pro"
service_id="vertex_agent"

kernel = Kernel()
kernel.add_service(
    VertexAIChatCompletion(
        service_id=service_id,
        project_id=VERTEX_PROJECT_ID,
        gemini_model_id=model,
    )
)

async def invoke_agent(agent: ChatCompletionAgent, input: str, chat: ChatHistory):
    """Invoke the agent with the user input."""
    chat.add_user_message(input)

    print(f"# {AuthorRole.USER}: '{input}'")

    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(chat):
            content_name = content.name
            contents.append(content)
        message_content = "".join([content.content for content in contents])
        print(f"# {content.role} - {content_name or '*'}: '{message_content}'")
        chat.add_assistant_message(message_content)
    else:
        async for content in agent.invoke(chat):
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
        chat.add_message(content)


# Define a sample plugin/tool for use
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"

# Define the agent name and instructions
HOST_NAME = "Host"
HOST_INSTRUCTIONS = "Answer questions about the menu."

settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
# Configure the function choice behavior to auto invoke kernel functions
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

kernel.add_plugin(MenuPlugin(), plugin_name="menu")
# Create the agent
agent = ChatCompletionAgent(
    service_id=service_id, kernel=kernel, name=HOST_NAME, instructions=HOST_INSTRUCTIONS, execution_settings=settings
)

# Define the chat history
chat = ChatHistory()

# Respond to user input
await invoke_agent(agent, "Hello", chat)
await invoke_agent(agent, "What is the special soup?", chat)
await invoke_agent(agent, "What is the special drink?", chat)
await invoke_agent(agent, "Thank you", chat)
```

```markdown
### output

# user: 'Hello'

INFO:semantic_kernel.agents.chat_completion.chat_completion_agent:[ChatCompletionAgent] Invoked VertexAIChatCompletion with message count: 2.

# assistant - Host: 'How can I help you today? 😊 
'
# user: 'What is the special soup?'

INFO:semantic_kernel.connectors.ai.chat_completion_client_base:processing 1 tool calls in parallel.
INFO:semantic_kernel.kernel:Calling menu-get_specials function with args: {}
INFO:semantic_kernel.functions.kernel_function:Function menu-get_specials invoking.
INFO:semantic_kernel.functions.kernel_function:Function menu-get_specials succeeded.
INFO:semantic_kernel.functions.kernel_function:Function completed. Duration: 0.001728s
INFO:semantic_kernel.agents.chat_completion.chat_completion_agent:[ChatCompletionAgent] Invoked VertexAIChatCompletion with message count: 4.

# assistant - Host: 'The special soup is Clam Chowder. 
'
# user: 'What is the special drink?'

INFO:semantic_kernel.connectors.ai.chat_completion_client_base:processing 1 tool calls in parallel.
INFO:semantic_kernel.kernel:Calling menu-get_specials function with args: {}
INFO:semantic_kernel.functions.kernel_function:Function menu-get_specials invoking.
INFO:semantic_kernel.functions.kernel_function:Function menu-get_specials succeeded.
INFO:semantic_kernel.functions.kernel_function:Function completed. Duration: 0.001284s
INFO:semantic_kernel.agents.chat_completion.chat_completion_agent:[ChatCompletionAgent] Invoked VertexAIChatCompletion with message count: 8.

# assistant - Host: 'The special drink is Chai Tea. 
'
# user: 'Thank you'

INFO:semantic_kernel.agents.chat_completion.chat_completion_agent:[ChatCompletionAgent] Invoked VertexAIChatCompletion with message count: 12.

# assistant - Host: 'You're welcome! Is there anything else I can help you with? 😊 
```

This was great! Rather large and a lot of setup compared to others but performed well and relatively easy to create tools. 

I attempted to use the [multi agent chat as described here](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/examples/example-agent-collaboration?pivots=programming-language-python) but I was unable to get it to work. 
It returned errors about the message formatting which I suspect is an incompatibilty with Gemini's preferred formatting
```markdown
### output

InvalidArgument: 400 Unable to submit request because it must include at least one parts field, which describes the prompt input. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
```

# Conclusion
Whew, you read this far?! Congrats and thanks for riding along with this exploration of just some of the emerging agent frameworks. There are a lot more, and I'm sure I'm missing some. 0x7eff on most social networks if you'd like to point me in a particular direction. 
