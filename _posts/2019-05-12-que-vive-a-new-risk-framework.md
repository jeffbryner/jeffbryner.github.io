---
title:  "Que Vive: A new risk framework"
tags: [infosec, risk, framework]
author: Jeff
---
## What?
I know right? Another risk framework?! This one focuses not on the *what* but the *why*.

If you are like me, you see infosec folks all too often getting stuck in the *what* lane. 

```
We are deplying a new EDR system! 
Just finished buying a new SIEM!
etc..
```

While these base infrastructure pieces are important, it can be hard to explain *why* they are important without 
a framework to allow you to tie these back to business impact/need. This is where Que Vive can help. 

## Que Vive: what's in a name?
[Que Vive](https://www.wordnik.com/words/qui%20vive) comes from the 'who goes there' call of the French Sentinel and signifies a state of heightened vigilance, watchfullness, an vigilence. 

This framework is meant to give you the same heightened awareness of why you are doing what you are doing.

## The big idea
I'm a fan of the [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) (except the cyber part, but that's a pet peeve). It's categorization of infosec work into Identify, Protect, Detect, Respond, Recover effectively covers all activities in an easily relatable structure.

Que Vive takes this framework and uses your own rating of your progress in the NIST CSF as an indicator of likelihood. Coupling that with Impact, Assets and Threats gives you an view into your state of affairs in an approachable, methodology that you can use to explain *why* you are engaged in an infosec project.

## The Framework
Here's a link to the [base, starter spreadsheet for Que Vive](https://docs.google.com/spreadsheets/d/1jVFS6Uh6BTsGGc6yTFi1mQmCC8Vf0sQPfsu_9JpK7XE/edit?usp=sharing).

The process is as follows:

- List your assets; the things you worry about being hacked
- List your threats; the things you worry about happening to your assets
- Rate the worst case impact of each threat occuring to each asset
- Check each category of the NIST CSF that you feel you have covered for this occurance
- Que Vive will then calculate your resulting risk

You can then use this as a planning tool for areas that need further focus or prioritization. Since it's a simple spreadsheet, you can fully customize the rankings, risk calc, etc.

Here's an image of an in progress que vive session: ![semi-complete Que Vive](/assets/que-vive-sample.png)


## Next steps & feedback

[Grab yourself a copy here](https://docs.google.com/spreadsheets/d/1jVFS6Uh6BTsGGc6yTFi1mQmCC8Vf0sQPfsu_9JpK7XE/edit?usp=sharing), give it a go with ratings for your environment and let me know how it works, what can be improved, things that aren't clear, etc. Best bet is to [open issues in the github repo](https://github.com/jeffbryner/que-vive).
