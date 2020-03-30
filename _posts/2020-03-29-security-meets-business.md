---
title:  "Security meets Business"
tags: [infosec, vCISO, framework]
author: Jeff
---
## Security, meet Business
When some information security folks interact with other business people, it can lead to a painful realization that they may know what they do but they don’t know why they do it.

Worse still, they may lean on compliance as an overarching explanation: “We have to turn X on because of PCI”. I’m a believer that it’s the responsibility of security folks to explain things in a way that the business can understand. Let me walk through a couple common examples.

## Security - Mental Model
First, what is information security all about anyway. Do we have a shared mental model?

If you strip away CISSP terminology trivia, and regulatory prescriptions it gets simpler. I like to describe the security function of an organization like this (with references to terminology security folks may use).

Starting with understanding the things you care about (inventory) vs the threats to those things (threat modeling), our goal is to protect those things through:

- Controls to prevent attacks ([NIST CSF](https://www.nist.gov/cyberframework)/[sp800-53](https://csrc.nist.gov/publications/detail/sp/800-53/rev-4/final), [CIS20](https://www.cisecurity.org/controls/cis-controls-list/))
- Alarms to detect attacks ([MITRE ATT&CK](https://attack.mitre.org/))
- Response plans for when attacks happen ([GCIH](https://www.giac.org/certification/certified-incident-handler-gcih))
- Compliance as proof we are doing it ([PCI](https://en.wikipedia.org/wiki/Payment_Card_Industry_Data_Security_Standard))
- Red team efforts to see how well things are working ([Pentesting](https://en.wikipedia.org/wiki/Penetration_test))
- Risk assessment frameworks to be able to talk consistently about it ([FAIR](https://www.fairinstitute.org/))
- Tools to help us automate it ([DevSecOps](https://devsecops.github.io/), [SIEM](https://en.wikipedia.org/wiki/Security_information_and_event_management), [SOAR](https://resources.infosecinstitute.com/security-orchestration-automation-and-response-soar/))

This isn’t a catchy one-line phrase and it isn’t meant to be. It is meant to allow both security and business folks to arrive at a shared mental model of how the different aspects of security work together to enable the business.

Having an accurate mental model is a first step in really understanding how your work fits into the overall picture.

## PCI

Lets take a walk through PCI with this mental model in mind and see where security and business can drift.

With [PCI](https://en.wikipedia.org/wiki/Payment_Card_Industry_Data_Security_Standard) we are attempting to satisfy a compliance requirement to prove we are doing security. Compliance efforts will often talk about the difference between being compliant and being auditably compliant. The latter meaning you can reliably produce evidence that you can and have met the particulars of a requirement.

Business and security often drift apart here as the business (or the non-security technical groups) can start to believe that it’s security’s job to “do PCI.” Partnership is essential here and shared responsibility is critical to establish. It’s simply impossible for security to chase the business around and clean up card data in email, missing change control procedures, poor TLS implementations, etc.

## Red Team/Pentesting
Give me the leeway to lump these two offensive security efforts together for a moment, even though they (should) differ significantly. Misunderstanding the role of a pentest isn’t just limited to security folks though.

Quite often I’ve heard a strong desire from a board member or C-level executive insistent on performing a pentest to “improve our security”.  Remember:
“Red team efforts to see how well things are working (Pentesting)”

Pentesting a neglected security program is like taking a formula one car with a flat tire out to the races. The driver already knows what’s going to happen, the pit crew already knows what's going to happen. If you are an executive insisting on a pentest and have never supported security initiatives, you can expect the same reaction as our metaphoric pit crew.

The pit crew has a list of maintenance items, improvement points and drills they’ve love to implement before you get to the track. So does your security team.

Granted that in today’s world you are being pentested everyday, you just aren’t getting the report. Your security team also knows this and likely has a list of controls, alarms and response plans they’d love to get funding for.

## Mind Meld
While not a technical post, I hope reading this can help shore up a critical area of security programs that is often overlooked: establishing a common understanding.

>Once security and business reach an appreciation of the challenges and roles they play, you can start tackling the difficulties faced by modern organizations.

Security teams and businesses are more exposed and more challenged than ever to keep up a functional security program. If you’d like help with your efforts, consider the services of a [vCISO](https://jeffbryner.com/vciso) to help get a kickstart.
