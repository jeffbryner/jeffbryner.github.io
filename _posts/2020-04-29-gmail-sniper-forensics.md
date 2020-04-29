---
title:  "Gmail sniper forensics"
tags: [infosec, forensics, incident response, jupyter, gmail]
author: Jeff
---

## Gmail Forensics
Why a post on gmail forensics? I fielded a question from a colleague recently about a scenario where someone was sent a file by accident and how best to prove that they had removed it from their inbox. Most mail forensics deals more with e-discovery; finding emails pertaining to specific keywords or subject matter. Uncovering what actions a user performed on an email message is a different sort of investigation and it made for some interesting discoveries that may be of use to those of you doing forensic work in gmail. Honing in on particular points of interest rather than wading through massive quantities of data (sometimes called [‘sniper forensics’](https://digital-forensics.sans.org/summit-archives/2010/2-newell-spiderlabs-sniper-forensics.pdf)) using the api can help you speed up your investigations and quickly solve forensic investigations.

![message history](/assets/gmail-sniper-forensics/message_mini_history.png)

## Gmail API
Why use the [Gmail API?](https://developers.google.com/gmail/api/v1/reference/users/messages) Most forensic processes for email involve exporting email to a PST or MBOX file and then using a custom tool to analyze it. Usually this means you need all the email from the account and you can’t necessarily tell what actions a user performed, you can only tell if a mail was received, sent, deleted based on whether it exists in their mailbox.

With the Gmail API, you can hone in on a targeted message and get much more information about actions the user took which may be helpful to show intent.


## Jupyter/Scenario
I’ve done [other posts about jupyter notebooks](http://blog.jeffbryner.com/2020/04/02/jupyter-notebooks-up-and-running.html) and we will use another one in this investigation. [Here’s the notebook we will use this time around.](
    https://github.com/jeffbryner/jeffbryner.github.io/blob/master/assets/gmail-sniper-forensics/Gmail_api_v1_sniper_forensics.ipynb)

In this scenario, I’ve sent myself an empty spreadsheet (‘aSheet.xls’) as an attachment to an email with the subject ‘a spreadsheet’. Let's see what we can discover about what I did with it using the Gmail api.

Let's load up the gmail api using the [same method discussed earlier](http://blog.jeffbryner.com/2020/04/07/incident-response-using-google-drive-api.html) by creating a Google cloud project with delegated access to the Gsuite domain. The scopes you’ll need are:

```Python
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
```

You’ll create credentials and gain delegated access to the mailbox in question:

```Python
# setup delegated access to a user's mail
user_email="jeff@jeffbryner.com"
user_id="me"
try:
    credentials = credentials.with_subject(user_email)
    gmail = build('gmail', 'v1', credentials=credentials)
except Exception as e:
    print(e)

```

## Orienting
I find it useful to gain context first about a target. Everyone uses email differently, some folks are inbox zero, some use it as a dumping ground, some are folder obsessed. First off lets get a quick overview of what we are dealing with:

```Python
# historyId returned in the profile is useful for orienting to
# where to start looking in history for actions on a message

gmail.users().getProfile(userId=user_id).execute()
{'emailAddress': '0x7eff@jeffbryner.com',
 'messagesTotal': 2101,
 'threadsTotal': 1215,
 'historyId': '3467145'}
```

You can see I’m a bit of an email pack-rat, but not unruly. The history ID will come in handy later as we investigate activity. Let's get a look at the folders with:

```Python
gmail.users().labels().list(userId=user_id).execute()
```

This will return a list of standard folders/labels like:

```
{'id': 'SENT', 'name': 'SENT', 'type': 'system'},
{'id': 'INBOX', 'name': 'INBOX', 'type': 'system'},
```
And custom ones the user has created like:

```
{'id': 'Label_2539442913597987508', 'name': 'guitar', 'type': 'user'},
```

## Investigating
Now that we have a bit of orientation, we can narrow our focus on our target email. To find the email in question we can use the api to search just as if we had the UI searchbox in front of us

We can setup a query:

```Python
query=r"""
from:jeff@jeffbryner.com subject:'a spreadsheet'
"""
```

And then do a simple search, returning the headers and some message identifiers:

```Python
messages=gmail.users().messages().list(userId=user_id,
                              maxResults=100,
                              includeSpamTrash=True,
                              q=query).execute()
for item in messages['messages']:
    message = gmail.users().messages().get(userId=user_id, id=item['id']).execute()
    print(f"id:{message['id']}, historyId: {message['historyId']}, labels: {message['labelIds']}")
    print(yaml.dump(message['payload']['headers']))
```

Here we uncover the message ID in question:

![message id](/assets/gmail-sniper-forensics/message_metadata.png)

We can use that message ID to get the full details of the message, including attachments

```Python
message=gmail.users().messages().get(userId=user_id, format='full',id='171a29788138b18f').execute()
message
```
![message id](/assets/gmail-sniper-forensics/message_details.png)

A couple call outs here. We can see from the labelIds field that this email is currently in both the TRASH and SENT system folders. So immediately we know the user sent this email, and it is currently in TRASH awaiting final, permanent deletion.

The historyId is important when looking back to see what a user did with a message. Recall the current historyId for the users mailbox is '3467145' and the historyId for this message is ‘3462182’. As far as I can tell, the historyId field is solely used to help synchronize mailboxes across clients. So if you were writing an app to display a gmail inbox, you’d store the historyId when you last synchronized with Google, and then roll forward transactions by historyId to resynchronize.


## Message history
Let's use historyId to see what the user did with this message before landing it in the TRASH folder/label. The history records appear to be a bit of a dark art, but they at least contain logical fields for the historyId, the messageId and some idea of intent through the ‘labelsAdded’ and ‘labelsRemoved’ fields.

It’s not an exact science to know where to start, but if we subtract a thousand from the historyId in the message we can see if we have any hits for our target message:


```Python
# set the message ID
message_id='171a29788138b18f'

# for history ID, start at least a thousand back from what is returned from 'historyId' for a particular message
# and adjust accordingly
start_history_id=3461100

history = (gmail.users().history().list(userId=user_id, startHistoryId=start_history_id).execute())
changes = history['history'] if 'history' in history else []
while 'nextPageToken' in history:
    page_token = history['nextPageToken']
    history = (gmail.users().history().list(userId=user_id,
                                    startHistoryId=start_history_id,
                                    pageToken=page_token).execute())
    current_historyId=history['historyId']
    changes.extend(history['history'])


print(f'current historyId:{current_historyId}')
for item in changes:
    if message_id in [i['id'] for i in item['messages']]:
        print(item)
```

![message history](/assets/gmail-sniper-forensics/message_history.png)


If you are lucky you can get a fairly good timeline of what happened to a message. Here we see the message added in the UNREAD, SENT and INBOX folders, removing the UNREAD label when it is read and TRASH added when it is deleted. Note the history records do not contain a timestamp. This feature request is currently idling in the Gmail api backlog and would be extremely useful, especially when determining when a user read a message. Feel free to [star/upvote it here](https://issuetracker.google.com/issues/36759772)

## Raw message
Finally to retrieve the raw message itself for further processing, or for evidence you can retrieve the ‘raw’ version and decode it

```Python
# get the raw message, decode it
message_id='171a29788138b18f'
message=gmail.users().messages().get(userId=user_id, format='raw',id=message_id).execute()
mime_message=base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
email_message=email.message_from_bytes(mime_message)
print(email_message.as_string())
```

![raw message](/assets/gmail-sniper-forensics/raw_message.png)


And finally to compare the message contents we retrieved to what was sent

### Sent

![file contents](/assets/gmail-sniper-forensics/file_contents.png)

### Retrieved

![file retrieved](/assets/gmail-sniper-forensics/file_retrieved.png)

## Sniper/Gmail forensics
So there you have it, we were able to set our sights on one particular email message, determine what the user did with it in their inbox and retrieve the metadata and contents of the email without resorting to a mass export of an entire mailbox. I hope this helps you with your future sniper forensic missions when it comes to navigating the gmail api!
