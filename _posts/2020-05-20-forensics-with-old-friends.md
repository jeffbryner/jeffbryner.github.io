---
title:  "Forensics with old friends: Hachoir file carving with Jupyter"
tags: [infosec, forensics, jupyter, hachoir]
author: Jeff
---

## Old Friends
When you do infosec for awhile you gather a collection of 'old friends'; tools you rely on over the years to help you get your job done. Some are simple like dd, or xxd. Some are complex (vsCode?) but it's fun to revisit some old friends in new context to see how they play out.
![a gif we found](/assets/forensics-with-old-friends/a-gif-we-found.gif)

<small>Note to reader: Taylor is not actually an old friend. </small>

## Forensics
I recently had the chance to take [Sara Edwards' SANS course FOR 518 for Mac Forensics](https://www.sans.org/course/mac-and-ios-forensic-analysis-and-incident-response) and it was a blast. A week's worth of digging through Mac/Iphone artifacts culminating with a team challenge to solve a case using what we had learned. It was great fun and brought me back to revisit some core skills.

## File Carving
If you do forensics for any length of time, you'll have occasion to [carve out a file](https://en.wikipedia.org/wiki/File_carving) from a binary blob of nothing. Could be a broken disk, could be a stream intercepted from the network, regardless having an old friend that helps you find and retrieve a file is invaluable.

## Hachoir
Hachoir means 'meat grinder' in French and is the name of a [python project used to parse out structures in all sorts of data](https://hachoir.readthedocs.io/en/latest/). The project was of great help to me over the years when I needed to, oh say [parse out MSTask job files](https://github.com/vstinner/hachoir/blob/master/hachoir/parser/misc/mstask.py#L5) to find malicious entries.

## Jupyter
I decided to revisit this old friend to see if I could whip up a jupyter notebook for file carving making use of the variety of [parsers available in Hachoir](https://hachoir.readthedocs.io/en/latest/parser.html).

## Scenario
In this scenario we've been handed a blob of data and asked to retrieve the last frame of a .gif within the blob. This could have come from a network stream, a bad flash drive, memory, anywhere. To Hachoir it doesn't matter.

## The Notebook
[Lets dig into the notebook for this task](/assets/forensics-with-old-friends/hachoir-inspection-file-carving.ipynb). If you want to play along at home here is the source file, [a blob of who knows what](/assets/forensics-with-old-friends/ablob_of_who_knows_what).

First off, we can get a good look at the parsers imbedded in Hachoir:
![parsers](/assets/forensics-with-old-friends/parserlist.png)

Any chance we just get it right off the bat?
![no joy](/assets/forensics-with-old-friends/nojoy.png)

Do we have a wild guess from the header?
![guess](/assets/forensics-with-old-friends/guess.png)

No luck, ok lets turn to our old friend and see if Hachoir can step through the file and find anything

```Python
# step through the file to see if we can recognize a portion of it
view=io.BytesIO(io.open(target,"rb").read()).getbuffer()

# step through the first x of the file 8 bits at a time looking for recognized files
for x in range(0,4096,8):
    parser=guessParser(InputIOStream(io.BytesIO(view[x::])))
    if parser:
        print(f"{parser} found at position {x}")

```

### Voila:
```
<GifFile path=/, current_size=104, current length=3> found at position 512
```

Now we've got the bytes from our blob in a readable, scannable view thanks to the [Bytes IO library](https://docs.python.org/3/library/io.html#io.BytesIO).

Lets take a peek at this header to validate it's a GIF:
![GIF](/assets/forensics-with-old-friends/gif.png)

Sure enough, Harchoir found it. Now lets see if we can find our way around it and export the last frame.

### Inspection
After re-initing our parser

```Python
parser=guessParser(InputIOStream(io.BytesIO(view[512::])))
```
Lets see the tags Hachoir associated:
![tags](/assets/forensics-with-old-friends/tags.png)

Fields in Hachoir are lazily evaluated, lets take a look at some of what it can find:
![fields](/assets/forensics-with-old-friends/fields.png)

Within the fields there are tree structures for components of the various parts, like the individual frames of the image:
![frames](/assets/forensics-with-old-friends/frames.png)

You can inspect values for individual fields:
![values](/assets/forensics-with-old-friends/values.png)

You can see the image will loop forever with a loop count of 0.

Now normally in file carving, you can count on your old friend dd to grab some bytes like so:

```
dd if=/Users/jeff/work/ablob_of_who_knows_what ibs=1 skip=<number> count=number of=put_it_here
```

Here our task is to grab the last frame and GIF's make our task a bit harder in that each frame can have a local color map, or share a global one. Luckily Hachoir can tell us which one we are dealing with:

![colormap](/assets/forensics-with-old-friends/colormap.png)

So we will need the help of an image processing library. Luckily the [PIL/Pillow library](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html?highlight=loop#fully-supported-formats) can do just the trick.

In the home stretch, we load up the bytestream, count the frames and save out just the last one:

![home stretch](/assets/forensics-with-old-friends/homestretch.png)

Did it work?! Lets use jupyter's embedded image viewer to find out!
![worked](/assets/forensics-with-old-friends/worked.png)

You can find other cells in the notebook to save out the entire gif, change the loop count, etc if that's of interest.

## File Carving: Check!
Thanks to our old friend Hachoir, we now have a handy Jupyter playbook for traversing binary data and pulling out pretty much anything!