# Build Your Own Chatbot in Python

- [Build Your Own Chatbot in Python](#build-your-own-chatbot-in-python)
  - [Implementing a Chatbot](#implementing-a-chatbot)
    - [Prerequisites](#prerequisites)
    - [Features](#features)
    - [Explanation](#explanation)
  - [Training your House AI](#training-your-house-ai)


Some part of this lesson will not be summarized because it intersects with other lesson I've previously covered.

## Implementing a Chatbot

It is a way to use an AI that is similar to simply be talking to a human.

### Prerequisites  

It is important to know what type of chatbot we are building. We want either one that:
- Rule-based: 
- Self-learning: save and process old interaction with human.
- Mix of both

It is also important to know the target audience to make our chatbot better at performing a specific task. We also need to answer appropriate answers.

### Features

We will use `chatterbot` library. Our chatbot will have a specific domain knowledge like any other chatbot. We will encounter wrong answers so we need to treat them accordingly. 

We also want our chatbot to have human-like behavior such as:
- Empathy
- Emotions
- Intelligence

There is a simple example [here](Code_example/main.py). **NOTE**: because it requires CPP library and the installer seems to hate my computer, I am going to run the program on [google colab](https://colab.research.google.com/drive/10ynxm4PpEfCteNRQo2PMQJ9kRWUTQvjX?usp=sharing). **NOTE NOTE**: I didn't find any ways to run the program on the cloud or on both of my local machine.

### Explanation

1. We first create the chatbot with `create_bot`. It takes a name as argument.
2. Training: done with `train_all_data`.
3. Training custom data: we let it run on our data. First argument is the function of the bot itself. Then we feed as second argument the data.
4. Start: we simply use `start_chatbot`.

## Training your House AI

We need to feed our own data to identify ourself into the AI.
