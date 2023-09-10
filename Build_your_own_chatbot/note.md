# Build Your Own Chatbot in Python

- [Build Your Own Chatbot in Python](#build-your-own-chatbot-in-python)
  - [Implementing a Chatbot](#implementing-a-chatbot)
    - [Prerequisites](#prerequisites)
    - [Features](#features)


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

There is a simple example [here](Code_example/main.py)