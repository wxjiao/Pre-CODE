# Pre-CODE
Implementation of our paper "Exploiting Unsupervised Data for Emotion Recognition in Conversations" to appear in the Findings of EMNLP 2020.


## Brief Introduction

## Brief Introduction
Emotion Recognition in Conversations (ERC) aims to predict the emotion state of speakers in conversations, which is essentially a text classification task. Unlike the sentence-level text classification problem, the available supervised data for the ERC task is limited, which potentially prevents the models from playing their maximum effect. In this paper, we propose a novel approach to leverage unsupervised conversation data, which is more accessible. Specifically, we propose the Conversation Completion (ConvCom) task, which attempts to select the correct answer from candidate answers to fill a masked utterance in a conversation. Then, we Pre-train a basic COntext-Dependent Encoder (**Pre-CODE**) on the ConvCom task. Finally, we fine-tune the **Pre-CODE** on the datasets of ERC. Experimental results demonstrate that pre-training on unsupervised data achieves significant improvement of performance on the ERC datasets, particularly on the minority emotion classes. 

<div align="center">
    <img src="/image/PreCODE.png" width="80%" title="Framework of Pre-CODE."</img>
    <p class="image-caption">Figure 1: The framework of Pre-CODE.</p>
</div>
