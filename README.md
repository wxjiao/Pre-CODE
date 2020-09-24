# Pre-CODE
Implementation of our paper "Exploiting Unsupervised Data for Emotion Recognition in Conversations" to appear in the Findings of EMNLP 2020.


## Brief Introduction

## Brief Introduction
Emotion Recognition in Conversations (ERC) aims to predict the emotion state of speakers in conversations, which is essentially a text classification task. Unlike the sentence-level text classification problem, the available supervised data for the ERC task is limited, which potentially prevents the models from playing their maximum effect. In this paper, we propose a novel approach to leverage unsupervised conversation data, which is more accessible. Specifically, we propose the Conversation Completion (ConvCom) task, which attempts to select the correct answer from candidate answers to fill a masked utterance in a conversation. Then, we Pre-train a basic COntext-Dependent Encoder (**Pre-CODE**) on the ConvCom task. Finally, we fine-tune the **Pre-CODE** on the datasets of ERC. Experimental results demonstrate that pre-training on unsupervised data achieves significant improvement of performance on the ERC datasets, particularly on the minority emotion classes. 

<div align="center">
    <img src="/image/PreCODE.png" width="60%" title="Framework of Pre-CODE."</img>
    <p class="image-caption">Figure 1: The framework of Pre-CODE.</p>
</div>


**Conversation Completion.** We exploit the self-supervision signal in conversations to construct our pre-training task.
Formally, given a conversation, $\mathcal{U}=\{ u_1, u_2, \cdots, u_L \}$, we mask a target utterance $u_l$ as $\mathcal{U}\backslash{u_l}=\{ \cdots, u_{l-1}, [mask] , u_{l+1}, \cdots \}$ to create a question, and try to retrieve the correct utterance $u_l$ from the whole training corpus. Since the choice of filling the mask involves all possible utterances, which are countless, formulating the task into a multi-label classification task with softmax is infeasible. We instead simplify the task into a response selection task~\cite{DBLP:conf/acl/TongZJM17} using negative sampling~\cite{DBLP:conf/nips/MikolovSCCD13}, which is a variant of noise-contrastive estimation~\cite[NCE,][]{gutmann2010noise}.

To achieve so, we sample $N-1$ noise utterances elsewhere, along with the target utterance, to form a set of $N$ candidate answers. Then the goal is to select the correct answer, i.e., $u_l$, from the candidate answers to fill the mask, conditioned on the context utterances. We term this task "Conversation Completion", abbreviated as ConvCom. 
Figure 2 shows an example, where the utterance \texttt{u4} is masked out from the original conversation and \texttt{two} noise utterances are sampled elsewhere together with \texttt{u4} to form the candidate answers. 

<div align="center">
    <img src="/image/ConvCom.png" width="60%" title="An Example of the ConvCom Task."</img>
    <p class="image-caption">Figure 2: An Example of the ConvCom Task.</p>
</div>
