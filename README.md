# Pre-CODE: Exploiting Unsupervised Data for Emotion Recognition in Conversations

Implementation of our paper "Exploiting Unsupervised Data for Emotion Recognition in Conversations" to appear in the Findings of EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.findings-emnlp.435/) [[old version]](https://arxiv.org/abs/1910.08916)


## Brief Introduction

Emotion Recognition in Conversations (ERC) aims to predict the emotion state of speakers in conversations, which is essentially a text classification task. Unlike the sentence-level text classification problem, the available supervised data for the ERC task is limited, which potentially prevents the models from playing their maximum effect. In this paper, we propose a novel approach to leverage unsupervised conversation data, which is more accessible. Specifically, we propose the Conversation Completion (ConvCom) task, which attempts to select the correct answer from candidate answers to fill a masked utterance in a conversation. Then, we Pre-train a basic COntext-Dependent Encoder (**Pre-CODE**) on the ConvCom task. Finally, we fine-tune the **Pre-CODE** on the datasets of ERC. Experimental results demonstrate that pre-training on unsupervised data achieves significant improvement of performance on the ERC datasets, particularly on the minority emotion classes. 

<div align="center">
    <img src="/image/PreCODE.png" width="60%" title="Framework of Pre-CODE."</img>
    <p class="image-caption">Figure 1: The framework of Pre-CODE.</p>
</div>


**Conversation Completion (ConvCom)** We exploit the self-supervision signal in conversations to construct our pre-training task.
Formally, given a conversation, U={ u<sub>1</sub>, u<sub>2</sub>, ..., u<sub>L</sub> }, we mask a target utterance u<sub>l</sub> as U\u<sub></sub>={ ..., u<sub>l-1</sub>, [mask] , u<sub>l+1</sub>, ... } to create a question, and try to retrieve the correct utterance u<sub>l</sub> from the whole training corpus. Since the choice of filling the mask involves all possible utterances, which are countless, formulating the task into a multi-label classification task with softmax is infeasible. We instead simplify the task into a response selection task using negative sampling, which is a variant of noise-contrastive estimation (NCE).
To achieve so, we sample N-1 noise utterances elsewhere, along with the target utterance, to form a set of N candidate answers. Then the goal is to select the correct answer, i.e., u<sub>l</sub>, from the candidate answers to fill the mask, conditioned on the context utterances. We term this task "Conversation Completion", abbreviated as ConvCom. 
Figure 2 shows an example, where the utterance u<sub>4</sub> is masked out from the original conversation and _two_ noise utterances are sampled elsewhere together with u<sub>4</sub> to form the candidate answers. 

<div align="center">
    <img src="/image/ConvCom.png" width="60%" title="An Example of the ConvCom Task."</img>
    <p class="image-caption">Figure 2: An Example of the ConvCom Task.</p>
</div>


## Code Base


### Dataset
Please find the datasets via the following links:
  - [Friends](http://doraemon.iis.sinica.edu.tw/emotionlines): **Friends** comes from the transcripts of Friends TV Sitcom, where each dialogue in the dataset consists of a scene of multiple speakers.
  - [EmotionPush](http://doraemon.iis.sinica.edu.tw/emotionlines): **EmotionPush** comes from private conversations between friends on the Facebook messenger collected by an App called EmotionPush.
  - [IEMOCAP](https://sail.usc.edu/iemocap/): **IEMOCAP** contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions.
  
### Prerequisites
- Python v3.6
- Pytorch v0.4.0-v0.4.1
- Pickle

### Data Preprocessing
Preprocess the OpenSubtitle dataset as:
```ruby
python Preprocess.py -datatype opsub -min_count 2 -max_seq_len 60
```

Preprocess one of the emotion dataset as:
```ruby
python Preprocess.py -datatype emo -emoset Friends -min_count 2 -max_seq_len 60
```
The arguments `-datatype`, `-emoset`, `-min_count`, and `-max_length` represent the type of data (i.e., pre-training data or emotion data), the dataset name, the minimum frequency of words when building the vocabulary, and the max_length for padding or truncating sentences.

[PreCODE storage](https://drive.google.com/drive/folders/1wiafGIdBdV2F9bUjdwZdAUJ3V3iIcmg7?usp=sharing) 
includes the raw data and preprocessed data of OpSub and Friends, and the pre-trained models with hidden sizes of 300.

### Pre-trained Word Embeddings
To reproduce the results reported in the paper, please adopt the pre-trained word embeddings for initialization. You can download the 300-dimentional embeddings from below:
- GloVe: [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/)

Decompress the file and re-name it `glove300.txt`.

### Train
1. Pre-train the context-dependent encoder on the ConvCom task.
```ruby
bash exec_src.sh
```
   - You can change the parameters in the script.
```ruby
#!bin/bash
# Var assignment
LR=2e-4
GPU=3
echo ========= lr=$LR ==============
for iter in 1
do
echo --- $Enc - $Dec $iter ---
python LMMain.py \
-lr $LR \
-gpu $GPU \
-d_hidden_low 300 \
-d_hidden_up 300 \
-sentEnc gru2 \
-layers 1 \
-patience 3 \
-data_path OpSub_data.pt \
-vocab_path glob_vocab.pt \
-embedding embedding.pt \
-dataset OpSub
done
```

2. Fine-tune the Pre-CODE on the emotion datasets.
```ruby
bash exec_emo.sh
```
  - You can change the parameters in the script.
```ruby
#!bin/bash
# Var assignment
LR=1e-4
GPU=1
du=300
dc=300
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5
do
echo --- $Enc - $Dec $iter ---
python EmoMain.py -load_model \
-lr $LR -gpu $GPU \
-d_hidden_low $du -d_hidden_up $dc \
-patience 6 -report_loss 720 \
-data_path Friends_data.pt \
-vocab_path glob_vocab.pt \
-emodict_path Friends_emodict.pt \
-tr_emodict_path Friends_tr_emodict.pt \
-dataset Friends \
-embedding embedding.pt
done
```

## Public Impact

### Citation
Please kindly cite our paper if you find it useful or highly related to your research:

```ruby
@misc{jiao2020exploiting,
      title={Exploiting Unsupervised Data for Emotion Recognition in Conversations}, 
      author={Wenxiang Jiao and Michael R. Lyu and Irwin King},
      year={2020},
      eprint={2010.01908},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

