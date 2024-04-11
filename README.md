# A Graph Convolution Network with a POS-aware Filter and Context Enhancement

This repository contains the code for ICMR 2024 paper **A Graph Convolution Network with a POS-aware Filter and Context Enhancement Mechanism for Event Detection.** 

You can get the paper from [here](https://doi.org/10.1145/3652583.3658076).

In this paper, we propose a novel graph convolution network with a POS-aware filter and context enhancement mechanism (GCN-PFCE). Specifically, a gating unit controlled by POS, which can learn the correlation between POS and keyword distribution, is added after each graph convolution layer. Besides, a parallel structure between BERT and GCN is implemented to enhance the context understanding ability of GCN-based methods in a better way. The proposed model achieves significant improvement over competitive baseline methods on ACE2005 dataset.

**In order to make the code easy to understand, we are integrating the code. We will try our best to provide the complete source code before the conference starts.**

## Datasets&DataFormat

In this paper, we use ACE2005  to implement all the experiments. It can be download from: The ACE 2005 dataset can be downloaded from：[ACE 2005 Multilingual Training Corpus - Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC2006T06)

The raw ace2005 dataset requires some preprocessing to meet the code's format requirements for data. Due to copyright restrictions, we cannot provide processed data in this repository. However, if you have the license of ACE2005, you can contact me via email to obtain the processed data.

Sample data that conforms to the format is shown in ./data_doc/example_data. The data format is generally based on [EEGCN](https://github.com/cuishiyao96/eegcned), with some minor modifications. One word per line, separate column for token and label, empty line between sentences.

The first column is the token.

The second column is document_id in EEGCN. But it will not be used in the experiments. However, to maintain a data format similar to EEGCN, the second column is set to null uniformly.

The third column is the type of the entity. (BIO schema)

The fourth column is the entity subtype of the entity. (BIO schema)

The fifth column is the event-type label. (BIO schema)

The sixth column is the syntactic dependency label.

The seventh column is index of the syntactic connected token. 

The last column is the part-of-speech annotation of the token.

For example:

```
Even null O O O advmod 8 RB
as null O O O mark 8 IN
the null B-1_PER B-2_Individual O det 3 DT
secretary null I-1_PER I-2_Individual O nsubj 8 NN
of null I-1_PER I-2_Individual O case 6 IN
homeland null B-1_ORG B-2_Government O compound 6 NN
security null I-1_ORG I-2_Government O nmod:of 3 NN
was null O O O aux 8 VBD
putting null O O O advcl:as 29 VBG
his null B-1_PER B-2_Group O nmod:poss 10 PRP$
people null I-1_PER I-2_Group O obj 8 NNS
on null O O O case 13 IN
high null O O O amod 13 JJ
alert null O O O obl:on 8 NN
last null B-1_TIM B-2_time O amod 15 JJ
month null I-1_TIM I-2_time O obl:tmod 8 NN
, null O O O punct 29 ,
a null B-1_VEH B-2_Water O det 23 DT
30 null I-1_VEH I-2_Water O nummod 20 CD
- null I-1_VEH I-2_Water O punct 20 HYPH
foot null I-1_VEH I-2_Water O compound 23 NN
Cuban null B-1_GPE B-2_Nation O amod 22 JJ
patrol null I-1_VEH I-2_Water O compound 23 NN
boat null I-1_VEH I-2_Water O nsubj 29 NN
with null I-1_VEH I-2_Water O case 28 IN
four null B-1_PER B-2_Group O nummod 28 CD
heavily null I-1_PER I-2_Group O advmod 27 RB
armed null I-1_PER I-2_Group O amod 28 JJ
men null I-1_PER I-2_Group O nmod:with 23 NNS
landed null O O B-Movement:Transport ROOT -1 VBD
on null O O O case 32 IN
American null B-1_LOC B-2_Region-General O amod 32 JJ
shores null I-1_LOC I-2_Region-General O obl:on 29 NNS
, null O O O punct 32 ,
utterly null O O O advmod 35 RB
undetected null O O O amod 32 JJ
by null O O O case 41 IN
the null B-1_ORG B-2_Government O det 41 DT
Coast null I-1_ORG I-2_Government O compound 40 NNP
Guard null I-1_ORG I-2_Government O compound 40 NNP
Secretary null B-1_PER B-2_Individual O compound 41 NNP
Ridge null I-1_PER I-2_Individual O obl:by 29 NNP
now null B-1_TIM B-2_time O advmod 29 RB
leads null O O O dep 29 VBZ
. null O O O punct 29 .

He null B-1_PER B-2_Individual O nsubj 1 PRP
lost null O O O ROOT -1 VBD
an null O O O det 3 DT
election null O O B-Personnel:Elect obj 1 NN
to null O O O case 7 IN
a null B-1_PER B-2_Individual O det 7 DT
dead null I-1_PER I-2_Individual O amod 7 JJ
man null I-1_PER I-2_Individual O obl:to 1 NN
. null O O O punct 1 .
```

## Requirements

Main requirements:

torch==1.9.0

python==3.8.10

More requirements are shown in requirements.txt

## Train&Test

The proposed model is trained in three stages.(We will provide specific running commands later.)

1、Finetune bert:

2、Train the other components in the model except for the context enhancement module:

3、The complete model is initialized with the parameters obtained in the first two
stages and undergoes 20 epochs of training:

Test:

## Computation Consumption

The hardware environment we used is Intel(R) Xeon(R) Gold 5320 CPU @ 2.20GHz + 1/2 NVIDIA A30. The complete training process takes approximately 25 minutes.

If a complete training process is required, please ensure that the graphics memory of the video card is greater than 12GB.

## Related Repository

This code is an improvement mainly based on [EEGCN](https://github.com/cuishiyao96/eegcned), and I would like to extend special thanks to [EEGCN](https://github.com/cuishiyao96/eegcned).
