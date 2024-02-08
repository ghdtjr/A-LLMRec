# A-LLMRec : Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System

## Overview
In this paper, we propose an efficient all-round LLM-based recommender system, called A-LLMRec (All-round LLM-based Recommender system). The main idea is to enable an LLM to directly leverage the collaborative knowledge contained in a pre-trained collaborative filtering recommender system (CF-RecSys) so that the emergent ability of the LLM can be jointly exploited. By doing so, A-LLMRec can outperform under the various scenarios including warm/cold, few-shot, cold user, and cross-domain scenarios.

## Dataset
Download [Amazon dataset](https://jmcauley.ucsd.edu/data/amazon/) for the experiment
  
## Pre-train CF-RecSys(SASRec)
```
cd pre_train/sasrec
python main.py --device=cuda --dataset Movies_and_TV --train_dir=default --num_epochs 200 --lr 0.001 --maxlen 50
```

## Train
- pretrain stage1
```
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV
```

- pretrain stage2
```
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV
```

For running with multi-gpu, assign devices with CUDA_VISIBLE_DEVICES command
- ex) CUDA_VISIBLE_DEVICES = 1,2 python main.py
  


## Evaluation

```
python main.py --inference --rec_pre_trained_data Movies_and_TV
python eval.py
```
