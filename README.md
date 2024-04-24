# A-LLMRec : Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System

## Overview
In this paper, we propose an efficient all-round LLM-based recommender system, called A-LLMRec (All-round LLM-based Recommender system). The main idea is to enable an LLM to directly leverage the collaborative knowledge contained in a pre-trained collaborative filtering recommender system (CF-RecSys) so that the emergent ability of the LLM can be jointly exploited. By doing so, A-LLMRec can outperform under the various scenarios including warm/cold, few-shot, cold user, and cross-domain scenarios.

## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Dataset
Download [Amazon dataset](https://jmcauley.ucsd.edu/data/amazon/) for the experiment. Should download metadata and reviews files and place them into data/amazon direcotory.
  
## Pre-train CF-RecSys (SASRec)
```
cd pre_train/sasrec
python main.py --device=cuda --dataset Movies_and_TV
```

## A-LLMRec Train
- train stage1
```
cd ../../
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV
```

- train stage2
```
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV
```

To run with multi-gpu setting, assign devices with CUDA_VISIBLE_DEVICES command and add '--multi_gpu' argument.
- ex) CUDA_VISIBLE_DEVICES = 0,1 python main.py ... --multi_gpu
  


## Evaluation
Inference stage generates "recommendation_output.txt" file and write the recommendation result generated from the LLMs into the file. 

```
python main.py --inference --rec_pre_trained_data Movies_and_TV
python eval.py
```
