# A-LLMRec : Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System

The source code for A-LLMRec : Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System paper, accepted at **KDD 2024**.

## Overview
In this [paper](https://arxiv.org/abs/2404.11343), we propose an efficient all-round LLM-based recommender system, called A-LLMRec (All-round LLM-based Recommender system). The main idea is to enable an LLM to directly leverage the collaborative knowledge contained in a pre-trained collaborative filtering recommender system (CF-RecSys) so that the emergent ability of the LLM can be jointly exploited. By doing so, A-LLMRec can outperform under the various scenarios including warm/cold, few-shot, cold user, and cross-domain scenarios.

## Env Setting
```
conda create -n [env name] python=3.10 pip
conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy=1.26.3
conda install tqdm
conda install pytz
conda install transformers=4.32.1
pip install sentence-transformers==2.2.2
conda install conda-forge::accelerate=0.25.0
conda install conda-forge::bitsandbytes=0.42.0
```

## Dataset
Download [dataset of 2018 Amazon Review dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for the experiment. Should download metadata and reviews files and place them into data/amazon direcotory.

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

To run with multi-GPU setting, assign devices using the CUDA_VISIBLE_DEVICES command and add '--multi_gpu' argument.
- ex) CUDA_VISIBLE_DEVICES = 0,1 python main.py ... --multi_gpu
  


## Evaluation
Inference stage generates "recommendation_output.txt" file and writes the recommendation result generated from the LLMs into the file. To evaluate the result, run the eval.py file.

```
python main.py --inference --rec_pre_trained_data Movies_and_TV
python eval.py
```
