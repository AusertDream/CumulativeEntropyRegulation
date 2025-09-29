# Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation
-----------------

[Paper](https://github.com/LWL-cpu/Question-Free-Fine-Tuning) [Model](#model) [Dataset](#dataset)

## Introduction

Welcome to the official repository for **Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation**.





## Start up













对verl的修改
新建entropy reward manager，使用这个计算reward

修改数据流，在计算old_log_probs的时候不会把对应的entropy pop掉，同时dataproto里面新增logits，内容。

涉及修改的文件: ec_manager.py, main_ppo.py, fsdp_workers.py, dp_actor.py, ray_trainer.py



## Models

<a name="model"></a> 

| Model Name                | Base LLM              | HF Link                                  | Modelscope Link                          |
| ------------------------- | --------------------- | ---------------------------------------- | ---------------------------------------- |
| Qwen3-4B-CER              | Qwen3-4B              | [link](https://huggingface.co/Ausert/Qwen3-4B-CER) | [link](https://www.modelscope.cn/models/ausertdream/Qwen3-4B-CER) |
| Qwen3-8B-CER              | Qwen3-8B              | [link](https://huggingface.co/Ausert/Qwen3-8B-CER) | [link](https://www.modelscope.cn/models/ausertdream/Qwen3-8B-CER) |
| R1-distill-Qwen2.5-7B-CER | R1-distill-Qwen2.5-7B | [link](https://huggingface.co/Ausert/R1-distill-Qwen2.5-7B-CER) | [link](https://www.modelscope.cn/models/ausertdream/R1-distill-qwen2.5-7B-CER) |



## Dataset

<a name="dataset"></a>

| Dataset | HF Link                                  | Modelscope Link                          |
| ------- | ---------------------------------------- | ---------------------------------------- |
| GSM8K   | [link](https://huggingface.co/datasets/openai/gsm8k) | [link](https://www.modelscope.cn/datasets/modelscope/gsm8k) |
| MATH500 | [link](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | [link](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500/summary) |
| AIME24  | [link](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) | [link](https://www.modelscope.cn/datasets/AI-ModelScope/AIME_2024) |
| AIME25  | [link](https://huggingface.co/datasets/math-ai/aime25) | [link](https://www.modelscope.cn/datasets/TIGER-Lab/AIME25/files) |

