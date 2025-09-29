# Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation
-----------------

[Paper](https://github.com/LWL-cpu/Question-Free-Fine-Tuning) [Model](#model) [Dataset](#dataset)

## Introduction

Welcome to the official repository for **Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation**.

我们的论文从熵的角度探究了模型为什么overthinking，根据token entropy定义了**T**oken **E**ntropy **C**umulative **A**verage(**TECA**)，他是token entropy的移动累计平均。我们同时提出了一种新的思考范式——"Explore Briefly, Then Decide"，我们认为模型目前的overthinking是由于在探索过程中对答案的不确定导致的，因此提出简洁探索，然后探索到答案后就确定的思考范式。我们通过将TECA的终值纳入强化学习的训练reward，得到的模型能够有效减缓overthinking问题。具体的论证与分析见论文。

We open-sourced our models, data, and code here.

## Environment

我们训练主要使用到了VERL训练框架，因此环境配置跟随VERL[官方配置教程](https://verl.readthedocs.io/en/latest/start/install.html)即可。以下为简化过程。

```
conda create --name CER python=3.10
conda activate CER

# only support running with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# install verl
cd verl
pip install --no-deps -e .
```

建议使用大显存GPU，因为在compute logprobs的时候，因为需要留下entropy作为后续的奖励计算，所以在compute logprobs阶段特别容易OOM。

## Models

<a name="model"></a> 

我们训练了Qwen3-4B和Qwen3-8B这两个模型，数据集使用了GSM8K的训练集部分，下面给出了huggingface和modelscope对应的开源模型链接。

|  Model Name  | Base LLM |                 HF Link                  |             Modelscope Link              |
| :----------: | :------: | :--------------------------------------: | :--------------------------------------: |
| Qwen3-4B-CER | Qwen3-4B | [link](https://huggingface.co/Ausert/Qwen3-4B-CER) | [link](https://www.modelscope.cn/models/ausertdream/Qwen3-4B-CER) |
| Qwen3-8B-CER | Qwen3-8B | [link](https://huggingface.co/Ausert/Qwen3-8B-CER) | [link](https://www.modelscope.cn/models/ausertdream/Qwen3-8B-CER) |


## Dataset

<a name="dataset"></a>

数据集使用GSM8K训练集部分用于训练，其他数据集均用于评估。

| Dataset |                 HF Link                  |             Modelscope Link              |
| :-----: | :--------------------------------------: | :--------------------------------------: |
|  GSM8K  | [link](https://huggingface.co/datasets/openai/gsm8k) | [link](https://www.modelscope.cn/datasets/modelscope/gsm8k) |
| MATH500 | [link](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | [link](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500/summary) |
| AIME24  | [link](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) | [link](https://www.modelscope.cn/datasets/AI-ModelScope/AIME_2024) |
| AIME25  | [link](https://huggingface.co/datasets/math-ai/aime25) | [link](https://www.modelscope.cn/datasets/TIGER-Lab/AIME25/files) |

## Training

### Getting Started

在对应的shell文件中填好对应超参数，以及模型位置，checkpoints位置，数据集位置。之后运行以下指令进行训练。

```
cd verl
bash examples/grpo_trainer/run_gsm8k_lora.sh
```



### Our Modification

为了实现能够在计算奖励的时候考虑TECA，也就需要将token entropy带入dataproto对象用于reward计算，我们对verl的源码以及dataproto数据流进行了修改。

1. 首先新增了`cer_manager.py`，他是全新的能够计算TECA奖励的manager。
2. 修改了dataproto数据流，确保从训练开始到计算reward结束，整个过程entropy能够正确的被传输，涉及修改文件如下：` main_ppo.py, fsdp_workers.py, dp_actor.py, ray_trainer.py` 。


## Evaluation

相关evaluation代码在src文件夹中，同时其中也有一些工具文件，比如自动化合并lora adapter和base model，使用matplotlib画图的代码，统计测评结果的代码等。

```
merge_model.py  -> 合并lora adapter和base model
plot_func.py  -> 画图函数文件
show_res.py  -> 显示测评结果
test_transformers.py  -> 基于transformers库实现的测评代码，会自动在evaluation的过程中收集token entropy以及TECA.
```

想要运行这些代码，在文件中修改对应的path路径，然后直接`python file.py`即可。



## Results

**表格1：GSM8K 和 MATH500 性能对比**

|   **Method**   |       | **GSM8K** |             |       | **MATH500** |             |
| :------------: | :---: | :-------: | :---------: | :---: | :---------: | :---------: |
|                |  ACC  |    LEN    | $\Delta$LEN |  ACC  |     LEN     | $\Delta$LEN |
|  **Qwen3-4B**  |       |           |             |       |             |             |
|   w thinking   | 92.80 |  1348.59  |      -      | 65.20 |   4458.60   |      -      |
|  w/o thinking  | 86.50 |  260.96   |   80.65%    | 61.20 |   846.35    |   81.02%    |
|      CoD       | 93.30 |  385.50   |   71.41%    | 52.60 |   1159.73   |   73.99%    |
|      CCoT      | 82.56 |  616.42   |   54.29%    | 64.00 |   2401.94   |   46.13%    |
| **CER (ours)** | 94.09 |  391.08   |   71.00%    | 64.80 |   2708.65   |   39.25%    |
|  **Qwen3-8B**  |       |           |             |       |             |             |
|   w thinking   | 94.62 |  1491.38  |      -      | 65.80 |   4669.74   |      -      |
|  w/o thinking  | 88.86 |  272.02   |   79.83%    | 59.00 |   837.16    |   81.22%    |
|      CoD       | 94.40 |  415.80   |   72.12%    | 60.80 |   1391.22   |   70.21%    |
|      CCoT      | 92.49 |  739.05   |   50.45%    | 65.20 |   2761.19   |   40.87%    |
| **CER (ours)** | 92.57 |  668.06   |   55.21%    | 65.80 |   3140.04   |   32.76%    |

---

**表格2：AIME24 和 AIME25 性能对比**

|   **Method**   |       | **AIME24** |             | **AIME25** |          |             | **Average** |         |
| :------------: | :---: | :--------: | :---------: | :--------: | :------: | :---------: | :---------: | :-----: |
|                |  ACC  |    LEN     | $\Delta$LEN |    ACC     |   LEN    | $\Delta$LEN |     ACC     |   LEN   |
|  **Qwen3-4B**  |       |            |             |            |          |             |             |         |
|   w thinking   | 64.44 |  11343.57  |      -      |   48.89    | 12119.62 |      -      |    67.83    | 7317.59 |
|  w/o thinking  | 26.67 |   2132.2   |   78.34%    |   20.00    | 2503.93  |   82.84%    |    48.59    | 1435.86 |
|      CoD       | 23.30 |  3607.83   |   68.19%    |   26.70    | 3535.23  |   70.83%    |    48.98    | 2172.07 |
|      CCoT      | 56.67 |  9491.87   |   16.32%    |   40.00    | 10775.93 |   11.09%    |    60.81    | 5821.54 |
| **CER (ours)** | 61.11 |  9215.77   |   18.76%    |   51.11    | 9565.64  |   21.07%    |    67.78    | 5470.29 |
|  **Qwen3-8B**  |       |            |             |            |          |             |             |         |
|   w thinking   | 63.33 |  11247.68  |      -      |   46.67    | 12708.16 |      -      |    67.60    | 7529.24 |
|  w/o thinking  | 20.00 |  2399.13   |   80.10%    |   23.33    | 2300.13  |   80.69%    |    47.80    | 1452.11 |
|      CoD       | 20.00 |  3657.57   |   67.48%    |   20.00    | 3709.20  |   70.81%    |    48.80    | 2293.45 |
|      CCoT      | 63.33 |  9286.63   |   17.44%    |   53.33    | 10438.07 |   17.86%    |    68.59    | 5806.23 |
| **CER (ours)** | 65.56 |  9171.56   |   18.46%    |   53.33    | 9894.51  |   22.14%    |    69.32    | 5718.54 |




## Citation

```
Coming Soon...
```

