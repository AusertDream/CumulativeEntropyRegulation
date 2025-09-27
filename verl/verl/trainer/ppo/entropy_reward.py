# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
from functools import partial
import torch
import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def grpo_entropy_reward_fn(
    data: DataProto,
    tokenizer,
    entropy_bonus_coef: float = 0.01,
    logprob_penalty_coef: float = 0.001,
    **kwargs,
):
    """
    Args:
        data (DataProto): 包含所有 rollout 信息的 DataProto 对象。
        tokenizer: 分词器实例。
        entropy_bonus_coef (float): 熵奖励的系数。
        logprob_penalty_coef (float): 对数概率惩罚的系数。

    Returns:
        dict: 一个包含 "reward_tensor" 和 "reward_extra_info" 的字典。
    """
    print("dataproto: ", data)
    print("get the accurate reward. ")
    print("entropys: ", data.batch.get("entropys"))
    

    return {"reward_tensor": torch.ones(data.batch.get("responses").shape[0]), "reward_extra_info": {}}


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence."""
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config):
    """Load and return a custom reward function from external file."""
    custom_reward_config = config.reward_model.get("custom_reward_function")
    if custom_reward_config is None:
        raise ValueError("custom_reward_function is not specified in the config")

    path = custom_reward_config.get("path")
    if path is None:
        raise ValueError("path is not specified in custom_reward_function config")

    fn_name = custom_reward_config.get("fn_name")
    if fn_name is None:
        raise ValueError("fn_name is not specified in custom_reward_function config")

    extra_kwargs = custom_reward_config.get("kwargs", {})
    fn = dynamic_import(fn_name, path)
    return partial(_call_with_kwargs, fn, extra_kwargs)


def load_reward_manager(config, tokenizer, **kwargs):
    """
    Load reward manager based on config.
    """
    # 注册所有可用的 reward function
    REWARD_REGISTRY = {
        "default": default_compute_score,
        "dapo": default_compute_score, 
        "grpo_with_entropy": grpo_entropy_reward_fn, 
    }

    # use entropy reward
    reward_fn = REWARD_REGISTRY["grpo_with_entropy"]
    reward_kwargs = {
        "tokenizer": tokenizer,
        **kwargs,
        **dict(config.reward_model.get("kwargs", {})),
    }
    
    # 使用 partial 将固定的参数（如 tokenizer 和配置中的超参数）绑定到 reward_fn
    return partial(
        reward_fn,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception:
        # 兼容不返回 dict 的旧版 reward_fn
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )
        import warnings

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0,
        )
    return compute_reward(data, reward_fn)