# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.utils.reward_score import default_compute_score
import torch
from collections import defaultdict, Counter
import math
from verl.utils.reward_score.prime_math import grade_answer
from verl.utils.reward_score.prime_math.grader import math_equal
from verl.workers.reward_manager import register
from typing import List

@register("CumulativeEntropyRegulation")
class EntropyConstraintRewardManager:
    """The reward manager.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len


    def _last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1

        if left_brace_idx is None or right_brace_idx is None:
            return None

        return string[left_brace_idx + 1 : right_brace_idx].strip()

    

    def accuracy_reward_fn(self, model_answer: str, ground_truth: int) -> int:
        model_answer_only = self._last_boxed_only_string(model_answer)
        
        if model_answer_only == None or model_answer_only.strip() != ground_truth.strip():
            return 0
        else:
            return 1
    
    def entropy_reward_fn(self, entropy: List[int]) -> int:
        final_step_entropy = entropy[-1]
        entropy_reward = torch.exp(-final_step_entropy)
        return entropy_reward + 1
    
    def mix_all_reward_fn(self, rewards: List[int], weights: List[int] = None) -> int:
        if weights == None:
            weights = [1 for i in range(len(rewards))]
        reward_with_weight = [a*b for a, b in zip(rewards, weights)] 
        final_reward: int = 0
        for reward in reward_with_weight:
            final_reward+=reward
        
        return final_reward / sum(weights)


    def __call__(self, data: DataProto, return_dict: bool = False, is_valid: bool = False):
        """We will expand this function gradually based on the available datasets"""
        reward_extra_info = defaultdict(list)
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            
            
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            # print("ground_truth is: ", ground_truth)
            # print("model answer:", response_str)
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            # get accuracy reward
            entropy_reward=0
            accuracy_reward = self.accuracy_reward_fn(response_str, ground_truth)
            avg_entropys = []
            # calculate the entropy reward only if it is correct
            if accuracy_reward == 1:
                # get token_entropys
                entropys = data_item.batch['entropys'][:valid_response_length]
                for j in range(entropys.shape[0]):
                    if j == 0:
                        avg_entropys.append(entropys[0])
                    else:
                        avg_entropys.append(avg_entropys[j-1] + entropys[j])
                # get the avg_step_entropy
                for j in range(len(avg_entropys)):
                    avg_entropys[j]=avg_entropys[j]/(j+1)
                
                # get the entropy reward
                entropy_reward = self.entropy_reward_fn(avg_entropys)
                # mix all the reward
                final_reward = self.mix_all_reward_fn([accuracy_reward, entropy_reward])
            else:
                final_reward = accuracy_reward
            
            reward_tensor[i, valid_response_length - 1] = final_reward

    
            reward_extra_info['entropy_reward'].append(float(entropy_reward))
            reward_extra_info['accuracy_reward'].append(float(accuracy_reward))
            reward_extra_info['final_reward'].append(float(final_reward))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
