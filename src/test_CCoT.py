from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset
import os
from tqdm import tqdm
import sys
import json
import collections
import matplotlib.pyplot as plt
from scipy.stats import entropy
from accelerate.utils import set_seed
import wandb
import gc 
import multiprocessing
import time


plot_path = '/path/to//src/plots'


def get_entropy(text: str) -> float:
  counts = collections.Counter(text)
  probabilities = [count / len(text) for count in counts.values()]
  shannon_entropy = entropy(probabilities, base=2)
  
  return shannon_entropy

def save_to_json_partly(data: dict, file_path: str, num: int, max_num: int) -> None:
    # For the first item, start with an opening bracket and append the data
    if num == 1:
        with open(file_path, 'w') as f:
            f.write('[\n')
            json.dump(data, f, ensure_ascii=False)
    # For the last item, append data and close the list with a bracket
    elif num == max_num:
        with open(file_path, 'a') as f:
            f.write(',\n')
            json.dump(data, f, ensure_ascii=False)
            f.write('\n]')
    # For items in the middle, just append the data with a comma
    else:
        with open(file_path, 'a') as f:
            f.write(',\n')
            json.dump(data, f, ensure_ascii=False)

def main(model_path, model_name, data_source, dataset_path, wandb_mode='online', project_name='test', name=None, save_to=None, enable_thinking=True, start=0, end=100000000000000000):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    micro_bs = int(1)
    max_num = 10000000
    eff_bs = 2
    accum_steps = max(1, eff_bs // (micro_bs * world_size))
    temp = 0.8
    lr = 5e-5

    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model.eval()
    if wandb_mode != 'off':
        wandb.init(
            project=project_name,
            name="entropy graph " + model_name + ' Q1_zh',
            config={
                "model": model_path,
                "temperature": temp,
                "learning_rate": lr,
                "batch_size": eff_bs,
                "accumulation_steps": accum_steps,
                "world_size": world_size
            },
            mode=wandb_mode,
        )
    
    system_prompt = """You are an intelligent assistant.
Your task is to answer the following multiple-choice questions.
Think step-by-step through the problem to ensure you have the correct answer.
Then, answer the question using the following format 'Action: Answer("[choice]")'  
The parameter [choice] is the letter or number of the answer you want to select (e.g. "A", "B", "C", or "D")
For example, 'Answer("C")' will select choice "C" as the best answer.
If no choice to choose, put the step answer in 'Answer("number")'.
Be concise. At last, make sure that put your final answer in $\\boxed{}$"""

    if data_source == 'gsm8k':
        ds = load_dataset('parquet', data_files=dataset_path, split='train')
        questions = [item['prompt'][0]['content'] for item in ds]
        answers_only = [item['reward_model']['ground_truth'] for item in ds]
    elif data_source == 'math500':
        ds = load_dataset('json', data_files=dataset_path, split='train')
        questions = [item['problem'] for item in ds]
        answers_only = [item['answer'] for item in ds]
    elif data_source == 'aime24':
        ds = load_dataset('parquet', data_files=dataset_path, split='train')
        questions = [item['Problem'] for item in ds]
        answers_only = [str(item['Answer']) for item in ds]
    elif data_source == 'aime25':
        ds = load_dataset('json', data_files=dataset_path, split='train')
        questions = [item['question'] for item in ds]
        answers_only = [item['answer'] for item in ds]
    elif data_source == 'omni-math':
        ds = load_dataset('json', data_files=dataset_path, split='train')
        questions = [item['problem'] for item in ds]
        answers_only = [item['answer'] for item in ds]
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    is_thinking = 'thinking' if enable_thinking else 'nothinking'
    save_entropy_file_name = data_source + '_' + model_name + '_' + is_thinking + '.json'
    if save_to is not None:
        save_entropy_file_name = os.path.join(save_to, save_entropy_file_name)

    print(f"Dataset {data_source} loaded successfully with {len(ds)} samples.")
    print("process range:", start, min(end, len(ds)))
    # 模型回答一整个 数据集。
    for j in tqdm(range(start, min(end, len(ds))), desc=f"{model_name} is answering"):
        step_entropy = []
        question = questions[j]
        ground_truth = answers_only[j]
        CCoT_prompt = """Question: What is the capital of the state where Johns Hopkins University is located?
Choices:
  A: Baltimore
  B: Annapolis
  C: Des Moines
  D: Las Vegas
  Answer: Thought: 
  Johns Hopkins University is located in Baltimore, Maryland.
  The capital of Maryland is Annapolis.
Action: Answer("B"). So finally, the answer is $\\boxed{B}$. \n
  """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": CCoT_prompt + question}
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        print("thinking status: ", enable_thinking)
    #     answer_1 = """Okay, let me try to figure out this problem. So Natalia sold clips to 48 friends in April, and then in May she sold half as many clips. The question is asking how many clips she sold altogether in April and May.

    # First, I need to make sure I understand the numbers here. In April, it says she sold clips to 48 friends. Does that mean each friend bought one clip or multiple clips? """
    #     rethink_last = """But again, the problem does not give us enough data. Hence, the standard interpretation must be taken.

    # In most math problems, similar statements are treated as direct equivalences. For example, if a problem states "She gave cookies to 10 children", unless specified otherwise, the count refers to the number of cookies being same as the number of children, implying one cookie per child. Similarly, if someone sells something to a certain number of people, the quantity sold corresponds numerically to the number of people if not further qualified."""
    #     inputs = inputs + answer_1 + '\n' + rethink_last
        enc = tokenizer(inputs, return_tensors='pt').to(model.device)
        
        
        with torch.no_grad():
            gen_ids = model.generate(**enc, 
                                    max_new_tokens=16384, 
                                    do_sample=True, 
                                    top_p=0.95, 
                                    temperature=temp,  
                                    repetition_penalty=1.15,
                                    pad_token_id=tokenizer.pad_token_id, 
                                    use_cache=False)
                
            seq = torch.cat([enc.input_ids, gen_ids[:, enc.input_ids.shape[1]:]], dim=1)[:, :16384]
            pad_mask = seq.ne(tokenizer.pad_token_id)
            prompt_len = pad_mask[:, :enc.input_ids.shape[1]].sum(-1)
            token_idx = torch.arange(seq.size(1), device=seq.device)
            gen_mask = (token_idx.unsqueeze(0) >= prompt_len.unsqueeze(1)) & pad_mask
            wandb_data = []
            token_entropys = []
            
            if False:
                logits = model(seq, attention_mask=pad_mask).logits
                
                probs = F.softmax(logits / temp, dim=-1)
                
                for i in range(enc.input_ids.shape[1], seq.size(1)):
                    probs_i = probs[:, :i]
                    gen_mask_i = gen_mask[:, :i]
                    H_tok = -(probs_i * torch.log(probs_i + 1e-12)).sum(-1)
                    if gen_mask_i[:, i-1] == True:
                        token_entropys.append(H_tok[:,i-1].item())
                    loss = (H_tok * gen_mask_i).sum() / gen_mask_i.sum().clamp_min(1)
                    if wandb_mode != 'off':
                        wandb.log({"entropy": loss.item(), "step": i})
                    step_entropy.append(loss.item())
                    wandb_data.append([i, loss.item(), seq[0, i].item(), tokenizer.decode(seq[0, i].item())])

                loss_table = wandb.Table(data=wandb_data, columns=["step", "entropy", "token_id", "token"])
            model_answer = tokenizer.decode(gen_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            response_length = len(gen_ids[0, enc.input_ids.shape[1]:])
            model_answer_only = model_answer.split('\\boxed{')[-1].split('}')[0] if '\\boxed{' in model_answer else model_answer
            saved_data = {
                "question": question,
                "token_entropys": token_entropys,
                "step_entropys": step_entropy,
                "model_name": model_name,
                "ground_truth": ground_truth,
                "model_answer_only": model_answer_only,
                "model_answer": model_answer,
                "response_length": response_length,
                'is_correct': model_answer_only == ground_truth
            }
            save_to_json_partly(saved_data, save_entropy_file_name, j+1, min(max_num, len(ds)))
            # print("model answer", tokenizer.decode(gen_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True))

    # 保存回答及结果
    # with open(save_entropy_file_name, 'w') as f:
    #     json.dump(json_entropy_res, f, ensure_ascii=False, indent=4)
    print(f"Entropy results saved to {save_entropy_file_name}")
    if wandb_mode != 'off':
        wandb.finish()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    print("cuda used: ", os.environ['CUDA_VISIBLE_DEVICES'])
    models_path = '/path/to/models'
    model_list = os.listdir(models_path)
    model_list = ['qwen3_8B']
    dataset_path = '/path/to/datasets'
    dataset_list = os.listdir(dataset_path)
    dataset_list = ['aime24'] 
    error_list = []
    output_dir = '/path/to/outputs'
    for i in range(len(model_list)):
        for dataset in dataset_list:
            if 'gsm8k' in dataset:
                data_source = 'gsm8k'
                dataset_file_path = os.path.join(dataset_path, 'gsm8k', 'test.parquet')
            elif 'math500' in dataset:
                data_source = 'math500'
                dataset_file_path = os.path.join(dataset_path, 'math500', 'test.jsonl')
            elif 'aime24' in dataset:
                data_source = 'aime24'
                dataset_file_path = os.path.join(dataset_path, 'aime24', 'aime_2024_problems.parquet')
            elif 'aime25' in dataset:
                data_source = 'aime25'
                dataset_file_path = os.path.join(dataset_path, 'aime25', 'aime2025.jsonl')
            elif 'omni-math' in dataset:
                data_source = 'omni-math'
                dataset_file_path = os.path.join(dataset_path, 'omni-math', 'test.jsonl')
            else:
                print(f"Unsupported dataset: {dataset}")
                continue
            model_path = os.path.join(models_path, model_list[i])
            if not os.path.exists(model_path):
                print(f"Model path {model_path} does not exist.")
                error_list.append(model_path)
                continue
            
            try:
                print(f"Starting thread for model: {model_list[i]}")
                multiprocessing.set_start_method("spawn", force=True)
                part_name = 'try_' + "cuda used_" + os.environ['CUDA_VISIBLE_DEVICES']
                output_dir = os.path.join(output_dir, part_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(os.path.join(output_dir, data_source)):
                    os.makedirs(os.path.join(output_dir, data_source))
                
                thread = multiprocessing.Process(target=main, args=(
                    model_path, 
                    model_list[i], 
                    data_source, 
                    dataset_file_path, 
                    'off', 
                    'project name', 
                    'expr name', 
                    os.path.join(output_dir, data_source),
                    True)    
                )
                thread.start()
                thread.join()  # Wait for the thread to complete
                print(f"Thread for model {model_list[i]} completed")
            except Exception as e:
                print(f"Error processing model {model_list[i]}: {e}")
                error_list.append(model_list[i])
        
    
    print("all models expr completed.")
    print("success model number:", len(model_list)-len(error_list))
    if len(error_list) > 0:
        print("error model number:", len(error_list))
        print("error model list:", error_list)
    
