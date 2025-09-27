import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def merge_lora_weights(base_model_path, adapter_path, output_path, device="auto"):
    """
    合并基础模型和LoRA adapter权重
    
    Args:
        base_model_path (str): 原始模型路径
        adapter_path (str): LoRA adapter路径
        output_path (str): 合并后模型保存路径
        device (str): 设备类型，"auto", "cpu", "cuda"等
    """
    
    print(f"开始加载基础模型: {base_model_path}")
    
    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用fp16节省显存
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"开始加载LoRA adapter: {adapter_path}")
    
    # 加载LoRA配置
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # 加载PEFT模型（基础模型 + adapter）
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        dtype=torch.float16
    )
    
    print("开始合并权重...")
    
    # 合并权重 - 这会将LoRA权重合并到基础模型中
    merged_model = model.merge_and_unload()
    
    print(f"保存合并后的模型到: {output_path}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_path,
        save_function=torch.save,
        safe_serialization=True  # 使用safetensors格式更安全
    )
    
    # 同时保存tokenizer（如果存在）
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print("Tokenizer已保存")
    except Exception as e:
        print(f"保存tokenizer时出现错误: {e}")
    
    # 修复JSON序列化问题的函数
    def make_json_serializable(obj):
        """递归转换对象为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'value'):  # 枚举类型
            return obj.value
        elif hasattr(obj, 'name'):   # 枚举类型的name属性
            return obj.name
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)  # 其他复杂对象转为字符串
    
    # 保存合并信息
    merge_info = {
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "merge_timestamp": str(datetime.now()),
        "device_info": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    }
    
    # 安全地处理peft_config
    try:
        if hasattr(peft_config, 'to_dict'):
            raw_config = peft_config.to_dict()
            merge_info["peft_config"] = make_json_serializable(raw_config)
        else:
            merge_info["peft_config"] = str(peft_config)
    except Exception as e:
        print(f"警告: 无法序列化PEFT配置: {e}")
        merge_info["peft_config"] = {
            "error": f"配置序列化失败: {str(e)}",
            "config_type": str(type(peft_config))
        }
    
    # 保存合并信息到JSON文件
    try:
        with open(os.path.join(output_path, "merge_info.json"), 'w', encoding='utf-8') as f:
            json.dump(merge_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 保存合并信息失败: {e}")
    
    print("模型合并完成！")
    return merged_model

# 其余函数保持不变
def manual_merge_lora_weights(base_model_path, adapter_path, output_path):
    """
    手动合并LoRA权重的方法（适用于自定义需求）
    """
    print("使用手动方法合并LoRA权重...")
    
    # 加载基础模型权重
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
    )
    
    # 加载adapter权重
    adapter_weights = torch.load(
        os.path.join(adapter_path, "adapter_model.bin"),
        map_location="cpu"
    )
    
    # 加载adapter配置
    with open(os.path.join(adapter_path, "adapter_config.json"), 'r') as f:
        adapter_config = json.load(f)
    
    r = adapter_config["r"]  # LoRA rank
    alpha = adapter_config["lora_alpha"]
    scaling = alpha / r
    
    print(f"LoRA配置: r={r}, alpha={alpha}, scaling={scaling}")
    
    # 手动合并权重
    base_state_dict = base_model.state_dict()
    
    for key, value in adapter_weights.items():
        if "lora_A" in key:
            # 获取对应的lora_B权重
            base_key = key.replace("lora_A", "").replace(".default", "")
            lora_B_key = key.replace("lora_A", "lora_B")
            
            if lora_B_key in adapter_weights:
                # 计算LoRA增量: B @ A
                lora_A = value
                lora_B = adapter_weights[lora_B_key]
                
                # LoRA增量
                delta_weight = (lora_B @ lora_A) * scaling
                
                # 找到基础模型中对应的权重
                target_key = None
                for base_key_candidate in base_state_dict.keys():
                    if base_key in base_key_candidate:
                        target_key = base_key_candidate
                        break
                
                if target_key and target_key in base_state_dict:
                    print(f"合并权重: {target_key}")
                    base_state_dict[target_key] = base_state_dict[target_key] + delta_weight.to(base_state_dict[target_key].device)
    
    # 更新模型权重
    base_model.load_state_dict(base_state_dict)
    
    # 保存合并后的模型
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(output_path, safe_serialization=True)
    
    print("手动合并完成！")
    return base_model

def validate_merged_model(merged_model_path, test_input="Hello, how are you?"):
    """
    验证合并后的模型是否正常工作
    """
    print(f"验证合并后的模型: {merged_model_path}")
    
    try:
        # 加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        
        # 测试推理
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"测试输入: {test_input}")
        print(f"模型输出: {generated_text}")
        print("模型验证成功！")
        
        return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

# 批量合并多个adapter的示例
def batch_merge_adapters(base_model_path, adapter_configs, output_base_path):
    """
    批量合并多个LoRA adapters
    
    Args:
        base_model_path: 基础模型路径
        adapter_configs: [(adapter_path, output_suffix), ...] 配置列表
        output_base_path: 输出基础路径
    """
    for adapter_path, suffix in adapter_configs:
        output_path = os.path.join(output_base_path, f"merged_model_{suffix}")
        print(f"\n处理adapter: {adapter_path} -> {output_path}")
        
        try:
            merge_lora_weights(base_model_path, adapter_path, output_path)
            print(f"✅ {suffix} 合并成功")
        except Exception as e:
            print(f"❌ {suffix} 合并失败: {e}")

if __name__ == "__main__":
    BASE_MODEL_PATH = '/path/to/models/qwen3_8B/qwen3_8B'
    OUTPUT_MODEL_PATH = '/path/to/qwen3_8B_merged'
    CKPT_MODEL_PATH = '/path/to/qwen3_8B_ckpts'

    global_steps = os.listdir(CKPT_MODEL_PATH)
    global_steps = ['global_step_10', 'global_step_260', 'global_step_270', 'global_step_280']
    
    def check_dirs(ckpt_model_path, global_steps):
        is_ok = True
        not_ok_dir = []
        for item in global_steps:
            lora_path = os.path.join(ckpt_model_path, item, 'actor', 'lora_adapter')
            if not os.path.exists(lora_path):
                print(f"缺少目录: {lora_path}")
                not_ok_dir.append(lora_path)
                is_ok = False
                continue
            
            if len(os.listdir(lora_path)) == 0:
                print(f"空lora adapter目录: {lora_path}")
                not_ok_dir.append(lora_path)
                is_ok = False
                continue
        
        if is_ok:
            print("all lora adapters exist")
        else:
            print("以下lora adapter存在问题:")
            for dir in not_ok_dir:
                print(f" - {dir}")
            raise ModuleNotFoundError("请检查缺失或空的lora adapter目录")
    check_dirs(CKPT_MODEL_PATH, global_steps)

    adapters = []
    for i, global_step in tqdm(enumerate(global_steps), desc="merging lora adapter ing..."):
        lora_path = os.path.join(CKPT_MODEL_PATH, global_step, 'actor', 'lora_adapter')
        adapters.append((lora_path, global_step))

    batch_merge_adapters(BASE_MODEL_PATH, adapters, OUTPUT_MODEL_PATH)