import json
import os
from typing import List
import matplotlib.pyplot as plt

exp_res = "/path/to/src/exp_res"
save_path = '/path/to/src/plots'


def plt_some_lines(line_names: List[str], y_data: List[List] | List, save_path: str, title: str) -> None:
    """
    give some lines or only a line to plot, the x axis is the index of y_data
    :param line_names: names of the lines
    :param y_data: data of the lines, can be a list of lists or a single list
    :param save_path: path to save the plot
    :param title: title of the plot
    """
    if len(line_names) != len(y_data):
        raise ValueError("The length of names and y_data must be the same.")
    
    if not isinstance(y_data[0], list):
        # Single line
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(y_data)))
        plt.plot(x_data, y_data)
        plt.xlabel('problem')
        plt.ylabel('token usage')
        plt.title(title)
        plt.grid(True)
        save_path = os.path.join(save_path, title + '.png')
        plt.savefig(save_path)
        plt.close()
    else:
        # Multiple lines
        plt.figure(figsize=(10, 6))
        for i, line_data in enumerate(y_data):
            x_data = list(range(len(line_data)))
            plt.plot(x_data, line_data, label=f'{line_names[i]}')
        plt.xlabel('problem')
        plt.ylabel('token usage')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_path, title + '.png')
        plt.savefig(save_path)
        plt.close()

def get_accuracy(data: List[dict]) -> float:
    total_number = len(data)
    correct_number = sum(1 for item in data if item.get('is_correct', False))
    return correct_number / total_number if total_number > 0 else 0.0

def get_response_len(data: List[dict]) -> float:
    response_lens = [item['response_length'] for item in data]
    return sum(response_lens) / len(data)

def show_res(file_path, name):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    accuracy = get_accuracy(data)
    avg_response_len = get_response_len(data)
    print(f"{name} - Accuracy: {accuracy}, Average Response Length: {avg_response_len}")
    print("---------------------------------------------------")

show_res('qwen3_8B_trained_test_more/try3/gsm8k_merged_model_global_step_260_thinking.json', 'gsm8k')

