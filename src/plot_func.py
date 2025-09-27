import json
from typing import List, Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from transformers import AutoTokenizer
import pandas as pd





def plot_entropy(data1: List, data2) -> None:
    plt.plot(data1)
    plt.plot(data2)
    plt.xscale('symlog', linthresh=10) # log scale for x axis
    plt.legend(['origin model', 'trained model'])
    plt.grid(True)
    plt.xlabel('Time step(log scale)')
    plt.ylabel('Token Entropy')
    plt.title('Different model Token Entropy Comparison')
    plt.show()

def split_data_by_accuracy(origin_data: List[dict]) -> tuple[List[dict], List[dict]]:
    correct_data: List[dict] = []
    incorrect_data: List[dict] = []
    for item in origin_data:
        if item['is_correct'] == True:
            correct_data.append(item)
        else:
            incorrect_data.append(item)
    return correct_data, incorrect_data


def summary_result(origin_data: List[dict], correct_data: List[dict], incorrect_data: List[dict], verbose = False, data_type = 'old') -> dict:
    if data_type == 'old':
        entropy_key = 'entropy'
    else:
        entropy_key = 'step_entropys'
    res = {}
    def show_stat(data: List[dict], title: str) -> None:
        min_len = int(1e9)
        max_len = -1
        avg_len = 0
        middle_len: int
        for item in data:
            min_len = min(min_len, len(item[entropy_key]))
            max_len = max(max_len, len(item[entropy_key]))
            avg_len = avg_len + len(item[entropy_key])
        avg_len = avg_len / len(data) if data else 0
        if verbose and title == 'origin data':
            print(f"{title} length - Min: {min_len}, Max: {max_len}, Avg: {avg_len}, Middle length: {len(data[len(data)//2][entropy_key])}")
            print(f"correct number: {len(correct_data)}//{len(origin_data)}, accuracy: {len(correct_data)/len(origin_data)}")
            print("-----------------------------------------------------------------")
        res[title] = {
            "min": min_len,
            "max": max_len,
            "avg": avg_len,
            "middle": len(data[len(data)//2][entropy_key])
        }
    show_stat(origin_data, "origin data")
    show_stat(correct_data, "correct data")
    show_stat(incorrect_data, "incorrect data")
    return res

def interpolate_data(data: List[dict], stat_res: dict, name:str, target_mode: str = "max_length", data_type = 'old') -> List:
    if data_type == 'old':
        entropy_key = 'entropy'
    else:
        entropy_key = 'step_entropys'
    if target_mode == "max_length":
        target_length = stat_res[name]["max"]
    elif target_mode == "avg_length":
        target_length = stat_res[name]["avg"]
    elif target_mode == "min_length":
        target_length = stat_res[name]["min"]
    elif target_mode == "middle_length":
        target_length = stat_res[name]["middle"]
    else:
        raise ValueError("target_mode should be one of ['max_length', 'avg_length', 'min_length', 'middle_length']")

    interpolated_data = []
    for series in data:
        origin_x = np.linspace(0, 1, len(series[entropy_key]))
        target_x = np.linspace(0, 1, int(target_length))
        interpolated_serie = np.interp(target_x, origin_x, series[entropy_key])
        interpolated_data.append(interpolated_serie)

    final_avg_data = np.mean(interpolated_data, axis=0).tolist()
    return final_avg_data

def plot_figure_1(llama3_2_3B_data: List[dict], qwen3_8B_data: List[dict], qwen_14B_data: List[dict], qwen3_8B_nothinking: List[dict], r1_7B: List[dict]) -> None:
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # 数据提取和预处理
    llama3_2_3B_data_correct, llama3_2_3B_data_incorrect = split_data_by_accuracy(llama3_2_3B_data)
    qwen3_8B_data_correct, qwen3_8B_data_incorrect = split_data_by_accuracy(qwen3_8B_data)
    qwen_14B_data_correct, qwen_14B_data_incorrect = split_data_by_accuracy(qwen_14B_data)
    qwen3_8B_nothinking_correct, qwen3_8B_nothinking_incorrect = split_data_by_accuracy(qwen3_8B_nothinking)
    r1_7B_correct, r1_7B_incorrect = split_data_by_accuracy(r1_7B)

    # 提取平均熵数据
    llama_entropy_avg = interpolate_data(llama3_2_3B_data_correct, summary_result(llama3_2_3B_data, llama3_2_3B_data_correct, llama3_2_3B_data_incorrect), "correct data", target_mode="avg_length")
    qwen8_entropy_avg = interpolate_data(qwen3_8B_data_correct, summary_result(qwen3_8B_data, qwen3_8B_data_correct, qwen3_8B_data_incorrect), "correct data", target_mode="avg_length")
    qwen14_entropy_avg = interpolate_data(qwen_14B_data_correct, summary_result(qwen_14B_data, qwen_14B_data_correct, qwen_14B_data_incorrect), "correct data", target_mode="avg_length")
    qwen3_8b_nothinking_avg = interpolate_data(qwen3_8B_nothinking_correct, summary_result(qwen3_8B_nothinking, qwen3_8B_nothinking_correct, qwen3_8B_nothinking_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
    r1_7b_avg = interpolate_data(r1_7B_correct, summary_result(r1_7B, r1_7B_correct, r1_7B_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')

    # 提取第一个正确示例的熵数据
    llama_entropy_first = llama3_2_3B_data_correct[0]['entropy'] if llama3_2_3B_data_correct else []
    qwen8_entropy_first = qwen3_8B_data_correct[0]['entropy'] if qwen3_8B_data_correct else []
    qwen14_entropy_first = qwen_14B_data_correct[0]['entropy'] if qwen_14B_data_correct else []
    qwen3_8b_nothinking_first = qwen3_8B_nothinking_correct[0]['step_entropys'] if qwen3_8B_nothinking_correct else []
    r1_7b_first = r1_7B_correct[0]['step_entropys'] if r1_7B_correct else []

    # 将四张图的数据组织成列表，方便循环
    # 每个元组包含: [数据列表], [线条名称], [文件名], [y轴标签], [子图标题]
    plots = [
        # 图 (a)
        ([(llama_entropy_first, 'red', '-.'), (qwen14_entropy_first, 'green', '-'), (qwen3_8b_nothinking_first, 'purple', '--')],
         ['Llama3.2-3B', 'Qwen2.5-14B', 'Qwen3-8B w/o Thinking'],
         'figure1_case_not_longcot.pdf', 'TECA', '(a)'),

        # 图 (b)
        ([(qwen8_entropy_first, 'blue', '-')],
         ['Qwen3-8B w Thinking'],
         'figure1_case_long_cot.pdf', 'TECA', '(b)'),

        # 图 (c)
        ([(llama_entropy_avg, 'red', '-.'), (qwen14_entropy_avg, 'green', '-'), (qwen3_8b_nothinking_avg, 'purple', '--')],
         ['Llama3.2-3B', 'Qwen2.5-14B', 'Qwen3-8B w/o Thinking'],
         'figure1_general_not_longcot.pdf', 'TECA', '(c)'),

        # 图 (d)
        ([(qwen8_entropy_avg, 'blue', '-')],
         ['Qwen3-8B w Thinking'],
         'figure1_general_longcot.pdf', 'TECA', '(d)')
    ]

    # 循环绘制并保存每张图
    for data_list, line_labels, filename, y_name, title_char in plots:
        plt.figure(figsize=(8, 6))
        lines = []
        # 绘制所有线条
        for data, color, linestyle in data_list:
            if data:  # 检查数据是否非空
                x_data = np.arange(1, len(data) + 1)
                line, =plt.plot(x_data, data, color=color, linestyle=linestyle, alpha=0.8, linewidth=2)
                lines.append(line)
                if title_char == '(a)':
                    if color == 'red':
                        star_x = 133
                    elif color == 'green':
                        star_x = 160
                    elif color =='purple':
                        star_x = 122
                    star_y = data[star_x]
                    scatter_point  = plt.scatter(star_x, star_y, s=500, color='orange', marker='*', zorder=5, label='point')
                elif title_char == '(b)':
                    star_x = 363
                    star_y = data[star_x]
                    scatter_point  = plt.scatter(star_x, star_y, s=500, color='orange', marker='*', zorder=5, label='point')

        plt.xlabel('Inference Step', fontfamily='Times New Roman', fontsize=20)
        plt.ylabel(y_name, fontfamily='Times New Roman', fontsize=20)
        if title_char == '(a)':
            lines[1], lines[2] = lines[2], lines[1]
            line_labels[1], line_labels[2] = line_labels[2], line_labels[1]
            plt.legend(lines, line_labels, prop={'family': 'Times New Roman', 'size': 20})
        else:
            plt.legend(line_labels, prop={'family': 'Times New Roman', 'size': 20})
        # plt.grid(True, alpha=0.3)
        # plt.title(title_char, fontfamily='Times New Roman', fontsize=16, loc='left')

        # 设置坐标轴刻度字体
        plt.xticks(fontfamily='Times New Roman')
        plt.yticks(fontfamily='Times New Roman')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
        plt.close()

def plot_figure_2(length_clip_ratio, length_avg, length_min) -> None:
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 25

    # 提取四组数据
    qwen3_4B_train_data = {
        'length_clip_ratio': length_clip_ratio['Group: Qwen3-4B - response_length/clip_ratio'],
        'length_avg': length_avg['Group: Qwen3-4B - response_length/mean'],
        'length_min': length_min['Group: Qwen3-4B - response_length/min']
    }
    qwen3_8B_train_data = {
        'length_clip_ratio': length_clip_ratio['Group: Qwen3-8B - response_length/clip_ratio'],
        'length_avg': length_avg['Group: Qwen3-8B - response_length/mean'],
        'length_min': length_min['Group: Qwen3-8B - response_length/min']
    }

    # 四组数据及其对应的标题和文件名
    datasets = [
        (qwen3_4B_train_data['length_clip_ratio'], 'Qwen3-4B', 'qwen3_4b_clip_ratio.pdf', 'Length Clip Ratio'),
        (qwen3_4B_train_data['length_avg'], 'Qwen3-4B', 'qwen3_4b_length_avg.pdf', 'Length Average'),
        (qwen3_8B_train_data['length_clip_ratio'], 'Qwen3-8B', 'qwen3_8b_clip_ratio.pdf', 'Length Clip Ratio'),
        (qwen3_8B_train_data['length_avg'], 'Qwen3-8B', 'qwen3_8b_length_avg.pdf', 'Length Average'),
        (qwen3_4B_train_data['length_min'], 'Qwen3-4B', 'qwen3_4b_length_min.pdf', 'Length Minimum'),
        (qwen3_8B_train_data['length_min'], 'Qwen3-8B', 'qwen3_8b_length_min.pdf', 'Length Minimum')
    ]

    # 为每组数据创建独立的图表
    for i, (data, line_name, filename, y_name) in enumerate(datasets):
        plt.figure(figsize=(6, 4))

        # 生成x轴（step）
        steps = list(range(1, len(data)+1))
        blue_color = '#1f77b4'      # 蓝色
        orange_color = 'red'
        line_color = blue_color if 'Qwen3-4B' in line_name else orange_color
        # 绘制图表
        plt.plot(steps, data, linewidth=2, color=line_color, alpha=0.8)

        plt.xlabel('Training Step', fontfamily='Times New Roman', fontsize=25)
        plt.ylabel(y_name, fontfamily='Times New Roman', fontsize=25)
        # plt.legend([line_name], prop={'family': 'Times New Roman', 'size': 12})

        # 移除网格
        plt.grid(False)

        # 设置坐标轴刻度字体
        plt.xticks(np.arange(0, max(steps)+1, step=100), fontfamily='Times New Roman')
        plt.yticks(fontfamily='Times New Roman')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
        # 清除当前图表，准备下一个
        plt.close()


def plot_figure_3(qwen3_4b_origin_gsm8k: List[dict], qwen3_4b_trained_gsm8k: List[dict]) -> None:
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # 数据提取和预处理
    qwen3_4b_origin_gsm8k_correct, qwen3_4b_origin_gsm8k_incorrect = split_data_by_accuracy(qwen3_4b_origin_gsm8k)
    qwen3_4b_trained_gsm8k_correct, qwen3_4b_trained_gsm8k_incorrect = split_data_by_accuracy(qwen3_4b_trained_gsm8k)

    qwen3_4b_origin_entropy_avg = interpolate_data(
        qwen3_4b_origin_gsm8k_correct,
        summary_result(qwen3_4b_origin_gsm8k, qwen3_4b_origin_gsm8k_correct, qwen3_4b_origin_gsm8k_incorrect, data_type='new'),
        "correct data", target_mode="avg_length", data_type='new')
    qwen3_4b_trained_entropy_avg = interpolate_data(
        qwen3_4b_trained_gsm8k_correct,
        summary_result(qwen3_4b_trained_gsm8k, qwen3_4b_trained_gsm8k_correct, qwen3_4b_trained_gsm8k_incorrect, data_type='new'),
        "correct data", target_mode="avg_length", data_type='new')

    qwen3_4b_origin_te_avg = []
    for i in range(len(qwen3_4b_origin_entropy_avg)):
        if i==0:
            qwen3_4b_origin_te_avg.append(qwen3_4b_origin_entropy_avg[i])
        else:
            te = qwen3_4b_origin_entropy_avg[i]*(i+1)-qwen3_4b_origin_entropy_avg[i-1]*i
            qwen3_4b_origin_te_avg.append(te)
    qwen3_4b_trained_te_avg = []
    for i in range(len(qwen3_4b_trained_entropy_avg)):
        if i==0:
            qwen3_4b_trained_te_avg.append(qwen3_4b_trained_entropy_avg[i])
        else:
            te = qwen3_4b_trained_entropy_avg[i]*(i+1)-qwen3_4b_trained_entropy_avg[i-1]*i
            qwen3_4b_trained_te_avg.append(te)

    qwen3_4b_origin_token_entropy = qwen3_4b_origin_gsm8k_correct[0]['token_entropys']
    qwen3_4b_trained_token_entropy = qwen3_4b_trained_gsm8k_correct[0]['token_entropys']
    qwen3_4b_origin_case = qwen3_4b_origin_gsm8k_correct[0]['step_entropys']
    qwen3_4b_trained_case = qwen3_4b_trained_gsm8k_correct[0]['step_entropys']

    # 数据坐标
    qwen3_4b_origin_x_avg = np.arange(1, len(qwen3_4b_origin_entropy_avg) + 1)
    qwen3_4b_trained_x_avg = np.arange(1, len(qwen3_4b_trained_entropy_avg) + 1)
    qwen3_4b_origin_token_x = np.arange(1, len(qwen3_4b_origin_token_entropy) + 1)
    qwen3_4b_trained_token_x = np.arange(1, len(qwen3_4b_trained_token_entropy) + 1)
    qwen3_4b_origin_case_x = np.arange(1, len(qwen3_4b_origin_case) + 1)
    qwen3_4b_trained_case_x = np.arange(1, len(qwen3_4b_trained_case) + 1)

    # 合并后的数据集（一个模型一张图，双Y轴）
    datasets = [
        ('Qwen3-4B-Origin', 'qwen3_4b_origin_dual.pdf',
         qwen3_4b_origin_x_avg, qwen3_4b_origin_te_avg,
         qwen3_4b_origin_x_avg, qwen3_4b_origin_entropy_avg),
        ('Qwen3-4B-Trained', 'qwen3_4b_trained_dual.pdf',
         qwen3_4b_trained_x_avg, qwen3_4b_trained_te_avg,
         qwen3_4b_trained_x_avg, qwen3_4b_trained_entropy_avg),
        ('Qwen3-4B-Origin', 'qwen3_4b_origin_case_dual.pdf',
         qwen3_4b_origin_token_x, qwen3_4b_origin_token_entropy,
         qwen3_4b_origin_case_x, qwen3_4b_origin_case),
        ('Qwen3-4B-Trained', 'qwen3_4b_trained_case_dual.pdf',
         qwen3_4b_trained_token_x, qwen3_4b_trained_token_entropy,
         qwen3_4b_trained_case_x, qwen3_4b_trained_case)
    ]

    for model_name, filename, token_x, token_y, tema_x, tema_y in datasets:
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # 左轴：Token Entropy
        color1 = '#1f77b4'
        ax1.plot(token_x, token_y, linewidth=2, color=color1, alpha=0.8, label=f'{model_name} Token Entropy')
        ax1.set_xlabel('Inference Step', fontfamily='Times New Roman', fontsize=20)
        ax1.set_ylabel('Token Entropy', fontfamily='Times New Roman', fontsize=20, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)

        # 右轴：TECA
        ax2 = ax1.twinx()
        color2 = 'red'
        ax2.plot(tema_x, tema_y, linewidth=2, color=color2, alpha=0.8, label=f'{model_name} TEMA')
        ax2.set_ylabel('TECA', fontfamily='Times New Roman', fontsize=20, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.tick_params(axis='y', labelsize=16)

        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
        plt.close()


def plot_figure_apdx_1(llama3_2_3B_data: List[dict], qwen3_8B_data: List[dict], qwen_14B_data: List[dict], qwen3_8B_nothinking: List[dict]) -> None:
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    llama3_2_3B_data_correct, llama3_2_3B_data_incorrect = split_data_by_accuracy(llama3_2_3B_data)
    qwen3_8B_data_correct, qwen3_8B_data_incorrect = split_data_by_accuracy(qwen3_8B_data)
    qwen_14B_data_correct, qwen_14B_data_incorrect = split_data_by_accuracy(qwen_14B_data)
    qwen3_8B_nothinking_correct, qwen3_8B_nothinking_incorrect = split_data_by_accuracy(qwen3_8B_nothinking)

    llama_correct_entropy_avg = interpolate_data(llama3_2_3B_data_correct, summary_result(llama3_2_3B_data, llama3_2_3B_data_correct, llama3_2_3B_data_incorrect), "correct data", target_mode="avg_length")
    qwen8_correct_entropy_avg = interpolate_data(qwen3_8B_data_correct, summary_result(qwen3_8B_data, qwen3_8B_data_correct, qwen3_8B_data_incorrect), "correct data", target_mode="avg_length")
    qwen14_correct_entropy_avg = interpolate_data(qwen_14B_data_correct, summary_result(qwen_14B_data, qwen_14B_data_correct, qwen_14B_data_incorrect), "correct data", target_mode="avg_length")
    qwen3_8b_correct_nothinking_avg = interpolate_data(qwen3_8B_nothinking_correct, summary_result(qwen3_8B_nothinking, qwen3_8B_nothinking_correct, qwen3_8B_nothinking_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')

    llama_incorrect_entropy_avg = interpolate_data(llama3_2_3B_data_incorrect, summary_result(llama3_2_3B_data, llama3_2_3B_data_correct, llama3_2_3B_data_incorrect), "incorrect data", target_mode="avg_length")
    qwen8_incorrect_entropy_avg = interpolate_data(qwen3_8B_data_incorrect, summary_result(qwen3_8B_data, qwen3_8B_data_correct, qwen3_8B_data_incorrect), "incorrect data", target_mode="avg_length")
    qwen14_incorrect_entropy_avg = interpolate_data(qwen_14B_data_incorrect, summary_result(qwen_14B_data, qwen_14B_data_correct, qwen_14B_data_incorrect), "incorrect data", target_mode="avg_length")
    qwen3_8b_incorrect_nothinking_avg = interpolate_data(qwen3_8B_nothinking_incorrect, summary_result(qwen3_8B_nothinking, qwen3_8B_nothinking_correct, qwen3_8B_nothinking_incorrect, data_type='new'), "incorrect data", target_mode="avg_length", data_type='new')

    llama_correct_x_avg = np.arange(1, len(llama_correct_entropy_avg)+1)
    qwen8_correct_x_avg = np.arange(1, len(qwen8_correct_entropy_avg)+1)
    qwen14_correct_x_avg = np.arange(1, len(qwen14_correct_entropy_avg)+1)
    qwen3_8b_correct_nothinking_x_avg = np.arange(1, len(qwen3_8b_correct_nothinking_avg)+1)
    llama_incorrect_x_avg = np.arange(1, len(llama_incorrect_entropy_avg)+1)
    qwen8_incorrect_x_avg = np.arange(1, len(qwen8_incorrect_entropy_avg)+1)
    qwen14_incorrect_x_avg = np.arange(1, len(qwen14_incorrect_entropy_avg)+1)
    qwen3_8b_incorrect_nothinking_x_avg = np.arange(1, len(qwen3_8b_incorrect_nothinking_avg)+1)

    datasets = [
        (llama_correct_x_avg, llama_correct_entropy_avg, llama_incorrect_x_avg, llama_incorrect_entropy_avg, 'Llama3.2-3B-correct', 'Llama3.2-3B-incorrect', 'apdx1_llama3_2_3b.pdf'),
        (qwen8_correct_x_avg, qwen8_correct_entropy_avg, qwen8_incorrect_x_avg, qwen8_incorrect_entropy_avg, 'Qwen3-8B-correct', 'Qwen3-8B-incorrect', 'apdx1_qwen3_8b.pdf'),
        (qwen14_correct_x_avg, qwen14_correct_entropy_avg, qwen14_incorrect_x_avg, qwen14_incorrect_entropy_avg, 'Qwen2.5-14B-correct', 'Qwen2.5-14B-incorrect', 'apdx1_qwen2_5_14b.pdf'),
    ]

    for correct_x, correct_y, incorrect_x, incorrect_y, correct_label, incorrect_label, filename in datasets:
        plt.figure(figsize=(6, 4))
        line1, =plt.plot(correct_x, correct_y, linestyle='-', color='#1f77b4', label=correct_label, alpha=0.8, linewidth=2)
        line2, =plt.plot(incorrect_x, incorrect_y, linestyle='--', color='red', label=incorrect_label, alpha=0.8, linewidth=2)
        plt.xlabel('Inference Step', fontfamily='Times New Roman', fontsize=20)
        plt.ylabel('TECA', fontfamily='Times New Roman', fontsize=20)
        plt.legend([line2, line1], [incorrect_label, correct_label], prop={'family': 'Times New Roman', 'size': 20})
        plt.grid(False)

        # 设置坐标轴刻度字体
        plt.xticks(fontfamily='Times New Roman')
        plt.yticks(fontfamily='Times New Roman')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
        plt.close()

def plot_figure_apdx_2(qwen3_4b_origin_AIME25: List[dict], qwen3_4b_trained_AIME25: List[dict], qwen3_8b_origin_aime25: List[dict], qwen3_8b_trained_aime25: List[dict]) -> None:
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    qwen3_4b_origin_aime25_correct, qwen3_4b_origin_aime25_incorrect = split_data_by_accuracy(qwen3_4b_origin_AIME25)
    qwen3_4b_trained_aime25_correct, qwen3_4b_trained_aime25_incorrect = split_data_by_accuracy(qwen3_4b_trained_AIME25)
    qwen3_8b_origin_aime25_correct, qwen3_8b_origin_aime25_incorrect = split_data_by_accuracy(qwen3_8b_origin_aime25)
    qwen3_8b_trained_aime25_correct, qwen3_8b_trained_aime25_incorrect = split_data_by_accuracy(qwen3_8b_trained_aime25)

    qwen3_4b_origin_entropy_avg = interpolate_data(qwen3_4b_origin_aime25_correct, summary_result(qwen3_4b_origin_AIME25, qwen3_4b_origin_aime25_correct, qwen3_4b_origin_aime25_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
    qwen3_4b_trained_entropy_avg = interpolate_data(qwen3_4b_trained_aime25_correct, summary_result(qwen3_4b_trained_AIME25, qwen3_4b_trained_aime25_correct, qwen3_4b_trained_aime25_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
    qwen3_8b_origin_entropy_avg = interpolate_data(qwen3_8b_origin_aime25_correct, summary_result(qwen3_8b_origin_aime25, qwen3_8b_origin_aime25_correct, qwen3_8b_origin_aime25_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
    qwen3_8b_trained_entropy_avg = interpolate_data(qwen3_8b_trained_aime25_correct, summary_result(qwen3_8b_trained_aime25, qwen3_8b_trained_aime25_correct, qwen3_8b_trained_aime25_incorrect, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
    qwen3_4b_origin_x_avg = np.arange(1, len(qwen3_4b_origin_entropy_avg)+1)
    qwen3_4b_trained_x_avg = np.arange(1, len(qwen3_4b_trained_entropy_avg)+1)
    qwen3_8b_origin_x_avg = np.arange(1, len(qwen3_8b_origin_entropy_avg)+1)
    qwen3_8b_trained_x_avg = np.arange(1, len(qwen3_8b_trained_entropy_avg)+1)
    datasets = [
        (qwen3_4b_origin_x_avg, qwen3_4b_origin_entropy_avg, qwen3_4b_trained_x_avg, qwen3_4b_trained_entropy_avg, 'Qwen3-4B', 'Qwen3-4B-CER', 'apdx2_qwen3_4b_origin_trained_aime25.pdf', 'TECA'),
        (qwen3_8b_origin_x_avg, qwen3_8b_origin_entropy_avg, qwen3_8b_trained_x_avg, qwen3_8b_trained_entropy_avg, 'Qwen3-8B', 'Qwen3-8B-CER', 'apdx2_qwen3_8b_origin_trained_aime25.pdf', 'TECA')
    ]

    for origin_x, origin_y, trained_x, trained_y, origin_label, trained_label, filename, y_name in datasets:
        plt.figure(figsize=(6, 4))
        line1, =plt.plot(origin_x, origin_y, linestyle='--', color='#1f77b4', label=origin_label, alpha=0.8, linewidth=2)
        line2, =plt.plot(trained_x, trained_y, linestyle='-', color='red', label=trained_label, alpha=0.8, linewidth=2)
        plt.xlabel('Inference Step', fontfamily='Times New Roman', fontsize=20)
        plt.ylabel(y_name, fontfamily='Times New Roman', fontsize=20)
        plt.legend([line2, line1], [trained_label, origin_label], prop={'family': 'Times New Roman', 'size': 20})
        plt.grid(False)

        # 设置坐标轴刻度字体
        plt.xticks(fontfamily='Times New Roman')
        plt.yticks(fontfamily='Times New Roman')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
        plt.close()




def plot_token_entropy(data: List[dict]) -> None:
    correct_data, incorrect_data = split_data_by_accuracy(data)
    entropy = correct_data[0]['token_entropys']
    plt.plot(entropy)
    plt.legend(['correct case'])
    plt.grid(True)
    plt.xlabel('Time step(log scale)')
    plt.ylabel('Token Entropy')
    plt.title('Different case Token Entropy Comparison')
    plt.show()
def plot_entropy_figure(datas: List[dict], log_scale: bool = False, need_general: bool = True):
    tokenizer = AutoTokenizer.from_pretrained("models/qwen3-8B", trust_remote_code=True)
    for data in datas:
        correct_data, incorrect_data = split_data_by_accuracy(data['data'])
        if need_general:
            entropy = interpolate_data(correct_data, summary_result(data['data'], correct_data, incorrect_data, data_type='new'), "correct data", target_mode="avg_length", data_type='new')
        else:
            entropy = correct_data[0]['step_entropys']
        plt.plot(entropy)


    if log_scale:
        plt.xscale('symlog', linthresh=10) # log scale for x axis


    plt.legend([datas[i]['name'] for i in range(len(datas))])

    plt.grid(True)
    plt.xlabel('Time step(log scale)' if log_scale else 'Time step')
    plt.ylabel('Token Entropy')
    # plt.title('Different case Token Entropy Comparison')
    plt.show()


def main():

    def plot1():
        llama3_2_3B_path = 'data/llama3_2_3B_entropy_modified.json'
        qwen3_8B_path = 'data/qwen3_8B_entropy.json'
        qwen_14B_path = 'data/qwen_14B_entropy.json'
        qwen3_8B_no_thinking_path = 'data/gsm8k_test_qwen3_8B_nothinking.json'
        r1_distill_qwen2_5_7B_path = 'data/gsm8k_test_deepseek_r1_distill_qwen_7B.json'
        with open(qwen3_8B_path, 'r', encoding='utf-8') as f:
            qwen3_8B_data = json.load(f)
        with open(qwen_14B_path, 'r', encoding='utf-8') as f:
            qwen_14B_data = json.load(f)
        with open(llama3_2_3B_path, 'r', encoding='utf-8') as f:
            llama3_2_3B_data = json.load(f)
        with open(qwen3_8B_no_thinking_path, 'r', encoding='utf-8') as f:
            qwen3_8B_no_thinking_data = json.load(f)
        with open(r1_distill_qwen2_5_7B_path, 'r', encoding='utf-8') as f:
            r1_distill_qwen2_5_7B_data = json.load(f)

        plot_figure_1(llama3_2_3B_data, qwen3_8B_data, qwen_14B_data, qwen3_8B_no_thinking_data, r1_distill_qwen2_5_7B_data)
    def plot2():
        final_reward = pd.read_csv('training_data/final_reward.csv')
        length_clip_ratio = pd.read_csv('training_data/length_clip_ratio.csv')
        length_mean = pd.read_csv('training_data/length_mean.csv')
        length_min = pd.read_csv('training_data/length_min.csv')
        plot_figure_2(
            length_clip_ratio,
            length_mean,
            length_min
        )
    def plot3():
        with open('qwen3_4B_origin_test/gsm8k/gsm8k_qwen3_4B.json', 'r', encoding='utf-8') as f:
            qwen3_4b_origin_gsm8k = json.load(f)
        with open('qwen3_4B_trained_test/gsm8k/gsm8k_merged_model_global_step_210.json', 'r', encoding='utf-8') as f:
            qwen3_4b_trained_gsm8k = json.load(f)
        plot_figure_3(qwen3_4b_origin_gsm8k, qwen3_4b_trained_gsm8k)

    def plot_apdx_1():
        llama3_2_3B_path = 'data/llama3_2_3B_entropy_modified.json'
        qwen3_8B_path = 'data/qwen3_8B_entropy.json'
        qwen_14B_path = 'data/qwen_14B_entropy.json'
        qwen3_8B_no_thinking_path = 'data/gsm8k_test_qwen3_8B_nothinking.json'
        with open(qwen3_8B_path, 'r', encoding='utf-8') as f:
            qwen3_8B_data = json.load(f)
        with open(qwen_14B_path, 'r', encoding='utf-8') as f:
            qwen_14B_data = json.load(f)
        with open(llama3_2_3B_path, 'r', encoding='utf-8') as f:
            llama3_2_3B_data = json.load(f)
        with open(qwen3_8B_no_thinking_path, 'r', encoding='utf-8') as f:
            qwen3_8B_no_thinking_data = json.load(f)
        plot_figure_apdx_1(llama3_2_3B_data, qwen3_8B_data, qwen_14B_data, qwen3_8B_no_thinking_data)

    def plot_apdx_2():
        with open('qwen3_4B_origin_test/aime25/aime25_qwen3_4B.json', 'r', encoding='utf-8') as f:
            qwen3_4b_origin_aime25 = json.load(f)
        with open('qwen3_4B_trained_test/aime25/aime25_merged_model_global_step_210.json', 'r', encoding='utf-8') as f:
            qwen3_4b_trained_aime25 = json.load(f)
        with open('qwen3_8B_origin_test/aime25/aime25_qwen3_8B.json', 'r', encoding='utf-8') as f:
            qwen3_8b_origin_aime25 = json.load(f)
        with open('qwen3_8B_trained_test/aime25/aime25_merged_model_global_step_260.json', 'r', encoding='utf-8') as f:
            qwen3_8b_trained_aime25 = json.load(f)
        plot_figure_apdx_2(qwen3_4b_origin_aime25, qwen3_4b_trained_aime25, qwen3_8b_origin_aime25, qwen3_8b_trained_aime25)
    def plot_token_entropy_try():
        with open('qwen3_8B_trained_test/gsm8k/gsm8k_merged_model_global_step_260.json', 'r', encoding='utf-8') as f:
            qwen3_8b_origin_gsm8k = json.load(f)
        plot_token_entropy(qwen3_8b_origin_gsm8k)
    def plot_try():
        with open('path/to/single_sample/gsm8k/qwen3_4B_ec_incorrect.json', 'r', encoding='utf-8') as f:
            incorrect_data = json.load(f)
        with open('path/to/single_sample/gsm8k/qwen3_4B_ec_repeat_until_correct.jsonl', 'r', encoding='utf-8') as f:
            correct_data = json.load(f)
        # plot_entropy_figure([{'name': 'qwen3_8B_origin_gsm8k', 'data': correct_data}], log_scale=False, need_general=False)

    plot_apdx_2()

if __name__ == '__main__':
    main()