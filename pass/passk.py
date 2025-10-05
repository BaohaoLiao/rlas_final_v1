
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

# JSON文件路径
ds_dir1 = '/home/wx13/reinforceflow/eval_bench/data/gen_data_validate_3/weqweasdas/validate/record.json'
#'/home/wx13/reinforceflow/eval_bench/data/gen_data_validate/weqweasdas/reinforce_ada_balance_step800/record.json'
ds_dir2 = '/home/wx13/reinforceflow/eval_bench/data/gen_data_validate_3/weqweasdas/validate/record.json'
#'/home/wx13/reinforceflow/eval_bench/data/gen_data_validate_2/weqweasdas/validate/record.json'
#'/home/wx13/reinforceflow/eval_bench/data/gen_data_validate/weqweasdas/grpo_step120/record.json'

#"/home/wx13/reinforceflow/eval_bench/data/gen_data/qwen15b_pass_rate_est/weqweasdas/from_default_filtered_openr1/record.json"
#"/home/wx13/reinforceflow/eval_bench/data/gen_data/qwen15b_step800_ada_balance_pass_rate_est/weqweasdas/from_default_filtered_openr1/record.json"

# 对应的标签
labels = ["Qwen2.5-Math-1.5B-Reinforce-ada400", "Qwen2.5-Math-1.5B"]
colors = ['blue', 'red']

def calculate_pass_at_k(scores: List[List[int]], k: int) -> float:
    """
    计算pass@k准确率
    scores: 每个样本的256个二进制分数列表
    k: 尝试次数
    """
    solved_count = 0
    total_samples = len(scores)
    
    for sample_scores in scores:
        # 取前k个分数，如果有任何一个是1，则认为该样本被解决
        if any(sample_scores[:k]):
            solved_count += 1
    
    return solved_count / total_samples

def compute_pass_k_curve(scores: List[List[int]], k_values: List[int], num_shuffles: int = 20) -> Tuple[List[float], List[float]]:
    """
    计算pass@k曲线，包含多次shuffle的均值和标准差
    """
    results = {k: [] for k in k_values}
    
    for shuffle_idx in range(num_shuffles):
        print(f"进行第 {shuffle_idx + 1}/{num_shuffles} 次shuffle...")
        
        # 对每个样本的scores进行shuffle
        shuffled_scores = []
        for sample_scores in scores:
            shuffled = sample_scores.copy()
            random.shuffle(shuffled)
            shuffled_scores.append(shuffled)
        
        # 计算每个k值的pass@k
        for k in k_values:
            pass_k = calculate_pass_at_k(shuffled_scores, k)
            results[k].append(pass_k)
    
    # 计算均值和标准差
    means = [np.mean(results[k]) for k in k_values]
    stds = [np.std(results[k]) for k in k_values]
    
    return means, stds

def load_and_process_data(ds_dir, label):
    """加载并处理单个数据集"""
    print(f"正在加载 {label} 数据...")
    with open(ds_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  数据类型: {type(data)}")

    # 提取scores特征
    scores_data = []
    if isinstance(data, list) and len(data) > 0:
        print(f"  数据长度: {len(data)}")
        
        # 删除responses特征并提取scores
        for item in data:
            print(item['prompt'])
            if 'responses' in item:
                del item['responses']
            if 'scores' in item:
                scores_data.append(item['scores'])
        
        print(f"  成功提取 {len(scores_data)} 个样本的scores")
        if len(scores_data) > 0:
            print(f"  每个样本的scores长度: {len(scores_data[0])}")

    else:
        print("  数据格式不是预期的列表格式")
        return None

    # 验证scores数据
    if not scores_data:
        print("  错误: 没有找到scores数据")
        return None

    return scores_data

# 加载两个数据集
all_scores_data = []
dataset_paths = [ds_dir1, ds_dir2]

for i, (ds_dir, label) in enumerate(zip(dataset_paths, labels)):
    scores_data = load_and_process_data(ds_dir, label)
    if scores_data is None:
        exit(1)
    all_scores_data.append(scores_data)

print(f"\n开始计算pass@k曲线...")

# 定义k值：2, 4, 8, 16, 32, 64, 128, 256
k_values = [2**i for i in range(9)]  # [2, 4, 8, 16, 32, 64, 128, 256]
print(f"k值: {k_values}")

# 设置随机种子以确保可重现性
random.seed(42)
np.random.seed(42)

# 计算每个数据集的pass@k曲线
all_means = []
all_stds = []

for i, (scores_data, label) in enumerate(zip(all_scores_data, labels)):
    print(f"\n计算 {label} 的pass@k曲线...")
    means, stds = compute_pass_k_curve(scores_data, k_values, num_shuffles=20)
    all_means.append(means)
    all_stds.append(stds)
    
    # 打印结果
    print(f"\n{label} pass@k结果:")
    for k, mean, std in zip(k_values, means, stds):
        print(f"  pass@{k}: {mean:.4f} ± {std:.4f}")

# 绘制曲线
plt.figure(figsize=(12, 7))

# 绘制两个数据集的曲线
for i, (means, stds, label, color) in enumerate(zip(all_means, all_stds, labels, colors)):
    # 计算上下边界
    upper_bounds = [mean + std for mean, std in zip(means, stds)]
    lower_bounds = [mean - std for mean, std in zip(means, stds)]
    
    # 绘制主曲线
    plt.plot(k_values, means, marker='o', linewidth=2, markersize=6, color=color, label=label)
    
    # 绘制阴影误差区域
    plt.fill_between(k_values, lower_bounds, upper_bounds, alpha=0.2, color=color)

plt.xlabel('k (Number of Attempts)', fontsize=18)
plt.ylabel('Accuracy (Pass@k)', fontsize=18)
plt.grid(True, alpha=0.3)
plt.xscale('log', base=2)
plt.xticks(k_values, [str(k) for k in k_values])
plt.ylim(0, 1)
plt.legend(fontsize=14, loc='lower right')

# 为每条曲线添加数值标签
for i, (means, label, color) in enumerate(zip(all_means, labels, colors)):
    for j, (k, mean) in enumerate(zip(k_values, means)):
        # 交替显示标签位置，避免重叠
        offset_y = 15 if i == 0 else -20
        plt.annotate(f'{mean:.3f}', (k, mean), textcoords="offset points", 
                    xytext=(0, offset_y), ha='center', fontsize=8, color=color)

plt.tight_layout()
plt.savefig('./pass_k_curve2.png', bbox_inches='tight')

plt.show()

print(f"\n图表已保存到: /home/wx13/reinforceflow/eval_bench/pass_k_curve.png")
print("处理完成！")
