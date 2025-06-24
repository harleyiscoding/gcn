import os
import json
import matplotlib.pyplot as plt
import numpy as np

def find_latest_flops_dir(base_dir):
    # 找到tf-flops下最新的时间戳目录
    tf_flops_dir = os.path.join(base_dir, 'tf-flops')
    if not os.path.exists(tf_flops_dir):
        raise FileNotFoundError(f"tf-flops dir not found: {tf_flops_dir}")
    timestamps = [d for d in os.listdir(tf_flops_dir) if os.path.isdir(os.path.join(tf_flops_dir, d))]
    if not timestamps:
        raise FileNotFoundError(f"No timestamp dirs in {tf_flops_dir}")
    latest = max(timestamps)
    return os.path.join(tf_flops_dir, latest)

def main():
    # 假设当前目录结构和perf分析脚本一致
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = 'pubmed'  # 可根据需要修改
    base_dir = os.path.join(current_dir, 'results', dataset)
    flops_dir = find_latest_flops_dir(base_dir)
    flops_file = os.path.join(flops_dir, 'flops_epoch_200.json')
    if not os.path.exists(flops_file):
        raise FileNotFoundError(f"flops_epoch_200.json not found in {flops_dir}")
    with open(flops_file, 'r') as f:
        flops = json.load(f)

    # layer2_aggregate需要乘以200
    flops['layer2_aggregate'] = flops['layer2_aggregate'] * 200

    # 画四个阶段的FLOPs对比柱状图
    phases = ['layer1_update', 'layer1_aggregate', 'layer2_update', 'layer2_aggregate']
    values = [flops[p] for p in phases]
    plt.figure(figsize=(8,6))
    plt.bar(phases, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('FLOPs')
    plt.title('FLOPs Comparison')
    plt.ticklabel_format(style='plain', axis='y')  # 不用科学计数法
    plt.tight_layout()
    plt.savefig(os.path.join(flops_dir, 'flops_phases_bar.png'), dpi=300)
    plt.close()

    # 画总update与总aggregate的FLOPs对比柱状图
    total_update = flops['layer1_update'] + flops['layer2_update']
    total_aggregate = flops['layer1_aggregate'] + flops['layer2_aggregate']
    plt.figure(figsize=(6,6))
    plt.bar(['Update', 'Aggregate'], [total_update, total_aggregate], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Total FLOPs')
    plt.title('FLOPs Comparison')
    plt.ticklabel_format(style='plain', axis='y')  # 不用科学计数法
    plt.tight_layout()
    plt.savefig(os.path.join(flops_dir, 'flops_update_vs_aggregate.png'), dpi=300)
    plt.close()

    # 写简单分析报告
    with open(os.path.join(flops_dir, 'flops_analysis_report.txt'), 'w') as f:
        f.write('FLOPs Analysis Report (Epoch 200, layer2_aggregate x200)\n')
        f.write('==============================================\n\n')
        for p, v in zip(phases, values):
            f.write(f'{p}: {v}\n')
        f.write('\n')
        f.write(f'Total Update FLOPs: {total_update}\n')
        f.write(f'Total Aggregate FLOPs: {total_aggregate}\n')
        f.write(f'Update/Aggregate Ratio: {total_update/total_aggregate:.2f}\n')
    print(f"Analysis completed. Results saved to {flops_dir}")

if __name__ == '__main__':
    main() 