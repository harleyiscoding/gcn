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

def load_flops_data(flops_dir, num_epochs=200):
    """加载所有epoch的FLOPs数据"""
    flops_data = []
    for epoch in range(1, num_epochs + 1):
        flops_file = os.path.join(flops_dir, f'flops_epoch_{epoch}.json')
        if os.path.exists(flops_file):
            with open(flops_file, 'r') as f:
                epoch_data = json.load(f)
                epoch_data['epoch'] = epoch
                flops_data.append(epoch_data)
        else:
            print(f"Warning: {flops_file} not found")
    
    return flops_data

def calculate_amir_trends(flops_data):
    """计算AMIR趋势"""
    epochs = [data['epoch'] for data in flops_data]
    amir_values = [data.get('amir', 0) for data in flops_data]
    
    return epochs, amir_values

def main():
    # 假设当前目录结构和perf分析脚本一致
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = 'pubmed'  # 可根据需要修改
    base_dir = os.path.join(current_dir, 'results', dataset)
    flops_dir = find_latest_flops_dir(base_dir)
    
    print(f"Loading FLOPs data from: {flops_dir}")
    
    # 加载数据集统计信息
    dataset_stats_file = os.path.join(flops_dir, 'dataset_stats.json')
    if os.path.exists(dataset_stats_file):
        with open(dataset_stats_file, 'r') as f:
            dataset_stats = json.load(f)
        print(f"Dataset stats: {dataset_stats}")
    
    # 加载FLOPs摘要
    summary_file = os.path.join(flops_dir, 'flops_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        total_flops = summary['total_flops']
        total_amir = summary['total_amir']
        print(f"Total AMIR: {total_amir:.4f}")
    else:
        # 如果没有摘要文件，使用最后一个epoch的数据
        flops_file = os.path.join(flops_dir, 'flops_epoch_200.json')
        if not os.path.exists(flops_file):
            raise FileNotFoundError(f"flops_epoch_200.json not found in {flops_dir}")
        with open(flops_file, 'r') as f:
            total_flops = json.load(f)
        total_amir = total_flops.get('amir', 0)

    # 加载所有epoch的FLOPs数据
    flops_data = load_flops_data(flops_dir)
    if not flops_data:
        raise FileNotFoundError(f"No FLOPs data found in {flops_dir}")
    
    print(f"Loaded {len(flops_data)} epochs of FLOPs data")

    # 画四个阶段的FLOPs对比柱状图
    phases = ['layer1_update', 'layer1_aggregate', 'layer2_update', 'layer2_aggregate']
    values = [total_flops[p] for p in phases]
    plt.figure(figsize=(10,6))
    plt.bar(phases, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('FLOPs')
    plt.title('FLOPs Comparison (Formula-based Calculation)')
    plt.ticklabel_format(style='plain', axis='y')  # 不用科学计数法
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(flops_dir, 'flops_phases_bar.png'), dpi=300)
    plt.close()

    # 画总update与总aggregate的FLOPs对比柱状图
    total_update = total_flops['layer1_update'] + total_flops['layer2_update']
    total_aggregate = total_flops['layer1_aggregate'] + total_flops['layer2_aggregate']
    plt.figure(figsize=(8,6))
    plt.bar(['Update', 'Aggregate'], [total_update, total_aggregate], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Total FLOPs')
    plt.title('FLOPs Comparison: Update vs Aggregate')
    plt.ticklabel_format(style='plain', axis='y')  # 不用科学计数法
    plt.tight_layout()
    plt.savefig(os.path.join(flops_dir, 'flops_update_vs_aggregate.png'), dpi=300)
    plt.close()

    # 画AMIR趋势图
    epochs, amir_values = calculate_amir_trends(flops_data)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, amir_values, color='#d62728', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('AMIR (Memory Access Intensity Ratio)')
    plt.title('AMIR Trend Over Training Epochs')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(flops_dir, 'amir_trend.png'), dpi=300)
    plt.close()

    # 写详细分析报告
    with open(os.path.join(flops_dir, 'flops_analysis_report.txt'), 'w') as f:
        f.write('FLOPs Analysis Report (Formula-based Calculation)\n')
        f.write('==============================================\n\n')
        
        # 数据集信息
        if 'dataset_stats' in locals():
            f.write('Dataset Statistics:\n')
            f.write('------------------\n')
            for key, value in dataset_stats.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')
        
        # FLOPs详情
        f.write('FLOPs Breakdown:\n')
        f.write('----------------\n')
        for p, v in zip(phases, values):
            f.write(f'{p}: {v:,}\n')
        f.write('\n')
        
        f.write(f'Total Update FLOPs: {total_update:,}\n')
        f.write(f'Total Aggregate FLOPs: {total_aggregate:,}\n')
        f.write(f'Update/Aggregate Ratio: {total_update/total_aggregate:.4f}\n')
        f.write(f'AMIR (Aggregate/Update): {total_amir:.4f}\n\n')
        
        # AMIR分析
        f.write('AMIR Analysis:\n')
        f.write('--------------\n')
        f.write(f'Average AMIR: {np.mean(amir_values):.4f}\n')
        f.write(f'Min AMIR: {np.min(amir_values):.4f}\n')
        f.write(f'Max AMIR: {np.max(amir_values):.4f}\n')
        f.write(f'AMIR Std Dev: {np.std(amir_values):.4f}\n\n')
        
        # 公式说明
        f.write('Formula Used:\n')
        f.write('-------------\n')
        f.write('MemoryAccess_agg ≈ adj.nnz × feature_dim\n')
        f.write('FLOPs_update ≈ num_nodes × feature_dim × hidden_dim\n')
        f.write('AMIR = MemoryAccess_aggregate / FLOPs_update\n')
    
    print(f"Analysis completed. Results saved to {flops_dir}")

if __name__ == '__main__':
    main() 