import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_latest_results(json_base_dir='./results/json'):
    """加载最新的结果文件"""
    if not os.path.exists(json_base_dir):
        raise FileNotFoundError(f"JSON results base directory not found: {json_base_dir}")
        
    # 列出所有条目，过滤出目录
    all_entries = [os.path.join(json_base_dir, d) for d in os.listdir(json_base_dir)]
    timestamp_dirs = sorted([d for d in all_entries if os.path.isdir(d)], key=os.path.getmtime, reverse=True)
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped results directories found in the json directory: {json_base_dir}")
    
    latest_timestamp_dir = timestamp_dirs[0]
    print(f"Found latest results directory: {latest_timestamp_dir}")
    
    # 在最新的时间戳目录中找到结果文件
    json_files = glob.glob(os.path.join(latest_timestamp_dir, 'results_*.json'))
    if not json_files:
        raise FileNotFoundError(f"No results files found in the latest directory: {latest_timestamp_dir}")
    
    # 获取最新的文件（如果一个时间戳目录中有多个文件，虽然 train.py 只保存一个）
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    with open(latest_file, 'r') as f:
        return json.load(f), os.path.basename(latest_timestamp_dir)

def plot_major_stages_timing(timing_stats, analysis_run_dir):
    """绘制主要阶段的时间占比柱状图"""
    # 提取主要阶段的时间
    major_stages = {
        'Data Loading': timing_stats['data_loading_time'],
        'Preprocessing': timing_stats['preprocessing_time'],
        'Model Building': timing_stats['model_building_time'],
        'Training': timing_stats['total_training_time'],
        'Testing': timing_stats['test_time']
    }
    
    # 计算总时间
    total_time = sum(major_stages.values())
    
    # 计算百分比
    percentages = {k: (v/total_time)*100 for k, v in major_stages.items()}
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(percentages.keys(), percentages.values())
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.title('Major Stages Time Distribution')
    plt.ylabel('Time Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图表到analysis目录
    plt.savefig(os.path.join(analysis_run_dir, 'major_stages_timing.png'))
    plt.close()

def plot_training_substages_timing(timing_stats, analysis_run_dir):
    """绘制训练和验证时间在Average Epoch中的占比饼状图"""
    # 提取训练和验证时间
    train_time = timing_stats['average_training_time']
    val_time = timing_stats['average_validation_time']
    epoch_time = timing_stats['average_epoch_time']
    
    # 计算占Average Epoch的百分比
    # 避免除以零
    if epoch_time == 0:
        print("Warning: Average epoch time is zero, cannot create pie chart.")
        return
        
    percentages = [
        (train_time / epoch_time) * 100,
        (val_time / epoch_time) * 100
    ]
    labels = ['Average Training', 'Average Validation']
    
    # 创建饼状图
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Training and Validation Time within Average Epoch')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # 保存图表到analysis目录
    plt.savefig(os.path.join(analysis_run_dir, 'training_validation_pie_chart.png'))
    plt.close()

def plot_epoch_timing(epoch_details, analysis_run_dir):
    """绘制每个epoch的训练时间、验证时间和总时间折线图"""
    epochs = [detail['epoch'] for detail in epoch_details]
    train_times = [detail['train_time'] for detail in epoch_details]
    val_times = [detail['val_time'] for detail in epoch_details]
    total_times = [detail['total_time'] for detail in epoch_details]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_times, 'b-', label='Training Time', marker='o', markersize=4)
    plt.plot(epochs, val_times, 'g-', label='Validation Time', marker='^', markersize=4)
    plt.plot(epochs, total_times, 'r-', label='Total Time', marker='s', markersize=4)
    
    plt.title('Training, Validation, and Total Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表到analysis目录
    plt.savefig(os.path.join(analysis_run_dir, 'epoch_timing.png'))
    plt.close()

def parse_perf_stats(file_path):
    """解析perf stat输出文件"""
    stats = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 定义要提取的指标
        metrics = [
            'cache-references',
            'cache-misses',
            'L1-dcache-load-misses',
            'L1-dcache-loads',
            'L1-dcache-stores',
            'L1-icache-load-misses',
            'LLC-loads',
            'LLC-load-misses'
        ]
        
        # 提取每个指标的值
        for metric in metrics:
            pattern = rf'(\d+(?:,\d+)*)\s+{metric}'
            match = re.search(pattern, content)
            if match:
                # 移除逗号并转换为整数
                value = int(match.group(1).replace(',', ''))
                stats[metric] = value
            else:
                stats[metric] = 0
                
        # 计算缺失率
        if stats['cache-references'] > 0:
            stats['cache_miss_rate'] = stats['cache-misses'] / stats['cache-references'] * 100
        if stats['L1-dcache-loads'] > 0:
            stats['l1_dcache_miss_rate'] = stats['L1-dcache-load-misses'] / stats['L1-dcache-loads'] * 100
        if stats['LLC-loads'] > 0:
            stats['llc_miss_rate'] = stats['LLC-load-misses'] / stats['LLC-loads'] * 100
            
        return stats
    except Exception as e:
        print(f"Error parsing perf stats file {file_path}: {e}")
        return None

def load_perf_stats(perf_dir):
    """加载所有perf统计文件"""
    epoch_stats = []
    stage_stats = {}
    
    # 读取epoch统计
    for i in range(10):  # 前10个epoch
        file_path = os.path.join(perf_dir, f'epoch_{i}_memory_stats.txt')
        if os.path.exists(file_path):
            stats = parse_perf_stats(file_path)
            if stats:
                stats['epoch'] = i
                epoch_stats.append(stats)
    
    # 读取阶段统计
    stages = ['model_building', 'preprocessing', 'training', 'validation', 'testing']
    for stage in stages:
        file_path = os.path.join(perf_dir, f'{stage}_memory_stats.txt')
        if os.path.exists(file_path):
            stats = parse_perf_stats(file_path)
            if stats:
                stage_stats[stage] = stats
    
    return epoch_stats, stage_stats

def plot_cache_metrics_epochs(epoch_stats, analysis_run_dir):
    """绘制epoch缓存指标折线图"""
    epochs = [stat['epoch'] for stat in epoch_stats]
    metrics = ['cache-references', 'cache-misses', 'L1-dcache-load-misses', 
              'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses',
              'LLC-loads', 'LLC-load-misses']
    
    plt.figure(figsize=(15, 8))
    for metric in metrics:
        values = [stat[metric] for stat in epoch_stats]
        plt.plot(epochs, values, marker='o', label=metric)
    
    plt.title('Cache Metrics Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_run_dir, 'cache_metrics_epochs.png'))
    plt.close()

def plot_cache_miss_rates_epochs(epoch_stats, analysis_run_dir):
    """绘制epoch缓存缺失率柱状图"""
    epochs = [stat['epoch'] for stat in epoch_stats]
    miss_rates = {
        'Cache Miss Rate': [stat['cache_miss_rate'] for stat in epoch_stats],
        'L1 DCache Miss Rate': [stat['l1_dcache_miss_rate'] for stat in epoch_stats],
        'LLC Miss Rate': [stat['llc_miss_rate'] for stat in epoch_stats]
    }
    
    x = np.arange(len(epochs))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    for i, (name, rates) in enumerate(miss_rates.items()):
        plt.bar(x + i*width, rates, width, label=name)
    
    plt.title('Cache Miss Rates Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Miss Rate (%)')
    plt.xticks(x + width, epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_run_dir, 'cache_miss_rates_epochs.png'))
    plt.close()

def plot_cache_metrics_stages(stage_stats, analysis_run_dir):
    """绘制各阶段缓存指标折线图"""
    stages = list(stage_stats.keys())
    metrics = ['cache-references', 'cache-misses', 'L1-dcache-load-misses', 
              'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses',
              'LLC-loads', 'LLC-load-misses']
    
    plt.figure(figsize=(15, 8))
    for metric in metrics:
        values = [stage_stats[stage][metric] for stage in stages]
        plt.plot(stages, values, marker='o', label=metric)
    
    plt.title('Cache Metrics Across Stages')
    plt.xlabel('Stage')
    plt.ylabel('Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_run_dir, 'cache_metrics_stages.png'))
    plt.close()

def plot_cache_miss_rates_stages(stage_stats, analysis_run_dir):
    """绘制各阶段缓存缺失率柱状图"""
    stages = list(stage_stats.keys())
    miss_rates = {
        'Cache Miss Rate': [stage_stats[stage]['cache_miss_rate'] for stage in stages],
        'L1 DCache Miss Rate': [stage_stats[stage]['l1_dcache_miss_rate'] for stage in stages],
        'LLC Miss Rate': [stage_stats[stage]['llc_miss_rate'] for stage in stages]
    }
    
    x = np.arange(len(stages))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    for i, (name, rates) in enumerate(miss_rates.items()):
        plt.bar(x + i*width, rates, width, label=name)
    
    plt.title('Cache Miss Rates Across Stages')
    plt.xlabel('Stage')
    plt.ylabel('Miss Rate (%)')
    plt.xticks(x + width, stages, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_run_dir, 'cache_miss_rates_stages.png'))
    plt.close()

def main():
    # 设置目录
    json_base_dir = './results/json'
    analysis_base_dir = './results/analysis'
    perf_base_dir = './results/perfs'
    
    # 加载最新的数据和对应的时间戳
    results, timestamp = load_latest_results(json_base_dir)
    
    # 创建当前运行的analysis目录
    analysis_run_dir = os.path.join(analysis_base_dir, timestamp)
    if not os.path.exists(analysis_run_dir):
        os.makedirs(analysis_run_dir)
    
    # 设置绘图风格
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 创建原有图表
    plot_major_stages_timing(results['timing_stats'], analysis_run_dir)
    plot_training_substages_timing(results['timing_stats'], analysis_run_dir)
    plot_epoch_timing(results['epoch_details'], analysis_run_dir)
    
    # 加载并分析性能统计
    perf_dir = os.path.join(perf_base_dir, timestamp)
    if os.path.exists(perf_dir):
        print(f"Loading performance statistics from {perf_dir}")
        epoch_stats, stage_stats = load_perf_stats(perf_dir)
        
        if epoch_stats:
            print("Generating cache metrics plots for epochs...")
            plot_cache_metrics_epochs(epoch_stats, analysis_run_dir)
            plot_cache_miss_rates_epochs(epoch_stats, analysis_run_dir)
        else:
            print("No epoch performance statistics found")
            
        if stage_stats:
            print("Generating cache metrics plots for stages...")
            plot_cache_metrics_stages(stage_stats, analysis_run_dir)
            plot_cache_miss_rates_stages(stage_stats, analysis_run_dir)
        else:
            print("No stage performance statistics found")
    else:
        print(f"Performance statistics directory not found: {perf_dir}")
    
    print("\nAnalysis complete! Generated charts in analysis directory:")
    print(f"1. {os.path.join(analysis_run_dir, 'major_stages_timing.png')} - Major stages time distribution")
    print(f"2. {os.path.join(analysis_run_dir, 'training_validation_pie_chart.png')} - Training and Validation Time within Average Epoch")
    print(f"3. {os.path.join(analysis_run_dir, 'epoch_timing.png')} - Training, Validation, and Total Time per Epoch")
    print(f"4. {os.path.join(analysis_run_dir, 'cache_metrics_epochs.png')} - Cache metrics across epochs")
    print(f"5. {os.path.join(analysis_run_dir, 'cache_miss_rates_epochs.png')} - Cache miss rates across epochs")
    print(f"6. {os.path.join(analysis_run_dir, 'cache_metrics_stages.png')} - Cache metrics across stages")
    print(f"7. {os.path.join(analysis_run_dir, 'cache_miss_rates_stages.png')} - Cache miss rates across stages")

if __name__ == "__main__":
    main() 