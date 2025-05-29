import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def main():
    # 设置目录
    json_base_dir = './results/json'
    analysis_base_dir = './results/analysis'
    
    # 加载最新的数据和对应的时间戳
    results, timestamp = load_latest_results(json_base_dir)
    
    # 创建当前运行的analysis目录
    analysis_run_dir = os.path.join(analysis_base_dir, timestamp)
    if not os.path.exists(analysis_run_dir):
        os.makedirs(analysis_run_dir)
    
    # 设置绘图风格
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 创建图表
    plot_major_stages_timing(results['timing_stats'], analysis_run_dir)
    plot_training_substages_timing(results['timing_stats'], analysis_run_dir)
    plot_epoch_timing(results['epoch_details'], analysis_run_dir)
    
    print("Analysis complete! Generated charts in analysis directory:")
    print(f"1. {os.path.join(analysis_run_dir, 'major_stages_timing.png')} - Major stages time distribution")
    print(f"2. {os.path.join(analysis_run_dir, 'training_validation_pie_chart.png')} - Training and Validation Time within Average Epoch")
    print(f"3. {os.path.join(analysis_run_dir, 'epoch_timing.png')} - Training, Validation, and Total Time per Epoch")

if __name__ == "__main__":
    main() 