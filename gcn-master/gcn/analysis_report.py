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
            
        # 定义要提取的指标，包括时间、指令、分支预测和内存访问相关的
        # 添加 cpu/mem-loads 和 cpu/mem-stores 事件
        metrics = [
            'cache-references',
            'cache-misses',
            'L1-dcache-load-misses',
            'L1-dcache-loads',
            'L1-dcache-stores',
            'L1-icache-load-misses',
            'LLC-loads',
            'LLC-load-misses',
            'instructions',
            'cpu-cycles',
            'branch-misses',
            'branch-loads',
            'cpu/mem-loads',  # Added memory load event
            'cpu/mem-stores', # Added memory store event
        ]

        # 提取每个指标的值
        for metric in metrics:
            # 使用通用的模式匹配数值和指标名称
            # 需要处理带有斜杠的事件名称，并考虑可能的单位和百分比信息
            # 尝试匹配以数值开头，后面跟指标名称的模式
            pattern = rf'(\d+(?:,\d+)*)\s+{re.escape(metric)}.*' # Use re.escape for metrics with special chars like /
            match = re.search(pattern, content)
            if match:
                # 移除逗号并转换为整数
                value = int(match.group(1).replace(',', ''))
                stats[metric] = value
            else:
                # 如果找不到精确匹配，尝试查找包含指标名称的行并提取数值
                # This might be less precise but can handle variations in perf output format
                fuzzy_pattern = rf'(\d+(?:,\d+)*).*{re.escape(metric)}.*'
                fuzzy_match = re.search(fuzzy_pattern, content, re.IGNORECASE)
                if fuzzy_match:
                     value = int(fuzzy_match.group(1).replace(',', ''))
                     stats[metric] = value
                else:
                    stats[metric] = 0
                
        # 提取执行时间
        time_match = re.search(r'(\d+\.\d+)\s+seconds time elapsed', content)
        if time_match:
            execution_time = float(time_match.group(1))
            stats['execution_time'] = execution_time
        else:
            # Fallback to task-clock if time elapsed is not found
            task_clock_match = re.search(r'(\d+\.\d+)\s+task-clock', content)
            if task_clock_match:
                execution_time = float(task_clock_match.group(1))
                stats['execution_time'] = execution_time / 1000 # task-clock is often in ms
            else:
                stats['execution_time'] = 0
                print("Warning: Could not extract execution time from perf stat output.")

        # 计算缺失率和IPC
        if stats.get('cache-references', 0) > 0:
            stats['cache_miss_rate'] = stats['cache-misses'] / stats['cache-references'] * 100
        else:
            stats['cache_miss_rate'] = 0
            
        if stats.get('L1-dcache-loads', 0) > 0:
            stats['l1_dcache_miss_rate'] = stats['L1-dcache-load-misses'] / stats['L1-dcache-loads'] * 100
        else:
            stats['l1_dcache_miss_rate'] = 0

        if stats.get('LLC-loads', 0) > 0:
            stats['llc_miss_rate'] = stats['LLC-load-misses'] / stats['LLC-loads'] * 100
        else:
            stats['llc_miss_rate'] = 0
            
        # 计算IPC
        if stats.get('cpu-cycles', 0) > 0:
            stats['ipc'] = stats['instructions'] / stats['cpu-cycles']
        else:
            stats['ipc'] = 0

        # 计算分支预测错误率
        if stats.get('branch-loads', 0) > 0:
            stats['branch_miss_rate'] = stats['branch-misses'] / stats['branch-loads'] * 100
        else:
            stats['branch_miss_rate'] = 0

        # 计算内存带宽 (估算)
        cache_line_size = 64 # Assuming a 64 byte cache line size
        total_memory_accesses = stats.get('cpu/mem-loads', 0) + stats.get('cpu/mem-stores', 0)
        estimated_bytes_transferred = total_memory_accesses * cache_line_size
        execution_time = stats.get('execution_time', 0)
        
        stats['estimated_memory_bandwidth_bytes_per_sec'] = 0
        stats['estimated_memory_bandwidth_gb_per_sec'] = 0

        if execution_time > 0:
            stats['estimated_memory_bandwidth_bytes_per_sec'] = estimated_bytes_transferred / execution_time
            stats['estimated_memory_bandwidth_gb_per_sec'] = stats['estimated_memory_bandwidth_bytes_per_sec'] / (1024**3) # Convert to GB/s
            
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
    
    # 定义L1和LLC相关的指标
    l1_metrics = [
        'L1-dcache-load-misses',
        'L1-dcache-loads',
        'L1-dcache-stores',
        'L1-icache-load-misses'
    ]
    llc_metrics = [
        'cache-references', # Note: perf's cache-references often refers to LLC
        'cache-misses',     # Note: perf's cache-misses often refers to LLC
        'LLC-loads',
        'LLC-load-misses'
    ]
    
    # 创建包含两个子图的figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 绘制L1缓存指标
    ax1 = axes[0]
    for metric in l1_metrics:
        values = [stage_stats[stage][metric] for stage in stages]
        ax1.plot(stages, values, marker='o', label=metric)
    
    ax1.set_ylabel('Count (L1 Cache)')
    ax1.set_title('L1 Cache Metrics Across Stages')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True)
    
    # 绘制LLC缓存指标
    ax2 = axes[1]
    for metric in llc_metrics:
        values = [stage_stats[stage][metric] for stage in stages]
        ax2.plot(stages, values, marker='o', label=metric)
    
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('Count (LLC)')
    ax2.set_title('LLC Cache Metrics Across Stages')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True)
    
    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表到analysis目录
    plt.savefig(os.path.join(analysis_run_dir, 'cache_metrics_stages_subplots.png'))
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

def find_latest_perf_dir(perf_base_dir):
    """找到最新的perf统计目录"""
    if not os.path.exists(perf_base_dir):
        raise FileNotFoundError(f"Perf base directory not found: {perf_base_dir}")
        
    # 列出所有条目，过滤出目录
    all_entries = [os.path.join(perf_base_dir, d) for d in os.listdir(perf_base_dir)]
    timestamp_dirs = sorted([d for d in all_entries if os.path.isdir(d)], key=os.path.getmtime, reverse=True)
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped perf directories found in: {perf_base_dir}")
    
    latest_timestamp_dir = timestamp_dirs[0]
    print(f"Found latest perf directory: {latest_timestamp_dir}")
    return latest_timestamp_dir

def plot_ipc_branch_miss_rate_stages(stage_stats, analysis_run_dir):
    """绘制各阶段IPC和分支预测错误率折线图"""
    stages = list(stage_stats.keys())
    ipcs = [stage_stats[stage]['ipc'] for stage in stages]
    branch_miss_rates = [stage_stats[stage]['branch_miss_rate'] for stage in stages]
    
    plt.figure(figsize=(12, 6))
    
    # 绘制IPC
    plt.plot(stages, ipcs, marker='o', label='Instructions Per Cycle (IPC)')
    
    # 绘制分支预测错误率
    plt.plot(stages, branch_miss_rates, marker='o', label='Branch Miss Rate (%)')
    
    plt.title('IPC and Branch Miss Rate Across Stages')
    plt.xlabel('Stage')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    
    # 设置Y轴下限为0
    plt.ylim(ymin=0)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(analysis_run_dir, 'ipc_branch_miss_rate_stages.png'))
    plt.close()

def generate_performance_report(timing_stats, epoch_stats, stage_stats, analysis_run_dir):
    """生成性能指标的文本报告"""
    # Re-adding the report function to include bandwidth
    report_path = os.path.join(analysis_run_dir, 'performance_report.txt')
    
    with open(report_path, 'w') as f:
        # 写入时间统计
        f.write("=== 时间统计 ===\n\n")
        total_time = sum([
            timing_stats['data_loading_time'],
            timing_stats['preprocessing_time'],
            timing_stats['model_building_time'],
            timing_stats['total_training_time'],
            timing_stats['test_time']
        ])
        
        f.write(f"总运行时间: {total_time:.2f} 秒\n\n")
        f.write("各阶段时间分布:\n")
        f.write(f"- 数据加载: {timing_stats['data_loading_time']:.2f} 秒 ({timing_stats['data_loading_time']/total_time*100:.1f}%)\n")
        f.write(f"- 预处理: {timing_stats['preprocessing_time']:.2f} 秒 ({timing_stats['preprocessing_time']/total_time*100:.1f}%)\n")
        f.write(f"- 模型构建: {timing_stats['model_building_time']:.2f} 秒 ({timing_stats['model_building_time']/total_time*100:.1f}%)\n")
        f.write(f"- 训练: {timing_stats['total_training_time']:.2f} 秒 ({timing_stats['total_training_time']/total_time*100:.1f}%)\n")
        f.write(f"- 测试: {timing_stats['test_time']:.2f} 秒 ({timing_stats['test_time']/total_time*100:.1f}%)\n\n")
        
        f.write("训练阶段详细时间:\n")
        f.write(f"- 平均每个epoch时间: {timing_stats['average_epoch_time']:.2f} 秒\n")
        f.write(f"- 平均训练时间: {timing_stats['average_training_time']:.2f} 秒\n")
        f.write(f"- 平均验证时间: {timing_stats['average_validation_time']:.2f} 秒\n\n")

        # 写入内存带宽统计 (估算)
        f.write("=== 内存带宽统计 (估算) ===\n\n")
        # Assuming stage_stats contains overall stats or use a specific stage like 'training'
        # If stage_stats keys are stages and each has a perf result:
        if stage_stats:
             for stage, stats in stage_stats.items():
                  f.write(f"{stage}:\n")
                  f.write(f"- 估算内存加载/存储次数: {stats.get('cpu/mem-loads', 0) + stats.get('cpu/mem-stores', 0):,}\n")
                  f.write(f"- 执行时间: {stats.get('execution_time', 0):.4f} 秒\n")
                  f.write(f"- 估算内存带宽: {stats.get('estimated_memory_bandwidth_gb_per_sec', 0):.2f} GB/s\n\n")
        elif epoch_stats: # If only epoch stats are available, might report average or for a specific epoch
             # For simplicity, let's just report for the first epoch if stages are not available
             if epoch_stats:
                 stats = epoch_stats[0]
                 f.write(f"Epoch {stats['epoch']}:\n")
                 f.write(f"- 估算内存加载/存储次数: {stats.get('cpu/mem-loads', 0) + stats.get('cpu/mem-stores', 0):,}\n")
                 f.write(f"- 执行时间: {stats.get('execution_time', 0):.4f} 秒\n")
                 f.write(f"- 估算内存带宽: {stats.get('estimated_memory_bandwidth_gb_per_sec', 0):.2f} GB/s\n\n")
        else:
            f.write("No performance statistics available to estimate memory bandwidth.\n\n")
            
        # 写入缓存统计 (现有代码保留)
        if epoch_stats:
            f.write("=== 缓存统计 (按Epoch) ===\n\n")
            for stat in epoch_stats:
                f.write(f"Epoch {stat['epoch']}:\n")
                f.write(f"- 缓存引用: {stat['cache-references']:,}\n")
                f.write(f"- 缓存缺失: {stat['cache-misses']:,} ({stat['cache_miss_rate']:.1f}%)\n")
                f.write(f"- L1数据缓存加载: {stat['L1-dcache-loads']:,}\n")
                f.write(f"- L1数据缓存缺失: {stat['L1-dcache-load-misses']:,} ({stat['l1_dcache_miss_rate']:.1f}%)\n")
                f.write(f"- LLC加载: {stat['LLC-loads']:,}\n")
                f.write(f"- LLC缺失: {stat['LLC-load-misses']:,} ({stat['llc_miss_rate']:.1f}%)\n\n")
        
        if stage_stats:
            f.write("=== 缓存统计 (按阶段) ===\n\n")
            for stage, stat in stage_stats.items():
                f.write(f"{stage}:\n")
                f.write(f"- 缓存引用: {stat['cache-references']:,}\n")
                f.write(f"- 缓存缺失: {stat['cache-misses']:,} ({stat['cache_miss_rate']:.1f}%)\n")
                f.write(f"- L1数据缓存加载: {stat['L1-dcache-loads']:,}\n")
                f.write(f"- L1数据缓存缺失: {stat['L1-dcache-load-misses']:,} ({stat['l1_dcache_miss_rate']:.1f}%)\n")
                f.write(f"- LLC加载: {stat['LLC-loads']:,}\n")
                f.write(f"- LLC缺失: {stat['LLC-load-misses']:,} ({stat['llc_miss_rate']:.1f}%)\n\n")
    
    print(f"Performance report generated: {report_path}")

def main():
    # 设置目录
    json_base_dir = './results/json'
    analysis_base_dir = './results/analysis'
    perf_base_dir = './results/perfs'
    
    # 加载最新的数据和对应的时间戳
    try:
        results, timestamp = load_latest_results(json_base_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 创建当前运行的analysis目录
    analysis_run_dir = os.path.join(analysis_base_dir, timestamp)
    if not os.path.exists(analysis_run_dir):
        os.makedirs(analysis_run_dir)
    
    # 设置绘图风格
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 创建原有图表
    try:
        plot_major_stages_timing(results['timing_stats'], analysis_run_dir)
        plot_training_substages_timing(results['timing_stats'], analysis_run_dir)
        plot_epoch_timing(results['epoch_details'], analysis_run_dir)
    except Exception as e:
        print(f"Error creating timing plots: {e}")
    
    # 加载并分析性能统计
    try:
        perf_dir = find_latest_perf_dir(perf_base_dir)
        print(f"Loading performance statistics from {perf_dir}")
        epoch_stats, stage_stats = load_perf_stats(perf_dir)
        
        # 生成性能报告 (包含内存带宽)
        generate_performance_report(results['timing_stats'], epoch_stats, stage_stats, analysis_run_dir)
        
        if epoch_stats:
            print("Generating cache metrics plots for epochs...")
            try:
                plot_cache_metrics_epochs(epoch_stats, analysis_run_dir)
                plot_cache_miss_rates_epochs(epoch_stats, analysis_run_dir)
            except Exception as e:
                print(f"Error creating epoch cache plots: {e}")
        else:
            print("No epoch performance statistics found")
            
        if stage_stats:
            print("Generating cache metrics plots for stages...")
            try:
                plot_cache_metrics_stages(stage_stats, analysis_run_dir)
                plot_cache_miss_rates_stages(stage_stats, analysis_run_dir)
            except Exception as e:
                print(f"Error creating stage cache plots: {e}")
        else:
            print("No stage performance statistics found")
            
        # 绘制IPC和分支预测错误率图表
        if stage_stats:
            print("Generating IPC and Branch Miss Rate plot for stages...")
            try:
                plot_ipc_branch_miss_rate_stages(stage_stats, analysis_run_dir)
            except Exception as e:
                print(f"Error creating IPC and Branch Miss Rate plot: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing performance statistics: {e}")
    
    print("\nAnalysis complete! Generated charts and reports in analysis directory:")
    print(f"1. {os.path.join(analysis_run_dir, 'performance_report.txt')} - Detailed performance metrics (including estimated memory bandwidth)")
    print(f"2. {os.path.join(analysis_run_dir, 'major_stages_timing.png')} - Major stages time distribution")
    print(f"3. {os.path.join(analysis_run_dir, 'training_validation_pie_chart.png')} - Training and Validation Time within Average Epoch")
    print(f"4. {os.path.join(analysis_run_dir, 'epoch_timing.png')} - Training, Validation, and Total Time per Epoch")
    print(f"5. {os.path.join(analysis_run_dir, 'cache_metrics_epochs.png')} - Cache metrics across epochs")
    print(f"6. {os.path.join(analysis_run_dir, 'cache_miss_rates_epochs.png')} - Cache miss rates across epochs")
    print(f"7. {os.path.join(analysis_run_dir, 'cache_metrics_stages_subplots.png')} - Cache metrics across stages")
    print(f"8. {os.path.join(analysis_run_dir, 'cache_miss_rates_stages.png')} - Cache miss rates across stages")
    print(f"9. {os.path.join(analysis_run_dir, 'ipc_branch_miss_rate_stages.png')} - IPC and Branch Miss Rate across stages")

if __name__ == "__main__":
    main() 