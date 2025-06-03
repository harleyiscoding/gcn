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
            'LLC-load-misses',
            'instructions',
            'cpu-cycles',
            'branch-misses',
            'branch-loads'
        ]

        # 提取每个指标的值
        for metric in metrics:
            # 修改正则表达式以匹配带逗号的数字
            pattern = rf'(\d+(?:,\d+)*)\s+{re.escape(metric)}'
            match = re.search(pattern, content)
            if match:
                value = int(match.group(1).replace(',', ''))
                stats[metric] = value
            else:
                stats[metric] = 0
                
        # 提取执行时间
        time_match = re.search(r'(\d+\.\d+)\s+seconds time elapsed', content)
        if time_match:
            execution_time = float(time_match.group(1))
            stats['execution_time'] = execution_time
        else:
            task_clock_match = re.search(r'(\d+\.\d+)\s+msec task-clock', content)
            if task_clock_match:
                execution_time = float(task_clock_match.group(1)) / 1000
                stats['execution_time'] = execution_time
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

        # 使用LLC-load-misses估算内存带宽
        cache_line_size = 64  # 假设缓存行大小为64字节
        llc_misses = stats.get('LLC-load-misses', 0)
        execution_time = stats.get('execution_time', 0)
        
        # 计算内存访问量和带宽
        memory_bytes = llc_misses * cache_line_size
        memory_mb = memory_bytes / (1024 * 1024)  # 转换为MB
        
        stats['memory_bytes'] = memory_bytes
        stats['memory_mb'] = memory_mb
        
        if execution_time > 0:
            bandwidth_mb_per_sec = memory_mb / execution_time
            bandwidth_gb_per_sec = bandwidth_mb_per_sec / 1024  # 转换为GB/s
            stats['memory_bandwidth_mb_per_sec'] = bandwidth_mb_per_sec
            stats['memory_bandwidth_gb_per_sec'] = bandwidth_gb_per_sec
        else:
            stats['memory_bandwidth_mb_per_sec'] = 0
            stats['memory_bandwidth_gb_per_sec'] = 0
            
        return stats
    except Exception as e:
        print(f"Error parsing perf stats file {file_path}: {e}")
        import traceback
        traceback.print_exc()
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
                # 添加默认的传播时间
                stats['forward_time'] = 0.0
                stats['backward_time'] = 0.0
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

def plot_propagation_times(epoch_details, analysis_run_dir):
    """绘制每个epoch的前向传播和后向传播时间"""
    epochs = [detail['epoch'] for detail in epoch_details]
    forward_times = [detail['forward_time'] for detail in epoch_details]
    backward_times = [detail['backward_time'] for detail in epoch_details]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, forward_times, 'b-', label='Forward Propagation', marker='o', markersize=4)
    plt.plot(epochs, backward_times, 'r-', label='Backward Propagation', marker='^', markersize=4)
    
    plt.title('Forward and Backward Propagation Times per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表到analysis目录
    plt.savefig(os.path.join(analysis_run_dir, 'propagation_times.png'))
    plt.close()

def plot_propagation_time_pie(epoch_details, analysis_run_dir):
    """绘制前向传播和后向传播时间的占比饼图"""
    try:
        # 检查是否有前向和后向传播时间数据
        has_propagation_times = all('forward_time' in detail and 'backward_time' in detail for detail in epoch_details)
        
        if not has_propagation_times:
            print("Warning: Forward and backward propagation times not available in the data")
            return
            
        # 计算总时间
        forward_times = [detail['forward_time'] for detail in epoch_details]
        backward_times = [detail['backward_time'] for detail in epoch_details]
        
        total_forward = sum(forward_times)
        total_backward = sum(backward_times)
        
        if total_forward == 0 and total_backward == 0:
            print("Warning: Both forward and backward propagation times are zero")
            return
        
        # 创建饼图
        plt.figure(figsize=(8, 8))
        labels = ['Forward Propagation', 'Backward Propagation']
        sizes = [total_forward, total_backward]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # 突出显示前向传播
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # 确保饼图是圆的
        
        plt.title('Forward vs Backward Propagation Time Distribution')
        
        # 保存图表到analysis目录
        plt.savefig(os.path.join(analysis_run_dir, 'propagation_time_pie.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating propagation time pie chart: {e}")

def plot_total_propagation_time_pie(epoch_details, analysis_run_dir):
    """绘制整个训练过程中前向和后向传播时间的总占比饼图"""
    try:
        # 计算总时间
        total_forward = sum(detail['forward_time'] for detail in epoch_details)
        total_backward = sum(detail['backward_time'] for detail in epoch_details)
        total_time = total_forward + total_backward
        
        # 创建饼图
        plt.figure(figsize=(8, 8))
        labels = ['Forward Propagation', 'Backward Propagation']
        sizes = [total_forward, total_backward]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # 突出显示前向传播
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct=lambda p: f'{p:.1f}%\n({p*total_time/100:.2f}s)',
                shadow=True, startangle=90)
        plt.axis('equal')  # 确保饼图是圆的
        
        plt.title('Total Forward vs Backward Propagation Time Distribution')
        
        # 添加总时间信息
        plt.figtext(0.5, 0.01, f'Total Time: {total_time:.2f} seconds', 
                   ha='center', fontsize=12)
        
        # 保存图表到analysis目录
        plt.savefig(os.path.join(analysis_run_dir, 'total_propagation_time_pie.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating total propagation time pie chart: {e}")

def generate_performance_report(timing_stats, epoch_stats, stage_stats, analysis_run_dir):
    """生成性能指标的文本报告"""
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

        # 写入传播时间统计
        if epoch_stats:
            f.write("=== 传播时间统计 ===\n\n")
            forward_times = [stat['forward_time'] for stat in epoch_stats]
            backward_times = [stat['backward_time'] for stat in epoch_stats]
            f.write(f"平均前向传播时间: {np.mean(forward_times):.4f} 秒\n")
            f.write(f"平均后向传播时间: {np.mean(backward_times):.4f} 秒\n")
            f.write(f"前向/后向时间比: {np.mean(forward_times)/np.mean(backward_times):.2f}\n\n")

        # 写入内存带宽统计 (基于LLC-load-misses)
        f.write("=== 内存带宽统计 (基于LLC-load-misses) ===\n\n")
        if stage_stats:
            for stage, stats in stage_stats.items():
                f.write(f"{stage}:\n")
                f.write(f"- LLC未命中次数: {stats.get('LLC-load-misses', 0):,}\n")
                f.write(f"- 估算内存访问量: {stats.get('memory_mb', 0):.2f} MB\n")
                f.write(f"- 执行时间: {stats.get('execution_time', 0):.4f} 秒\n")
                f.write(f"- 估算内存带宽: {stats.get('memory_bandwidth_mb_per_sec', 0):.2f} MB/s ({stats.get('memory_bandwidth_gb_per_sec', 0):.2f} GB/s)\n\n")
        elif epoch_stats:
            if epoch_stats:
                stats = epoch_stats[0]
                f.write(f"Epoch {stats['epoch']}:\n")
                f.write(f"- LLC未命中次数: {stats.get('LLC-load-misses', 0):,}\n")
                f.write(f"- 估算内存访问量: {stats.get('memory_mb', 0):.2f} MB\n")
                f.write(f"- 执行时间: {stats.get('execution_time', 0):.4f} 秒\n")
                f.write(f"- 估算内存带宽: {stats.get('memory_bandwidth_mb_per_sec', 0):.2f} MB/s ({stats.get('memory_bandwidth_gb_per_sec', 0):.2f} GB/s)\n\n")
        else:
            f.write("No performance statistics available to estimate memory bandwidth.\n\n")
            
        # 写入缓存统计
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

def plot_propagation_metrics_bar_chart(perf_dir, analysis_run_dir):
    """绘制前向和后向传播的性能指标对比柱状图"""
    try:
        print(f"\nAnalyzing propagation metrics from directory: {perf_dir}")
        
        # 获取前向和后向传播的统计目录
        forward_dir = os.path.join(perf_dir, 'forward')
        backward_dir = os.path.join(perf_dir, 'backward')
        
        print(f"Looking for stats in:")
        print(f"- Forward directory: {forward_dir}")
        print(f"- Backward directory: {backward_dir}")
        
        if not os.path.exists(forward_dir):
            print(f"Error: Forward directory not found: {forward_dir}")
            return
        if not os.path.exists(backward_dir):
            print(f"Error: Backward directory not found: {backward_dir}")
            return
            
        # 读取所有前向和后向传播的统计文件
        forward_files = sorted(glob.glob(os.path.join(forward_dir, 'forward_stats_*.txt')))
        backward_files = sorted(glob.glob(os.path.join(backward_dir, 'backward_stats_*.txt')))
        
        print(f"\nFound files:")
        print(f"- Forward files: {len(forward_files)}")
        for f in forward_files:
            print(f"  - {os.path.basename(f)}")
        print(f"- Backward files: {len(backward_files)}")
        for f in backward_files:
            print(f"  - {os.path.basename(f)}")
        
        if not forward_files:
            print("Error: No forward propagation stats files found")
            return
        if not backward_files:
            print("Error: No backward propagation stats files found")
            return
            
        # 解析所有文件并计算平均值
        forward_stats = []
        backward_stats = []
        
        print("\nParsing forward propagation stats...")
        for f_file in forward_files:
            print(f"Processing: {os.path.basename(f_file)}")
            stats = parse_perf_stats(f_file)
            if stats:
                forward_stats.append(stats)
                print(f"  Successfully parsed {os.path.basename(f_file)}")
            else:
                print(f"  Failed to parse {os.path.basename(f_file)}")
                
        print("\nParsing backward propagation stats...")
        for b_file in backward_files:
            print(f"Processing: {os.path.basename(b_file)}")
            stats = parse_perf_stats(b_file)
            if stats:
                backward_stats.append(stats)
                print(f"  Successfully parsed {os.path.basename(b_file)}")
            else:
                print(f"  Failed to parse {os.path.basename(b_file)}")
        
        if not forward_stats:
            print("Error: Failed to parse any forward propagation stats")
            return
        if not backward_stats:
            print("Error: Failed to parse any backward propagation stats")
            return
            
        print(f"\nSuccessfully parsed:")
        print(f"- Forward stats: {len(forward_stats)} files")
        print(f"- Backward stats: {len(backward_stats)} files")
            
        # 计算平均值
        def calculate_average_stats(stats_list):
            avg_stats = {}
            for key in stats_list[0].keys():
                if isinstance(stats_list[0][key], (int, float)):
                    avg_stats[key] = np.mean([stat[key] for stat in stats_list])
            return avg_stats
            
        forward_avg = calculate_average_stats(forward_stats)
        backward_avg = calculate_average_stats(backward_stats)
        
        print("\nCalculated average metrics:")
        print("Forward propagation averages:")
        for key, value in forward_avg.items():
            print(f"- {key}: {value}")
        print("\nBackward propagation averages:")
        for key, value in backward_avg.items():
            print(f"- {key}: {value}")
        
        # 定义要对比的指标
        metrics = {
            'Cache Miss Rate (%)': ('cache_miss_rate', 'Cache Miss Rate'),
            'L1 DCache Miss Rate (%)': ('l1_dcache_miss_rate', 'L1 DCache Miss Rate'),
            'LLC Miss Rate (%)': ('llc_miss_rate', 'LLC Miss Rate'),
            'IPC': ('ipc', 'Instructions Per Cycle'),
            'Branch Miss Rate (%)': ('branch_miss_rate', 'Branch Miss Rate')
        }
        
        # 创建柱状图
        plt.figure(figsize=(15, 10))
        x = np.arange(len(metrics))
        width = 0.35
        
        forward_values = [forward_avg[metric[0]] for metric in metrics.values()]
        backward_values = [backward_avg[metric[0]] for metric in metrics.values()]
        
        print("\nPlotting metrics:")
        for i, (metric_name, (metric_key, _)) in enumerate(metrics.items()):
            print(f"- {metric_name}:")
            print(f"  Forward: {forward_values[i]:.2f}")
            print(f"  Backward: {backward_values[i]:.2f}")
        
        # 绘制柱状图
        plt.bar(x - width/2, forward_values, width, label='Forward Propagation', color='#ff9999')
        plt.bar(x + width/2, backward_values, width, label='Backward Propagation', color='#66b3ff')
        
        # 设置图表属性
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison: Forward vs Backward Propagation')
        plt.xticks(x, [metric[1] for metric in metrics.values()], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(forward_values):
            plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(backward_values):
            plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(analysis_run_dir, 'propagation_metrics_bar_chart.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"\nGenerated bar chart: {output_file}")
        
        # 生成详细的对比报告
        report_path = os.path.join(analysis_run_dir, 'propagation_metrics_comparison.txt')
        with open(report_path, 'w') as f:
            f.write("=== Forward vs Backward Propagation Performance Comparison ===\n\n")
            
            for metric_name, (metric_key, _) in metrics.items():
                forward_value = forward_avg[metric_key]
                backward_value = backward_avg[metric_key]
                ratio = forward_value / backward_value if backward_value != 0 else float('inf')
                
                f.write(f"{metric_name}:\n")
                f.write(f"- Forward: {forward_value:.4f}\n")
                f.write(f"- Backward: {backward_value:.4f}\n")
                f.write(f"- Forward/Backward Ratio: {ratio:.2f}\n\n")
            
            # 添加其他重要指标
            f.write("\nAdditional Metrics:\n")
            f.write(f"Cache References:\n")
            f.write(f"- Forward: {forward_avg['cache-references']:,.0f}\n")
            f.write(f"- Backward: {backward_avg['cache-references']:,.0f}\n\n")
            
            f.write(f"L1 DCache Loads:\n")
            f.write(f"- Forward: {forward_avg['L1-dcache-loads']:,.0f}\n")
            f.write(f"- Backward: {backward_avg['L1-dcache-loads']:,.0f}\n\n")
            
            f.write(f"LLC Loads:\n")
            f.write(f"- Forward: {forward_avg['LLC-loads']:,.0f}\n")
            f.write(f"- Backward: {backward_avg['LLC-loads']:,.0f}\n\n")
            
            f.write(f"Instructions:\n")
            f.write(f"- Forward: {forward_avg['instructions']:,.0f}\n")
            f.write(f"- Backward: {backward_avg['instructions']:,.0f}\n\n")
            
            f.write(f"CPU Cycles:\n")
            f.write(f"- Forward: {forward_avg['cpu-cycles']:,.0f}\n")
            f.write(f"- Backward: {backward_avg['cpu-cycles']:,.0f}\n")
            
        print(f"Generated comparison report: {report_path}")
            
    except Exception as e:
        print(f"Error creating propagation metrics bar chart: {e}")
        import traceback
        traceback.print_exc()

def plot_memory_bandwidth_comparison(perf_dir, analysis_run_dir):
    """绘制各个阶段的内存带宽对比柱状图"""
    try:
        print(f"\nAnalyzing memory bandwidth from directory: {perf_dir}")
        
        # 获取各个阶段的统计文件
        stage_files = {
            'Model Building': os.path.join(perf_dir, 'model_building_memory_stats.txt'),
            'Preprocessing': os.path.join(perf_dir, 'preprocessing_memory_stats.txt'),
            'Training': os.path.join(perf_dir, 'training_memory_stats.txt'),
            'Validation': os.path.join(perf_dir, 'validation_memory_stats.txt'),
            'Testing': os.path.join(perf_dir, 'testing_memory_stats.txt')
        }
        
        # 获取前向和后向传播的统计目录
        forward_dir = os.path.join(perf_dir, 'forward')
        backward_dir = os.path.join(perf_dir, 'backward')
        
        # 读取所有前向和后向传播的统计文件
        forward_files = sorted(glob.glob(os.path.join(forward_dir, 'forward_stats_*.txt')))
        backward_files = sorted(glob.glob(os.path.join(backward_dir, 'backward_stats_*.txt')))
        
        # 解析各个阶段的统计
        stage_bandwidths = {}
        for stage_name, file_path in stage_files.items():
            if os.path.exists(file_path):
                stats = parse_perf_stats(file_path)
                if stats:
                    stage_bandwidths[stage_name] = stats.get('memory_bandwidth_mb_per_sec', 0)
        
        # 解析前向和后向传播的统计
        forward_bandwidths = []
        backward_bandwidths = []
        
        for f_file in forward_files:
            stats = parse_perf_stats(f_file)
            if stats:
                forward_bandwidths.append(stats.get('memory_bandwidth_mb_per_sec', 0))
                
        for b_file in backward_files:
            stats = parse_perf_stats(b_file)
            if stats:
                backward_bandwidths.append(stats.get('memory_bandwidth_mb_per_sec', 0))
        
        # 计算前向和后向传播的平均带宽
        if forward_bandwidths:
            stage_bandwidths['Forward Propagation'] = np.mean(forward_bandwidths)
        if backward_bandwidths:
            stage_bandwidths['Backward Propagation'] = np.mean(backward_bandwidths)
        
        # 创建柱状图
        plt.figure(figsize=(15, 8))
        
        # 设置柱状图的位置和宽度
        x = np.arange(len(stage_bandwidths))
        width = 0.6
        
        # 绘制柱状图
        bars = plt.bar(x, list(stage_bandwidths.values()), width)
        
        # 设置图表属性
        plt.xlabel('Stage')
        plt.ylabel('Memory Bandwidth (MB/s)')
        plt.title('Memory Bandwidth Comparison Across Stages')
        plt.xticks(x, stage_bandwidths.keys(), rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(analysis_run_dir, 'memory_bandwidth_comparison.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"\nGenerated memory bandwidth comparison chart: {output_file}")
        
        # 生成详细的带宽报告
        report_path = os.path.join(analysis_run_dir, 'memory_bandwidth_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== Memory Bandwidth Analysis Report ===\n\n")
            
            f.write("Memory Bandwidth by Stage:\n")
            for stage, bandwidth in stage_bandwidths.items():
                f.write(f"{stage}:\n")
                f.write(f"- Bandwidth: {bandwidth:.2f} MB/s ({bandwidth/1024:.2f} GB/s)\n\n")
            
            if forward_bandwidths and backward_bandwidths:
                f.write("\nPropagation Bandwidth Statistics:\n")
                f.write(f"Forward Propagation:\n")
                f.write(f"- Average: {np.mean(forward_bandwidths):.2f} MB/s\n")
                f.write(f"- Min: {np.min(forward_bandwidths):.2f} MB/s\n")
                f.write(f"- Max: {np.max(forward_bandwidths):.2f} MB/s\n\n")
                
                f.write(f"Backward Propagation:\n")
                f.write(f"- Average: {np.mean(backward_bandwidths):.2f} MB/s\n")
                f.write(f"- Min: {np.min(backward_bandwidths):.2f} MB/s\n")
                f.write(f"- Max: {np.max(backward_bandwidths):.2f} MB/s\n\n")
                
                ratio = np.mean(forward_bandwidths) / np.mean(backward_bandwidths)
                f.write(f"Forward/Backward Bandwidth Ratio: {ratio:.2f}\n")
        
        print(f"Generated memory bandwidth report: {report_path}")
            
    except Exception as e:
        print(f"Error creating memory bandwidth comparison chart: {e}")
        import traceback
        traceback.print_exc()

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
        plot_propagation_times(results['epoch_details'], analysis_run_dir)
        plot_propagation_time_pie(results['epoch_details'], analysis_run_dir)
        plot_total_propagation_time_pie(results['epoch_details'], analysis_run_dir)
    except Exception as e:
        print(f"Error creating timing plots: {e}")
    
    # 加载并分析性能统计
    try:
        perf_dir = find_latest_perf_dir(perf_base_dir)
        print(f"Loading performance statistics from {perf_dir}")
        epoch_stats, stage_stats = load_perf_stats(perf_dir)
        
        # 生成性能报告 (包含内存带宽)
        generate_performance_report(results['timing_stats'], epoch_stats, stage_stats, analysis_run_dir)
        
        # 绘制前向和后向传播的性能指标对比柱状图
        plot_propagation_metrics_bar_chart(perf_dir, analysis_run_dir)
        
        # 绘制内存带宽对比图
        plot_memory_bandwidth_comparison(perf_dir, analysis_run_dir)
        
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
    print(f"10. {os.path.join(analysis_run_dir, 'propagation_times.png')} - Forward and Backward Propagation Times per Epoch")
    print(f"11. {os.path.join(analysis_run_dir, 'propagation_time_pie.png')} - Forward vs Backward Propagation Time Distribution")
    print(f"12. {os.path.join(analysis_run_dir, 'total_propagation_time_pie.png')} - Total Forward vs Backward Propagation Time Distribution")
    print(f"13. {os.path.join(analysis_run_dir, 'propagation_metrics_bar_chart.png')} - Forward vs Backward Propagation Performance Metrics Comparison")
    print(f"14. {os.path.join(analysis_run_dir, 'propagation_metrics_comparison.txt')} - Detailed Forward vs Backward Propagation Performance Metrics Report")
    print(f"15. {os.path.join(analysis_run_dir, 'memory_bandwidth_comparison.png')} - Memory Bandwidth Comparison Across Stages")
    print(f"16. {os.path.join(analysis_run_dir, 'memory_bandwidth_report.txt')} - Detailed Memory Bandwidth Analysis Report")

if __name__ == "__main__":
    main() 