#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_perf_stats(stats_file):
    """解析性能统计文件"""
    try:
        with open(stats_file, 'r') as f:
            content = f.read()
            
        # 检查是否所有事件都是 not counted
        if 'not counted' in content and content.count('not counted') >= 5:
            return None
            
        # 解析缓存相关指标
        cache_misses = re.search(r'(\d+(?:,\d+)*)\s+cache-misses', content)
        cache_refs = re.search(r'(\d+(?:,\d+)*)\s+cache-references', content)
        l1_misses = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-load-misses', content)
        l1_loads = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-loads', content)
        llc_misses = re.search(r'(\d+(?:,\d+)*)\s+LLC-load-misses', content)
        llc_loads = re.search(r'(\d+(?:,\d+)*)\s+LLC-loads', content)
        
        # 解析分支预测指标
        branch_misses = re.search(r'(\d+(?:,\d+)*)\s+branch-misses', content)
        branch_loads = re.search(r'(\d+(?:,\d+)*)\s+branch-loads', content)
        
        # 解析IPC相关指标
        instructions = re.search(r'(\d+(?:,\d+)*)\s+instructions', content)
        cpu_cycles = re.search(r'(\d+(?:,\d+)*)\s+cpu-cycles', content)
        
        # 解析执行时间
        time_elapsed = re.search(r'(\d+\.\d+)\s+seconds time elapsed', content)
        
        stats = {}
        
        # 检查每个指标是否被标记为 not counted
        def is_not_counted(pattern):
            return bool(re.search(f'{pattern}.*not counted', content))
        
        # 计算缓存缺失率
        if cache_misses and cache_refs and not is_not_counted('cache-misses'):
            misses = int(cache_misses.group(1).replace(',', ''))
            refs = int(cache_refs.group(1).replace(',', ''))
            stats['cache_miss_rate'] = (misses / refs * 100) if refs > 0 else 0
            
        if l1_misses and l1_loads and not is_not_counted('L1-dcache-load-misses'):
            misses = int(l1_misses.group(1).replace(',', ''))
            loads = int(l1_loads.group(1).replace(',', ''))
            stats['l1_miss_rate'] = (misses / loads * 100) if loads > 0 else 0
            stats['l1_miss_count'] = misses  # 保存L1缺失数量
            
        if llc_misses and llc_loads and not is_not_counted('LLC-load-misses'):
            misses = int(llc_misses.group(1).replace(',', ''))
            loads = int(llc_loads.group(1).replace(',', ''))
            stats['llc_miss_rate'] = (misses / loads * 100) if loads > 0 else 0
            stats['llc_miss_count'] = misses  # 保存LLC缺失数量
            
        # 计算分支预测失败率
        if branch_misses and branch_loads and not is_not_counted('branch-misses'):
            misses = int(branch_misses.group(1).replace(',', ''))
            loads = int(branch_loads.group(1).replace(',', ''))
            stats['branch_miss_rate'] = (misses / loads * 100) if loads > 0 else 0
            
        # 计算IPC
        if instructions and cpu_cycles and not is_not_counted('instructions') and not is_not_counted('cpu-cycles'):
            instr = int(instructions.group(1).replace(',', ''))
            cycles = int(cpu_cycles.group(1).replace(',', ''))
            stats['ipc'] = instr / cycles if cycles > 0 else 0
            
        # 计算内存带宽
        if llc_misses and time_elapsed and not is_not_counted('LLC-load-misses'):
            misses = int(llc_misses.group(1).replace(',', ''))
            time = float(time_elapsed.group(1))
            # 假设每次缓存未命中访问64字节
            memory_bytes = misses * 64
            stats['memory_bandwidth'] = memory_bytes / time / (1024 * 1024)  # MB/s
            
        # 只有当至少有一个有效指标时才返回统计结果
        if stats:
            return stats
        return None
    except Exception as e:
        print(f"Error parsing stats file {stats_file}: {e}")
        return None

def analyze_performance_data(perf_dirs):
    """分析性能数据"""
    all_stats = []
    
    for perf_dir in perf_dirs:
        forward_dir = os.path.join(perf_dir, 'forward')
        backward_dir = os.path.join(perf_dir, 'backward')
        
        # 读取前向传播数据
        for file in sorted(os.listdir(forward_dir)):
            if file.startswith('forward_stats_') and file.endswith('.txt'):
                stats_file = os.path.join(forward_dir, file)
                stats = parse_perf_stats(stats_file)
                if stats:
                    stats['phase'] = 'forward'
                    stats['file'] = file
                    all_stats.append(stats)
        
        # 读取后向传播数据
        for file in sorted(os.listdir(backward_dir)):
            if file.startswith('backward_stats_') and file.endswith('.txt'):
                stats_file = os.path.join(backward_dir, file)
                stats = parse_perf_stats(stats_file)
                if stats:
                    stats['phase'] = 'backward'
                    stats['file'] = file
                    all_stats.append(stats)
    
    if not all_stats:
        print("No valid performance data found")
        return
        
    # 计算平均值
    metrics = ['cache_miss_rate', 'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate', 'memory_bandwidth', 'ipc']
    forward_avg = {}
    backward_avg = {}
    
    for metric in metrics:
        forward_values = [s[metric] for s in all_stats if s['phase'] == 'forward' and metric in s]
        backward_values = [s[metric] for s in all_stats if s['phase'] == 'backward' and metric in s]
        
        if forward_values:
            forward_avg[metric] = np.mean(forward_values)
        if backward_values:
            backward_avg[metric] = np.mean(backward_values)
    
    # 计算平均缓存缺失数量
    forward_l1_misses = [s['l1_miss_count'] for s in all_stats if s['phase'] == 'forward' and 'l1_miss_count' in s]
    forward_llc_misses = [s['llc_miss_count'] for s in all_stats if s['phase'] == 'forward' and 'llc_miss_count' in s]
    backward_l1_misses = [s['l1_miss_count'] for s in all_stats if s['phase'] == 'backward' and 'l1_miss_count' in s]
    backward_llc_misses = [s['llc_miss_count'] for s in all_stats if s['phase'] == 'backward' and 'llc_miss_count' in s]
    
    forward_avg['l1_miss_count'] = np.mean(forward_l1_misses) if forward_l1_misses else 0
    forward_avg['llc_miss_count'] = np.mean(forward_llc_misses) if forward_llc_misses else 0
    backward_avg['l1_miss_count'] = np.mean(backward_l1_misses) if backward_l1_misses else 0
    backward_avg['llc_miss_count'] = np.mean(backward_llc_misses) if backward_llc_misses else 0
    
    # 1. 缓存缺失率对比图
    plt.figure(figsize=(10, 6))
    cache_metrics = ['cache_miss_rate', 'l1_miss_rate', 'llc_miss_rate']
    x = np.arange(len(cache_metrics))
    width = 0.35
    
    plt.bar(x - width/2, [forward_avg[m] for m in cache_metrics], width, label='Forward')
    plt.bar(x + width/2, [backward_avg[m] for m in cache_metrics], width, label='Backward')
    
    plt.xlabel('Cache Levels')
    plt.ylabel('Miss Rate (%)')
    plt.title('Cache Miss Rates Comparison')
    plt.xticks(x, ['L1', 'L2', 'LLC'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cache_miss_rates.png')
    plt.close()
    
    # 2. 分支预测失败率对比图
    plt.figure(figsize=(8, 6))
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['branch_miss_rate'], backward_avg['branch_miss_rate']])
    plt.ylabel('Miss Rate (%)')
    plt.title('Branch Prediction Miss Rate')
    plt.tight_layout()
    plt.savefig('branch_miss_rate.png')
    plt.close()
    
    # 3. 内存带宽对比图
    plt.figure(figsize=(8, 6))
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['memory_bandwidth'], backward_avg['memory_bandwidth']])
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth')
    plt.tight_layout()
    plt.savefig('memory_bandwidth.png')
    plt.close()
    
    # 4. IPC对比图
    plt.figure(figsize=(8, 6))
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['ipc'], backward_avg['ipc']])
    plt.ylabel('IPC')
    plt.title('Instructions Per Cycle')
    plt.tight_layout()
    plt.savefig('ipc_comparison.png')
    plt.close()
    
    # 5. 平均缓存缺失数量对比图
    plt.figure(figsize=(8, 6))
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['l1_miss_count'], backward_avg['l1_miss_count']])
    plt.ylabel('Average L1 Miss Count')
    plt.title('Average L1 Cache Miss Counts')
    plt.tight_layout()
    plt.savefig('l1_miss_counts.png')
    plt.close()
    
    # 生成报告
    report = "Performance Analysis Report\n"
    report += "========================\n\n"
    
    report += "Forward Propagation (Average):\n"
    for metric in metrics:
        if metric in forward_avg:
            if metric == 'memory_bandwidth':
                report += f"{metric}: {forward_avg[metric]:.2f} MB/s\n"
            elif metric == 'ipc':
                report += f"{metric}: {forward_avg[metric]:.2f}\n"
            else:
                report += f"{metric}: {forward_avg[metric]:.2f}%\n"
    
    # 添加缓存缺失数量到报告
    report += f"Average L1 Miss Count: {forward_avg['l1_miss_count']:.2f}\n"
    
    report += "\nBackward Propagation (Average):\n"
    for metric in metrics:
        if metric in backward_avg:
            if metric == 'memory_bandwidth':
                report += f"{metric}: {backward_avg[metric]:.2f} MB/s\n"
            elif metric == 'ipc':
                report += f"{metric}: {backward_avg[metric]:.2f}\n"
            else:
                report += f"{metric}: {backward_avg[metric]:.2f}%\n"
    
    # 添加缓存缺失数量到报告
    report += f"Average L1 Miss Count: {backward_avg['l1_miss_count']:.2f}\n"
    
    report += "\nRatios (Backward/Forward):\n"
    for metric in metrics:
        if metric in forward_avg and metric in backward_avg and forward_avg[metric] != 0:
            ratio = backward_avg[metric] / forward_avg[metric]
            report += f"{metric}: {ratio:.2f}x\n"
    
    # 添加缓存缺失数量的比率
    if forward_avg['l1_miss_count'] != 0:
        ratio = backward_avg['l1_miss_count'] / forward_avg['l1_miss_count']
        report += f"L1 Miss Count: {ratio:.2f}x\n"
    
    # 保存报告
    with open('performance_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis completed. Results saved to:")
    print("- cache_miss_rates.png")
    print("- branch_miss_rate.png")
    print("- memory_bandwidth.png")
    print("- ipc_comparison.png")
    print("- l1_miss_counts.png")
    print("- performance_analysis_report.txt")

def main():
    # 指定要分析的目录
    perf_dirs = [
        os.path.join('results', 'perfs', '20250603-163317'),
        os.path.join('results', 'perfs', '20250603-163407')
    ]
    
    # 检查目录是否存在
    valid_dirs = [d for d in perf_dirs if os.path.exists(d)]
    if not valid_dirs:
        print("No valid performance data directories found")
        return
    
    print(f"Analyzing performance data from: {', '.join(valid_dirs)}")
    analyze_performance_data(valid_dirs)

if __name__ == "__main__":
    main() 