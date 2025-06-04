#!/usr/bin/env python3
import os
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def parse_perf_stats(stats_file):
    """解析性能统计文件"""
    try:
        with open(stats_file, 'r') as f:
            content = f.read()
            
        # 检查是否所有事件都是 not counted
        if 'not counted' in content and content.count('not counted') >= 5:
            return None
            
        # 解析浮点运算事件
        fp_scalar_single = re.search(r'(\d+(?:,\d+)*)\s+fp_arith_inst_retired.scalar_single', content)
        fp_scalar_double = re.search(r'(\d+(?:,\d+)*)\s+fp_arith_inst_retired.scalar_double', content)
        fp_vector_single = re.search(r'(\d+(?:,\d+)*)\s+fp_arith_inst_retired.128b_packed_single', content)
        fp_vector_double = re.search(r'(\d+(?:,\d+)*)\s+fp_arith_inst_retired.128b_packed_double', content)
        
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
        
        # 计算浮点运算总数
        total_flops = 0
        if fp_scalar_single and not is_not_counted('fp_arith_inst_retired.scalar_single'):
            total_flops += int(fp_scalar_single.group(1).replace(',', ''))
        if fp_scalar_double and not is_not_counted('fp_arith_inst_retired.scalar_double'):
            total_flops += int(fp_scalar_double.group(1).replace(',', ''))
        if fp_vector_single and not is_not_counted('fp_arith_inst_retired.128b_packed_single'):
            total_flops += int(fp_vector_single.group(1).replace(',', '')) * 4  # 每个向量指令包含4个单精度运算
        if fp_vector_double and not is_not_counted('fp_arith_inst_retired.128b_packed_double'):
            total_flops += int(fp_vector_double.group(1).replace(',', '')) * 2  # 每个向量指令包含2个双精度运算
            
        if total_flops > 0:
            stats['total_flops'] = total_flops
            if time_elapsed:
                time = float(time_elapsed.group(1))
                stats['flops_per_second'] = total_flops / time
        
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
    metrics = ['cache_miss_rate', 'l1_miss_rate', 'llc_miss_rate', 'branch_miss_rate', 
              'memory_bandwidth', 'ipc', 'total_flops', 'flops_per_second']
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
    
    # 创建一个大图，包含6个子图
    plt.figure(figsize=(20, 12))
    
    # 1. 缓存缺失率对比图
    plt.subplot(2, 3, 1)
    cache_metrics = ['cache_miss_rate', 'l1_miss_rate', 'llc_miss_rate']
    x = np.arange(len(cache_metrics))
    width = 0.35
    
    plt.bar(x - width/2, [forward_avg[m] for m in cache_metrics], width, label='Forward')
    plt.bar(x + width/2, [backward_avg[m] for m in cache_metrics], width, label='Backward')
    
    plt.xlabel('Cache Levels')
    plt.ylabel('Miss Rate (%)')
    plt.title('Cache Miss Rates')
    plt.xticks(x, ['L1', 'L2', 'LLC'], rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 2. 分支预测失败率对比图
    plt.subplot(2, 3, 2)
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['branch_miss_rate'], backward_avg['branch_miss_rate']])
    plt.ylabel('Miss Rate (%)')
    plt.title('Branch Prediction Miss Rate')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 3. 内存带宽对比图
    plt.subplot(2, 3, 3)
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['memory_bandwidth'], backward_avg['memory_bandwidth']])
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Memory Bandwidth')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 4. IPC对比图
    plt.subplot(2, 3, 4)
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['ipc'], backward_avg['ipc']])
    plt.ylabel('IPC')
    plt.title('Instructions Per Cycle')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 5. L1 Miss Counts对比图
    plt.subplot(2, 3, 5)
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['l1_miss_count'], backward_avg['l1_miss_count']])
    plt.ylabel('L1 Miss Count')
    plt.title('L1 Cache Miss Counts')
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 6. FLOPS对比图
    plt.subplot(2, 3, 6)
    plt.bar(['Forward', 'Backward'], 
            [forward_avg['flops_per_second'], backward_avg['flops_per_second']])
    plt.ylabel('FLOPS')
    plt.title('Floating Point Operations Per Second')
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'backward_forward_performance_comparison_{timestamp}.png')
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
            elif metric == 'total_flops':
                report += f"{metric}: {forward_avg[metric]:.2e} ({int(forward_avg[metric]):,})\n"
            elif metric == 'flops_per_second':
                report += f"{metric}: {forward_avg[metric]:.2e} ({int(forward_avg[metric]):,})\n"
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
            elif metric == 'total_flops':
                report += f"{metric}: {backward_avg[metric]:.2e} ({int(backward_avg[metric]):,})\n"
            elif metric == 'flops_per_second':
                report += f"{metric}: {backward_avg[metric]:.2e} ({int(backward_avg[metric]):,})\n"
            else:
                report += f"{metric}: {backward_avg[metric]:.2f}%\n"
    
    # 添加缓存缺失数量的比率
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
    report_file = f'backward_forward_performance_analysis_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis completed. Results saved to:")
    print(f"- backward_forward_performance_comparison_{timestamp}.png")
    print(f"- {report_file}")
    print(f"\nSummary:")
    print(f"Forward Propagation: {forward_avg['total_flops']:.2e} ({int(forward_avg['total_flops']):,}) FLOPs (avg)")
    print(f"Backward Propagation: {backward_avg['total_flops']:.2e} ({int(backward_avg['total_flops']):,}) FLOPs (avg)")
    if forward_avg['total_flops'] > 0:
        ratio = backward_avg['total_flops'] / forward_avg['total_flops']
        print(f"Backward/Forward Ratio: {ratio:.2f}x")

def main():
    # 查找所有性能数据目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    perf_base_dir = os.path.join(current_dir, 'results', 'cora', 'perfs')
    
    print(f"Looking for performance data in: {perf_base_dir}")
    
    if not os.path.exists(perf_base_dir):
        print(f"Performance data directory {perf_base_dir} not found")
        return
        
    # 获取所有包含 perf 数据的目录
    perf_dirs = []
    for root, dirs, files in os.walk(perf_base_dir):
        # 检查是否有 forward 和 backward 子目录
        if 'forward' in dirs and 'backward' in dirs:
            forward_dir = os.path.join(root, 'forward')
            backward_dir = os.path.join(root, 'backward')
            
            # 检查子目录中的文件
            forward_files = os.listdir(forward_dir)
            backward_files = os.listdir(backward_dir)
            
            has_forward = any('forward_stats_' in f and f.endswith('.txt') for f in forward_files)
            has_backward = any('backward_stats_' in f and f.endswith('.txt') for f in backward_files)
            
            if has_forward and has_backward:
                perf_dirs.append(root)
                print(f"Found valid performance data in: {root}")
    
    if not perf_dirs:
        print("No valid performance data directories found")
        return
    
    print(f"\nFound {len(perf_dirs)} performance data directories:")
    for d in perf_dirs:
        print(f"- {d}")
    
    analyze_performance_data(perf_dirs)

if __name__ == "__main__":
    main() 