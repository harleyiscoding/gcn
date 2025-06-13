import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

def parse_perf_file(file_path):
    """解析perf统计文件"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
        
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # 解析缓存缺失率
        cache_misses = re.search(r'(\d+,\d+|\d+)\s+cache-misses', content)
        cache_refs = re.search(r'(\d+,\d+|\d+)\s+cache-references', content)
        if cache_misses and cache_refs:
            misses = int(cache_misses.group(1).replace(',', ''))
            refs = int(cache_refs.group(1).replace(',', ''))
            metrics['cache_miss_rate'] = misses / refs if refs > 0 else 0
            
        # 解析LLC加载次数
        llc_loads = re.search(r'(\d+,\d+|\d+)\s+LLC-loads', content)
        if llc_loads:
            metrics['llc_loads'] = int(llc_loads.group(1).replace(',', ''))
            
        # 解析浮点运算指令
        fp_metrics = {}
        for fp_type in ['scalar_single', 'scalar_double', '128b_packed_single', '128b_packed_double']:
            pattern = f'(\d+,\d+|\d+)\s+fp_arith_inst_retired.{fp_type}'
            match = re.search(pattern, content)
            if match:
                fp_metrics[fp_type] = int(match.group(1).replace(',', ''))
        metrics['fp_ops'] = sum(fp_metrics.values())
        
    return metrics

def calculate_metrics(data_dir):
    """计算所有epoch的性能指标"""
    metrics = {
        'aggregation': defaultdict(list),
        'update': defaultdict(list)
    }
    
    # 获取所有epoch的perf文件
    agg_files = sorted(glob.glob(os.path.join(data_dir, 'forward_aggregation', '*.txt')))
    update_files = sorted(glob.glob(os.path.join(data_dir, 'forward_update', '*.txt')))
    
    # 解析所有文件
    for phase, files in [('aggregation', agg_files), ('update', update_files)]:
        for file_path in files:
            epoch_metrics = parse_perf_file(file_path)
            if epoch_metrics:
                for metric, value in epoch_metrics.items():
                    metrics[phase][metric].append(value)
    
    # 计算均值用于填充缺失值
    means = {}
    for metric in ['cache_miss_rate', 'llc_loads', 'fp_ops']:
        agg_values = metrics['aggregation'][metric]
        update_values = metrics['update'][metric]
        if agg_values and update_values:
            means[metric] = {
                'aggregation': np.mean(agg_values),
                'update': np.mean(update_values)
            }
    
    # 填充缺失值
    for phase in ['aggregation', 'update']:
        for metric in ['cache_miss_rate', 'llc_loads', 'fp_ops']:
            if not metrics[phase][metric]:
                metrics[phase][metric] = [means[metric][phase]]
    
    return metrics

def plot_metrics(metrics, output_dir):
    """生成性能对比图表"""
    # 确保所有指标的长度一致
    min_length = min(
        len(metrics['aggregation']['cache_miss_rate']),
        len(metrics['update']['cache_miss_rate']),
        len(metrics['aggregation']['llc_loads']),
        len(metrics['update']['llc_loads']),
        len(metrics['aggregation']['fp_ops']),
        len(metrics['update']['fp_ops'])
    )
    
    epochs = range(1, min_length + 1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置全局样式
    plt.style.use('seaborn')
    
    # 1. 缓存缺失率对比折线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['aggregation']['cache_miss_rate'][:min_length], 
             'b-', label='Aggregation', linewidth=1.5, alpha=0.8)
    plt.plot(epochs, metrics['update']['cache_miss_rate'][:min_length], 
             'r-', label='Update', linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Cache Miss Rate', fontsize=10)
    plt.title('Cache Miss Rate Comparison', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.yscale('log')  # 使用对数坐标轴
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cache_miss_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 内存带宽对比折线图
    plt.figure(figsize=(10, 6))
    agg_bandwidth = [x * 64 * 1024 for x in metrics['aggregation']['llc_loads'][:min_length]]  # 转换为字节
    update_bandwidth = [x * 64 * 1024 for x in metrics['update']['llc_loads'][:min_length]]
    plt.plot(epochs, agg_bandwidth, 'b-', label='Aggregation', linewidth=1.5, alpha=0.8)
    plt.plot(epochs, update_bandwidth, 'r-', label='Update', linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Memory Bandwidth (bytes)', fontsize=10)
    plt.title('Memory Bandwidth Comparison', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.yscale('log')  # 使用对数坐标轴
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_bandwidth_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. FLOPS对比折线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['aggregation']['fp_ops'][:min_length], 
             'b-', label='Aggregation', linewidth=1.5, alpha=0.8)
    plt.plot(epochs, metrics['update']['fp_ops'][:min_length], 
             'r-', label='Update', linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('FLOPS', fontsize=10)
    plt.title('FLOPS Comparison', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.yscale('log')  # 使用对数坐标轴
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flops_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4-6. 均值对比柱状图
    mean_metrics = {
        'Cache Miss Rate': {
            'Aggregation': np.mean(metrics['aggregation']['cache_miss_rate'][:min_length]),
            'Update': np.mean(metrics['update']['cache_miss_rate'][:min_length])
        },
        'Memory Bandwidth': {
            'Aggregation': np.mean(agg_bandwidth),
            'Update': np.mean(update_bandwidth)
        },
        'FLOPS': {
            'Aggregation': np.mean(metrics['aggregation']['fp_ops'][:min_length]),
            'Update': np.mean(metrics['update']['fp_ops'][:min_length])
        }
    }
    
    for metric_name, values in mean_metrics.items():
        plt.figure(figsize=(8, 6))
        phases = list(values.keys())
        means = list(values.values())
        
        # 创建柱状图
        bars = plt.bar(phases, means, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        
        plt.ylabel(metric_name, fontsize=10)
        plt.title(f'Average {metric_name} Comparison', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.yscale('log')  # 使用对数坐标轴
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'average_{metric_name.lower().replace(" ", "_")}_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成详细的分析报告
    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write("Performance Analysis Report\n")
        f.write("=========================\n\n")
        
        f.write("Theoretical Expectations:\n")
        f.write("------------------------\n")
        f.write("1. Cache Miss Rate: Aggregation should be higher (memory-bound)\n")
        f.write("2. Memory Bandwidth: Aggregation should be higher (memory-bound)\n")
        f.write("3. FLOPS: Update should be higher (compute-bound)\n\n")
        
        f.write("Actual Results:\n")
        f.write("--------------\n")
        for metric_name, values in mean_metrics.items():
            f.write(f"{metric_name}:\n")
            f.write(f"  Aggregation: {values['Aggregation']:.2e}\n")
            f.write(f"  Update: {values['Update']:.2e}\n")
            ratio = values['Update']/values['Aggregation']
            f.write(f"  Ratio (Update/Aggregation): {ratio:.2f}\n")
            
            # 添加理论验证
            if metric_name in ['Cache Miss Rate', 'Memory Bandwidth']:
                if ratio < 1:
                    f.write("  ✓ Matches theory: Aggregation has higher {}\n".format(metric_name.lower()))
                else:
                    f.write("  ✗ Contradicts theory: Update has higher {}\n".format(metric_name.lower()))
            elif metric_name == 'FLOPS':
                if ratio > 1:
                    f.write("  ✓ Matches theory: Update has higher FLOPS\n")
                else:
                    f.write("  ✗ Contradicts theory: Aggregation has higher FLOPS\n")
            f.write("\n")
        
        # 添加总体分析
        f.write("Overall Analysis:\n")
        f.write("---------------\n")
        matches = 0
        for metric_name, values in mean_metrics.items():
            ratio = values['Update']/values['Aggregation']
            if (metric_name in ['Cache Miss Rate', 'Memory Bandwidth'] and ratio < 1) or \
               (metric_name == 'FLOPS' and ratio > 1):
                matches += 1
        
        f.write(f"Matches theoretical expectations: {matches}/3 metrics\n")
        if matches == 3:
            f.write("✓ All metrics match theoretical expectations!\n")
        else:
            f.write("! Some metrics contradict theoretical expectations.\n")
            f.write("  This might indicate:\n")
            f.write("  1. Implementation differences\n")
            f.write("  2. Hardware characteristics\n")
            f.write("  3. Data access patterns\n")
            f.write("  4. Compiler optimizations\n")

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理每个数据集
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets:
        # 构建数据目录路径
        data_dir = os.path.join(current_dir, 'results', dataset, 'perfs')
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory not found: {data_dir}")
            continue
            
        # 获取最新的时间戳目录
        timestamps = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not timestamps:
            print(f"Warning: No timestamp directories found in {data_dir}")
            continue
            
        latest_timestamp = max(timestamps)
        perf_dir = os.path.join(data_dir, latest_timestamp)
        print(f"Processing performance data from: {perf_dir}")
        
        # 计算性能指标
        metrics = calculate_metrics(perf_dir)
        
        # 创建输出目录
        output_dir = os.path.join(current_dir, 'results', dataset, 'aggregation_update_compare')
        
        # 生成图表和报告
        plot_metrics(metrics, output_dir)
        
        print(f"Analysis completed for {dataset}. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 