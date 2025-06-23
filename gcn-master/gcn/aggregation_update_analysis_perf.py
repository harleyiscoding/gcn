import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

def parse_perf_file(file_path):
    """解析perf统计文件，提取L1缓存数据"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
        
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # 打印文件内容的前几行用于调试
        print(f"\nDebug: First few lines of {file_path}:")
        print(content.split('\n')[:5])
        
        # 解析L1缓存数据 - 更新正则表达式以匹配实际格式
        l1_misses = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-load-misses', content)
        l1_loads = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-loads', content)
        
        # 打印匹配结果用于调试
        print(f"\nDebug: L1 cache matches in {file_path}:")
        print(f"L1 Misses match: {l1_misses.group(1) if l1_misses else 'None'}")
        print(f"L1 Loads match: {l1_loads.group(1) if l1_loads else 'None'}")
        
        # 如果数据为notcounted则忽略
        if 'notcounted' not in content:
            # 处理L1缓存数据
            if l1_misses and l1_loads:
                misses = int(l1_misses.group(1).replace(',', ''))
                loads = int(l1_loads.group(1).replace(',', ''))
                
                if misses > loads:
                    print(f"Warning: Invalid cache miss rate detected in {file_path}")
                    print(f"L1 Misses ({misses:,}) > L1 Loads ({loads:,})")
                    metrics['cache_miss_rate'] = min(1.0, loads / misses if misses > 0 else 0)
                else:
                    metrics['cache_miss_rate'] = misses / loads if loads > 0 else 0
                    
                metrics['l1_misses'] = misses
                metrics['l1_loads'] = loads
                print(f"Debug: L1 Misses={misses:,}, L1 Loads={loads:,}, L1 Cache Miss Rate={metrics['cache_miss_rate']:.4f}")
        
    return metrics

def calculate_metrics(data_dir):
    """计算所有epoch的性能指标，忽略前10个epoch"""
    metrics = {
        'layer1_update': defaultdict(list),
        'layer1_aggregation': defaultdict(list),
        'layer2_update': defaultdict(list),
        'layer2_aggregation': defaultdict(list)
    }
    
    # 获取所有epoch的perf文件
    phase_files = {
        'layer1_update': sorted(glob.glob(os.path.join(data_dir, 'layer1_update', '*.txt'))),
        'layer1_aggregation': sorted(glob.glob(os.path.join(data_dir, 'layer1_aggregation', '*.txt'))),
        'layer2_update': sorted(glob.glob(os.path.join(data_dir, 'layer2_update', '*.txt'))),
        'layer2_aggregation': sorted(glob.glob(os.path.join(data_dir, 'layer2_aggregation', '*.txt')))
    }
    
    # 打印每个阶段的文件数量
    print("\nFile counts for each phase:")
    for phase, files in phase_files.items():
        print(f"{phase}: {len(files)} files")
    
    # 解析所有文件，跳过前10个epoch
    for layer in ['layer1', 'layer2']:
        prev_agg_metrics = None
        prev_update_metrics = None
        
        # 获取当前层的文件
        agg_files = phase_files[f'{layer}_aggregation'][10:]  # 跳过前10个epoch
        update_files = phase_files[f'{layer}_update'][10:]
        
        print(f"\nProcessing {layer}:")
        print(f"Number of aggregation files after epoch 10: {len(agg_files)}")
        print(f"Number of update files after epoch 10: {len(update_files)}")
        
        # 确保文件数量相同
        min_files = min(len(agg_files), len(update_files))
        print(f"Minimum number of files to process: {min_files}")
        
        valid_epochs = 0
        for i in range(min_files):
            # 解析聚合阶段
            agg_metrics = parse_perf_file(agg_files[i])
            if agg_metrics:
                # 计算聚合阶段的实际值
                if prev_update_metrics is not None:
                    # L1缓存数据
                    actual_agg_misses = agg_metrics['l1_misses'] - prev_update_metrics['l1_misses']
                    actual_agg_loads = agg_metrics['l1_loads'] - prev_update_metrics['l1_loads']
                    
                    # 确保差值非负且合理
                    if actual_agg_misses >= 0 and actual_agg_loads >= 0 and actual_agg_misses <= actual_agg_loads:
                        metrics[f'{layer}_aggregation']['l1_misses'].append(actual_agg_misses)
                        metrics[f'{layer}_aggregation']['l1_loads'].append(actual_agg_loads)
                        metrics[f'{layer}_aggregation']['cache_miss_rate'].append(
                            actual_agg_misses / actual_agg_loads if actual_agg_loads > 0 else 0
                        )
                        valid_epochs += 1
                    else:
                        print(f"Warning: Invalid difference in {layer} aggregation at epoch {i+11}")
                        print(f"  Current misses: {agg_metrics['l1_misses']}, Previous update misses: {prev_update_metrics['l1_misses']}")
                        print(f"  Current loads: {agg_metrics['l1_loads']}, Previous update loads: {prev_update_metrics['l1_loads']}")
                
                prev_agg_metrics = agg_metrics
            else:
                print(f"Warning: Could not parse aggregation file for {layer} at epoch {i+11}")
            
            # 解析更新阶段
            update_metrics = parse_perf_file(update_files[i])
            if update_metrics:
                # 计算更新阶段的实际值
                if prev_agg_metrics is not None:
                    # L1缓存数据
                    actual_update_misses = update_metrics['l1_misses'] - prev_agg_metrics['l1_misses']
                    actual_update_loads = update_metrics['l1_loads'] - prev_agg_metrics['l1_loads']
                    
                    # 确保差值非负且合理
                    if actual_update_misses >= 0 and actual_update_loads >= 0 and actual_update_misses <= actual_update_loads:
                        metrics[f'{layer}_update']['l1_misses'].append(actual_update_misses)
                        metrics[f'{layer}_update']['l1_loads'].append(actual_update_loads)
                        metrics[f'{layer}_update']['cache_miss_rate'].append(
                            actual_update_misses / actual_update_loads if actual_update_loads > 0 else 0
                        )
                    else:
                        print(f"Warning: Invalid difference in {layer} update at epoch {i+11}")
                        print(f"  Current misses: {update_metrics['l1_misses']}, Previous agg misses: {prev_agg_metrics['l1_misses']}")
                        print(f"  Current loads: {update_metrics['l1_loads']}, Previous agg loads: {prev_agg_metrics['l1_loads']}")
                
                prev_update_metrics = update_metrics
            else:
                print(f"Warning: Could not parse update file for {layer} at epoch {i+11}")
        
        print(f"Valid epochs processed for {layer}: {valid_epochs}")
    
    # 打印每个阶段的最终数据点数量
    print("\nFinal data points for each phase:")
    for phase in metrics:
        print(f"{phase}: {len(metrics[phase]['l1_misses'])} data points")
    
    return metrics

def plot_metrics(metrics, output_dir):
    """生成性能对比图表"""
    # 确保所有指标的长度一致
    min_length = min(
        len(metrics['layer1_update']['cache_miss_rate']),
        len(metrics['layer1_aggregation']['cache_miss_rate']),
        len(metrics['layer2_update']['cache_miss_rate']),
        len(metrics['layer2_aggregation']['cache_miss_rate'])
    )
    
    if min_length == 0:
        print("Warning: No valid data found for plotting")
        return
        
    epochs = range(11, 11 + min_length)  # 从第11个epoch开始
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置全局样式
    plt.style.use('seaborn')
    
    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Layer 1 Update', 'Layer 1 Aggregation', 'Layer 2 Update', 'Layer 2 Aggregation']
    
    # 1. 缓存缺失率对比折线图
    plt.figure(figsize=(10, 6))
    for i, phase in enumerate(['layer1_update', 'layer1_aggregation', 'layer2_update', 'layer2_aggregation']):
        plt.plot(epochs, metrics[phase]['cache_miss_rate'][:min_length], 
                color=colors[i], label=labels[i], linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('L1 Cache Miss Rate', fontsize=10)
    plt.title('L1 Cache Miss Rate Comparison (After Epoch 10)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_miss_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. L1缓存访问次数对比折线图
    plt.figure(figsize=(10, 6))
    for i, phase in enumerate(['layer1_update', 'layer1_aggregation', 'layer2_update', 'layer2_aggregation']):
        plt.plot(epochs, metrics[phase]['l1_loads'][:min_length], 
                color=colors[i], label=labels[i], linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('L1 Cache Loads per Epoch', fontsize=10)
    plt.title('L1 Cache Loads Comparison (After Epoch 10)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_loads_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. L1缓存缺失次数对比折线图
    plt.figure(figsize=(10, 6))
    for i, phase in enumerate(['layer1_update', 'layer1_aggregation', 'layer2_update', 'layer2_aggregation']):
        plt.plot(epochs, metrics[phase]['l1_misses'][:min_length], 
                color=colors[i], label=labels[i], linewidth=1.5, alpha=0.8)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('L1 Cache Misses per Epoch', fontsize=10)
    plt.title('L1 Cache Misses Comparison (After Epoch 10)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_misses_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 添加柱状图对比
    # 4.1 L1缓存缺失率柱状图对比
    plt.figure(figsize=(8, 6))
    # 计算平均值
    avg_metrics = {
        'layer1_update': np.mean(metrics['layer1_update']['cache_miss_rate'][:min_length]),
        'layer1_aggregation': np.mean(metrics['layer1_aggregation']['cache_miss_rate'][:min_length]),
        'layer2_update': np.mean(metrics['layer2_update']['cache_miss_rate'][:min_length]),
        'layer2_aggregation': np.mean(metrics['layer2_aggregation']['cache_miss_rate'][:min_length])
    }
    categories = ['Layer 1', 'Layer 2']
    update_data = [avg_metrics['layer1_update'], avg_metrics['layer2_update']]
    aggregation_data = [avg_metrics['layer1_aggregation'], avg_metrics['layer2_aggregation']]
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, update_data, width, label='Update', color='#1f77b4')
    plt.bar(x + width/2, aggregation_data, width, label='Aggregation', color='#ff7f0e')
    # 不显示数值标签
    plt.xlabel('Layer', fontsize=10)
    plt.ylabel('Average L1 Cache Miss Rate', fontsize=10)
    plt.title('L1 Cache Miss Rate: Update vs Aggregation', fontsize=12)
    plt.xticks(x, categories)
    plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_miss_rate_bar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 4.2 L1缓存访问次数柱状图对比
    plt.figure(figsize=(8, 6))
    avg_metrics = {
        'layer1_update': np.mean(metrics['layer1_update']['l1_loads'][:min_length]),
        'layer1_aggregation': np.mean(metrics['layer1_aggregation']['l1_loads'][:min_length]),
        'layer2_update': np.mean(metrics['layer2_update']['l1_loads'][:min_length]),
        'layer2_aggregation': np.mean(metrics['layer2_aggregation']['l1_loads'][:min_length])
    }
    update_data = [avg_metrics['layer1_update'], avg_metrics['layer2_update']]
    aggregation_data = [avg_metrics['layer1_aggregation'], avg_metrics['layer2_aggregation']]
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, update_data, width, label='Update', color='#1f77b4')
    plt.bar(x + width/2, aggregation_data, width, label='Aggregation', color='#ff7f0e')
    # 不显示数值标签
    plt.xlabel('Layer', fontsize=10)
    plt.ylabel('Average L1 Cache Loads per Epoch', fontsize=10)
    plt.title('L1 Cache Loads: Update vs Aggregation', fontsize=12)
    plt.xticks(x, categories)
    plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_loads_bar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4.3 L1缓存缺失次数柱状图对比
        plt.figure(figsize=(8, 6))
    avg_metrics = {
        'layer1_update': np.mean(metrics['layer1_update']['l1_misses'][:min_length]),
        'layer1_aggregation': np.mean(metrics['layer1_aggregation']['l1_misses'][:min_length]),
        'layer2_update': np.mean(metrics['layer2_update']['l1_misses'][:min_length]),
        'layer2_aggregation': np.mean(metrics['layer2_aggregation']['l1_misses'][:min_length])
    }
    update_data = [avg_metrics['layer1_update'], avg_metrics['layer2_update']]
    aggregation_data = [avg_metrics['layer1_aggregation'], avg_metrics['layer2_aggregation']]
        x = np.arange(len(categories))
        width = 0.35
    plt.bar(x - width/2, update_data, width, label='Update', color='#1f77b4')
    plt.bar(x + width/2, aggregation_data, width, label='Aggregation', color='#ff7f0e')
    # 不显示数值标签
    plt.xlabel('Layer', fontsize=10)
    plt.ylabel('Average L1 Cache Misses per Epoch', fontsize=10)
    plt.title('L1 Cache Misses: Update vs Aggregation', fontsize=12)
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l1_cache_misses_bar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成详细的分析报告
    with open(os.path.join(output_dir, 'performance_analysis_report.txt'), 'w') as f:
        f.write("Performance Analysis Report (After Epoch 10)\n")
        f.write("=====================================\n\n")
        
        # 计算每个阶段的平均值
        for phase in ['layer1_update', 'layer1_aggregation', 'layer2_update', 'layer2_aggregation']:
            f.write(f"{labels[['layer1_update', 'layer1_aggregation', 'layer2_update', 'layer2_aggregation'].index(phase)]}:\n")
            f.write(f"  Average L1 Cache Miss Rate: {np.mean(metrics[phase]['cache_miss_rate'][:min_length]):.4f}\n")
            f.write(f"  Average L1 Cache Loads per Epoch: {np.mean(metrics[phase]['l1_loads'][:min_length]):.2e}\n")
            f.write(f"  Average L1 Cache Misses per Epoch: {np.mean(metrics[phase]['l1_misses'][:min_length]):.2e}\n")
            f.write("\n")
        
        # 计算总聚合和更新的对比
        f.write("Total Aggregation vs Update Comparison:\n")
        f.write("------------------------------------\n")
        
        # 计算总聚合和更新的平均值
        agg_metrics = {
            'miss_rate': np.mean([
                np.mean(metrics['layer1_aggregation']['cache_miss_rate'][:min_length]),
                np.mean(metrics['layer2_aggregation']['cache_miss_rate'][:min_length])
            ]),
            'loads': np.mean([
                np.mean(metrics['layer1_aggregation']['l1_loads'][:min_length]),
                np.mean(metrics['layer2_aggregation']['l1_loads'][:min_length])
            ]),
            'misses': np.mean([
                np.mean(metrics['layer1_aggregation']['l1_misses'][:min_length]),
                np.mean(metrics['layer2_aggregation']['l1_misses'][:min_length])
            ])
        }
        
        update_metrics = {
            'miss_rate': np.mean([
                np.mean(metrics['layer1_update']['cache_miss_rate'][:min_length]),
                np.mean(metrics['layer2_update']['cache_miss_rate'][:min_length])
            ]),
            'loads': np.mean([
                np.mean(metrics['layer1_update']['l1_loads'][:min_length]),
                np.mean(metrics['layer2_update']['l1_loads'][:min_length])
            ]),
            'misses': np.mean([
                np.mean(metrics['layer1_update']['l1_misses'][:min_length]),
                np.mean(metrics['layer2_update']['l1_misses'][:min_length])
            ])
        }
        
        f.write("Aggregation Phase:\n")
        f.write(f"  Average L1 Cache Miss Rate: {agg_metrics['miss_rate']:.4f}\n")
        f.write(f"  Average L1 Cache Loads per Epoch: {agg_metrics['loads']:.2e}\n")
        f.write(f"  Average L1 Cache Misses per Epoch: {agg_metrics['misses']:.2e}\n")
        f.write("\n")
        
        f.write("Update Phase:\n")
        f.write(f"  Average L1 Cache Miss Rate: {update_metrics['miss_rate']:.4f}\n")
        f.write(f"  Average L1 Cache Loads per Epoch: {update_metrics['loads']:.2e}\n")
        f.write(f"  Average L1 Cache Misses per Epoch: {update_metrics['misses']:.2e}\n")
        f.write("\n")
        
        # 计算比率
        f.write("Ratios (Update/Aggregation):\n")
        f.write(f"  L1 Cache Miss Rate: {update_metrics['miss_rate']/agg_metrics['miss_rate']:.2f}\n")
        f.write(f"  L1 Cache Loads: {update_metrics['loads']/agg_metrics['loads']:.2f}\n")
        f.write(f"  L1 Cache Misses: {update_metrics['misses']/agg_metrics['misses']:.2f}\n")
        
        # 添加每层的对比
        f.write("\nLayer-wise Comparison:\n")
        f.write("--------------------\n")
        
        # Layer 1
        f.write("Layer 1:\n")
        layer1_agg = {
            'miss_rate': np.mean(metrics['layer1_aggregation']['cache_miss_rate'][:min_length]),
            'loads': np.mean(metrics['layer1_aggregation']['l1_loads'][:min_length]),
            'misses': np.mean(metrics['layer1_aggregation']['l1_misses'][:min_length])
        }
        
        layer1_update = {
            'miss_rate': np.mean(metrics['layer1_update']['cache_miss_rate'][:min_length]),
            'loads': np.mean(metrics['layer1_update']['l1_loads'][:min_length]),
            'misses': np.mean(metrics['layer1_update']['l1_misses'][:min_length])
        }
        
        f.write(f"  Aggregation vs Update:\n")
        f.write(f"    Miss Rate Ratio: {layer1_update['miss_rate']/layer1_agg['miss_rate']:.2f}\n")
        f.write(f"    Loads Ratio: {layer1_update['loads']/layer1_agg['loads']:.2f}\n")
        f.write(f"    Misses Ratio: {layer1_update['misses']/layer1_agg['misses']:.2f}\n")
        f.write("\n")
        
        # Layer 2
        f.write("Layer 2:\n")
        layer2_agg = {
            'miss_rate': np.mean(metrics['layer2_aggregation']['cache_miss_rate'][:min_length]),
            'loads': np.mean(metrics['layer2_aggregation']['l1_loads'][:min_length]),
            'misses': np.mean(metrics['layer2_aggregation']['l1_misses'][:min_length])
        }
        
        layer2_update = {
            'miss_rate': np.mean(metrics['layer2_update']['cache_miss_rate'][:min_length]),
            'loads': np.mean(metrics['layer2_update']['l1_loads'][:min_length]),
            'misses': np.mean(metrics['layer2_update']['l1_misses'][:min_length])
        }
        
        f.write(f"  Aggregation vs Update:\n")
        f.write(f"    Miss Rate Ratio: {layer2_update['miss_rate']/layer2_agg['miss_rate']:.2f}\n")
        f.write(f"    Loads Ratio: {layer2_update['loads']/layer2_agg['loads']:.2f}\n")
        f.write(f"    Misses Ratio: {layer2_update['misses']/layer2_agg['misses']:.2f}\n")

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
        output_dir = os.path.join(current_dir, 'results', dataset, 'l1_cache_analysis')
        
        # 生成图表和报告
        plot_metrics(metrics, output_dir)
        
        print(f"Analysis completed for {dataset}. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 