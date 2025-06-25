import os
import re
import numpy as np
import json

def parse_llc_loads(file_path):
    """更精确地解析LLC-loads数据"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 更严格的无效数据检测
    if 'notcounted' in content.lower() or 'not counted' in content.lower():
        return None
    
    # 尝试匹配更精确的格式
    patterns = [
        r'(\d+(?:,\d+)*)\s+LLC-loads',  # 标准格式
        r'LLC-loads\s*:\s*(\d+(?:,\d+)*)',  # 可能的不同格式
        r'(\d+)\s+LLC\s+loads'  # 可能的其他格式
    ]
    
    for pattern in patterns:
        m = re.search(pattern, content)
        if m:
            return int(m.group(1).replace(',', ''))
    
    return None

def fill_missing_values(arr, window=5):
    """改进的缺失值填充方法，结合前后数据"""
    arr = np.array(arr, dtype=float)
    mask = ~np.isnan(arr)
    
    # 使用移动平均填充缺失值
    for i in range(len(arr)):
        if not mask[i]:
            start = max(0, i - window)
            end = min(len(arr), i + window + 1)
            neighbors = arr[start:end][mask[start:end]]
            if len(neighbors) > 0:
                arr[i] = np.mean(neighbors)
            else:
                arr[i] = np.nan  # 如果完全没有邻居，保持缺失
    
    # 如果还有缺失值，使用线性插值
    if np.any(np.isnan(arr)):
        valid_idx = np.where(~np.isnan(arr))[0]
        invalid_idx = np.where(np.isnan(arr))[0]
        if len(valid_idx) > 0:
            arr[invalid_idx] = np.interp(invalid_idx, valid_idx, arr[valid_idx])
    
    return arr.astype(int)

def collect_dram_access_epochs(perfs_dir, num_epochs=200):
    """改进的内存带宽收集方法"""
    layer1_dir = os.path.join(perfs_dir, 'layer1_aggregation')
    layer2_dir = os.path.join(perfs_dir, 'layer2_aggregation')
    
    # 收集原始数据
    raw_layer1 = []
    raw_layer2 = []
    
    for epoch in range(num_epochs):
        f1 = os.path.join(layer1_dir, f'layer1_aggregation_stats_{epoch}.txt')
        f2 = os.path.join(layer2_dir, f'layer2_aggregation_stats_{epoch}.txt')
        
        l1 = parse_llc_loads(f1)
        l2 = parse_llc_loads(f2)
        
        raw_layer1.append(l1 * 64 if l1 is not None else np.nan)
        raw_layer2.append(l2 * 64 if l2 is not None else np.nan)
    
    # 改进的填充方法
    layer1_agg = fill_missing_values(raw_layer1)
    layer2_agg = fill_missing_values(raw_layer2)
    
    return layer1_agg.tolist(), layer2_agg.tolist()

def parse_flops_json(file_path):
    """Parses a single FLOPs JSON file from tf-flops directory."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            return {
                'layer1_update': data.get('layer1_update', 0),
                'layer2_update': data.get('layer2_update', 0)
            }
        except json.JSONDecodeError:
            return None

def collect_flops_epochs(flops_dir, num_epochs=200):
    """从tf-flops目录读取FLOPs数据"""
    layer1_update_flops = [0] * num_epochs
    layer2_update_flops = [0] * num_epochs
    
    for i in range(num_epochs):
        epoch = i + 1  # 文件编号从1开始
        file_path = os.path.join(flops_dir, f'flops_epoch_{epoch}.json')
        flops_data = parse_flops_json(file_path)
        if flops_data is not None:
            layer1_update_flops[i] = flops_data['layer1_update']
            layer2_update_flops[i] = flops_data['layer2_update']
        else:
            print(f"Warning: Could not read FLOPs data from {file_path}")
            layer1_update_flops[i] = 0
            layer2_update_flops[i] = 0
    
    return layer1_update_flops, layer2_update_flops

def process_dataset(dataset):
    """处理单个数据集的内存和FLOPs数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n=== Processing {dataset.upper()} ===")
    
    # --- DRAM Access Collection ---
    perfs_base = os.path.join(current_dir, 'results', dataset, 'perfs')
    if not os.path.exists(perfs_base):
        print(f"No perfs dir: {perfs_base}")
        return False
    
    try:
        # 选择最新的时间戳目录
        perfs_dirs = [d for d in os.listdir(perfs_base) if os.path.isdir(os.path.join(perfs_base, d))]
        if not perfs_dirs:
            print(f"No perf data directories found in {perfs_base}")
            return False
            
        latest_perfs_ts = max(perfs_dirs)
        perfs_dir = os.path.join(perfs_base, latest_perfs_ts)
        print(f"Reading DRAM access data from: {perfs_dir}")
        
        # 使用改进的内存收集方法
        layer1_agg_mem, layer2_agg_mem = collect_dram_access_epochs(perfs_dir)
        
        # 检查数据质量
        unique_l1 = len(set(layer1_agg_mem[:10]))  # 检查前10个epoch的唯一值数量
        unique_l2 = len(set(layer2_agg_mem[:10]))
        if unique_l1 < 5 or unique_l2 < 5:
            print(f"Warning: First 10 epochs have limited variability (L1: {unique_l1}, L2: {unique_l2})")
        
        # 打印前几个epoch的数据用于调试
        print(f"First 5 epochs L1 memory: {layer1_agg_mem[:5]}")
        print(f"First 5 epochs L2 memory: {layer2_agg_mem[:5]}")

        # --- FLOPs Collection from tf-flops ---
        flops_base = os.path.join(current_dir, 'results', dataset, 'tf-flops')
        if not os.path.exists(flops_base):
            print(f"No tf-flops dir for {dataset}")
            layer1_update_flops = [0] * 200
            layer2_update_flops = [0] * 200
        else:
            latest_flops_ts = max([d for d in os.listdir(flops_base) if os.path.isdir(os.path.join(flops_base, d))])
            flops_dir = os.path.join(flops_base, latest_flops_ts)
            print(f"Reading FLOPs data from: {flops_dir}")
            layer1_update_flops, layer2_update_flops = collect_flops_epochs(flops_dir, num_epochs=200)

        # --- Output Combined Table ---
        output_dir = os.path.join(current_dir, 'results', dataset, 'l1_cache_analysis')
        os.makedirs(output_dir, exist_ok=True)
        table_path = os.path.join(output_dir, 'memory_flops_epochs.txt')
        with open(table_path, 'w') as f:
            f.write('epoch\tlayer1_agg_mem\tlayer2_agg_mem\tlayer1_update_flops\tlayer2_update_flops\n')
            for i in range(200):
                epoch = i + 1
                f.write(f'{epoch}\t{layer1_agg_mem[i]}\t{layer2_agg_mem[i]}\t{layer1_update_flops[i]}\t{layer2_update_flops[i]}\n')
        print(f"Saved: {table_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {dataset}: {str(e)}")
        return False

def main():
    # 处理所有三个数据集
    datasets = ['cora', 'citeseer', 'pubmed']
    successful_datasets = []
    
    print("Starting memory and FLOPs analysis for all datasets...")
    
    for dataset in datasets:
        if process_dataset(dataset):
            successful_datasets.append(dataset)
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful_datasets}")
    print(f"Failed datasets: {[d for d in datasets if d not in successful_datasets]}")
    print(f"Total processed: {len(successful_datasets)}/{len(datasets)}")

if __name__ == "__main__":
    main() 