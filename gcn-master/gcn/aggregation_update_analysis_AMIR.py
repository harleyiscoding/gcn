import os
import re
import numpy as np
import json

def parse_llc_loads(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    with open(file_path, 'r') as f:
        content = f.read()
    if 'notcounted' in content or 'not counted' in content:
        return None
    m = re.search(r'(\d+(?:,\d+)*)\s+LLC-loads', content)
    if m:
        return int(m.group(1).replace(',', ''))
    return None

def fill_with_future_mean(arr, start_idx, window=10, global_mean=0):
    vals = []
    for i in range(start_idx+1, len(arr)):
        if arr[i] is not None:
            vals.append(arr[i])
        if len(vals) == window:
            break
    if vals:
        return int(np.mean(vals))
    else:
        return global_mean

def collect_dram_access_epochs(perfs_dir, num_epochs=200):
    layer1_dir = os.path.join(perfs_dir, 'layer1_aggregation')
    layer2_dir = os.path.join(perfs_dir, 'layer2_aggregation')
    layer1_vals = []
    layer2_vals = []
    layer1_agg = [None] * num_epochs
    layer2_agg = [None] * num_epochs
    # 先收集所有可用值
    for epoch in range(num_epochs):
        f1 = os.path.join(layer1_dir, f'layer1_aggregation_stats_{epoch}.txt')
        f2 = os.path.join(layer2_dir, f'layer2_aggregation_stats_{epoch}.txt')
        l1 = parse_llc_loads(f1)
        l2 = parse_llc_loads(f2)
        if l1 is not None:
            layer1_agg[epoch] = l1 * 64
            layer1_vals.append(layer1_agg[epoch])
        if l2 is not None:
            layer2_agg[epoch] = l2 * 64
            layer2_vals.append(layer2_agg[epoch])
    # 计算全局均值
    l1_mean = int(np.mean(layer1_vals)) if layer1_vals else 0
    l2_mean = int(np.mean(layer2_vals)) if layer2_vals else 0
    # 填充缺失，优先用后10个有效值的均值
    for epoch in range(num_epochs):
        if layer1_agg[epoch] is None:
            layer1_agg[epoch] = fill_with_future_mean(layer1_agg, epoch, window=10, global_mean=l1_mean)
        if layer2_agg[epoch] is None:
            layer2_agg[epoch] = fill_with_future_mean(layer2_agg, epoch, window=10, global_mean=l2_mean)
    return layer1_agg, layer2_agg

def parse_flops_json(file_path):
    """Parses a single FLOPs JSON file."""
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
    layer1_update_flops = [0] * num_epochs
    layer2_update_flops = [0] * num_epochs
    prev_flops = None
    for i in range(num_epochs):
        epoch = i + 1  # 文件编号从1开始
        file_path = os.path.join(flops_dir, f'flops_epoch_{epoch}.json')
        current_flops_cumulative = parse_flops_json(file_path)
        if current_flops_cumulative is not None:
            if prev_flops is None:
                layer1_update_flops[i] = current_flops_cumulative['layer1_update']
                layer2_update_flops[i] = current_flops_cumulative['layer2_update']
            else:
                l1_flops = current_flops_cumulative['layer1_update'] - prev_flops['layer1_update']
                l2_flops = current_flops_cumulative['layer2_update'] - prev_flops['layer2_update']
                layer1_update_flops[i] = l1_flops if l1_flops >= 0 else 0
                layer2_update_flops[i] = l2_flops if l2_flops >= 0 else 0
            prev_flops = current_flops_cumulative
        else:
            layer1_update_flops[i] = 0
            layer2_update_flops[i] = 0
    return layer1_update_flops, layer2_update_flops

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = 'pubmed'  # 可根据需要修改
    # --- DRAM Access Collection ---
    perfs_base = os.path.join(current_dir, 'results', dataset, 'perfs')
    if not os.path.exists(perfs_base):
        print(f"No perfs dir: {perfs_base}")
        return
    latest_perfs_ts = max([d for d in os.listdir(perfs_base) if os.path.isdir(os.path.join(perfs_base, d))])
    perfs_dir = os.path.join(perfs_base, latest_perfs_ts)
    layer1_agg_mem, layer2_agg_mem = collect_dram_access_epochs(perfs_dir, num_epochs=200)

    # --- FLOPs Collection ---
    flops_base = os.path.join(current_dir, 'results', dataset, 'tf-flops')
    if not os.path.exists(flops_base):
        print(f"No tf-flops dir for {dataset}")
        layer1_update_flops = [0] * 200
        layer2_update_flops = [0] * 200
    else:
        latest_flops_ts = max([d for d in os.listdir(flops_base) if os.path.isdir(os.path.join(flops_base, d))])
        flops_dir = os.path.join(flops_base, latest_flops_ts)
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

if __name__ == "__main__":
    main() 