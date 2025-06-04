#!/usr/bin/env python3
import os
import subprocess
import json
import time
from datetime import datetime
import re
import signal

def get_memory_events():
    """返回性能事件列表"""
    return [
        # 基本性能事件
        'branch-misses',
        'cache-misses',
        'cache-references',
        'cpu-cycles',
        'instructions',
        'alignment-faults',
        'context-switches',
        'cpu-clock',
        'cpu-migrations',
        'dummy',
        'emulation-faults',
        'major-faults',
        'minor-faults',
        'page-faults',
        'task-clock',
        
        # 缓存相关事件
        'L1-dcache-load-misses',
        'L1-dcache-loads',
        'L1-dcache-stores',
        'L1-icache-load-misses',
        'LLC-loads',          # 最后一级缓存加载
        'LLC-load-misses',    # 最后一级缓存加载未命中
        
        # 分支预测相关事件
        'branch-load-misses',
        'branch-loads',
        
        # TLB相关事件
        'dTLB-load-misses',
        'dTLB-loads',
        'iTLB-loads',
        'iTLB-load-misses',
        
        # 内存访问事件 (用于估算带宽)
        'cpu/mem-loads/',
        'cpu/mem-stores/'
    ]

def calculate_hit_rates(stats_file):
    """计算缓存命中率"""
    try:
        with open(stats_file, 'r') as f:
            content = f.read()
            
        # 解析缓存引用和未命中
        cache_refs = re.search(r'(\d+)\s+cache-references', content)
        cache_misses = re.search(r'(\d+)\s+cache-misses', content)
        
        # 解析L1数据缓存
        l1_loads = re.search(r'(\d+)\s+L1-dcache-loads', content)
        l1_misses = re.search(r'(\d+)\s+L1-dcache-load-misses', content)
        
        # 解析LLC缓存
        llc_loads = re.search(r'(\d+)\s+LLC-loads', content)
        llc_misses = re.search(r'(\d+)\s+LLC-load-misses', content)
        
        # 解析TLB
        dtlb_loads = re.search(r'(\d+)\s+dTLB-loads', content)
        dtlb_misses = re.search(r'(\d+)\s+dTLB-load-misses', content)
        
        hit_rates = {}
        
        # 计算各级缓存命中率
        if cache_refs and cache_misses:
            refs = int(cache_refs.group(1))
            misses = int(cache_misses.group(1))
            hit_rates['cache_hit_rate'] = (refs - misses) / refs * 100 if refs > 0 else 0
            
        if l1_loads and l1_misses:
            loads = int(l1_loads.group(1))
            misses = int(l1_misses.group(1))
            hit_rates['l1_dcache_hit_rate'] = (loads - misses) / loads * 100 if loads > 0 else 0
            
        if llc_loads and llc_misses:
            loads = int(llc_loads.group(1))
            misses = int(llc_misses.group(1))
            hit_rates['llc_hit_rate'] = (loads - misses) / loads * 100 if loads > 0 else 0
            
        if dtlb_loads and dtlb_misses:
            loads = int(dtlb_loads.group(1))
            misses = int(dtlb_misses.group(1))
            hit_rates['dtlb_hit_rate'] = (loads - misses) / loads * 100 if loads > 0 else 0
            
        return hit_rates
    except Exception as e:
        print(f"Error calculating hit rates: {e}")
        return {}

def collect_stage_memory_stats(pid, stage_name, output_file):
    """收集特定阶段的访存统计信息"""
    events = get_memory_events()
    perf_cmd = [
        'perf', 'stat',
        '-e', ','.join(events),
        '-p', str(pid),
        '--', 'sleep', '1'
    ]
    
    try:
        with open(output_file, 'w') as f:
            subprocess.run(perf_cmd, stdout=f, stderr=f, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error collecting memory stats for {stage_name}: {e}")
        return False

def collect_epoch_memory_stats(pid, epoch_num, output_file):
    """收集特定epoch的访存统计信息"""
    if not is_process_running(pid):
        print(f"Process {pid} is not running when trying to collect stats for epoch {epoch_num}")
        return False
        
    events = get_memory_events()
    perf_cmd = [
        'perf', 'stat',
        '-e', ','.join(events),
        '-p', str(pid),
        '--', 'sleep', '1'
    ]
    
    print(f"Running perf stat for epoch {epoch_num}")
    
    try:
        with open(output_file, 'w') as f:
            result = subprocess.run(perf_cmd, stdout=f, stderr=f, check=True)
            print(f"Successfully collected stats for epoch {epoch_num}")
            
            # 计算并添加命中率信息
            hit_rates = calculate_hit_rates(output_file)
            if hit_rates:
                with open(output_file, 'a') as f:
                    f.write("\nHit Rates:\n")
                    for name, rate in hit_rates.items():
                        f.write(f"{name}: {rate:.2f}%\n")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error collecting memory stats for epoch {epoch_num}: {e}")
        return False

def is_process_running(pid):
    """检查进程是否在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    # 从train.py中获取数据集名称
    try:
        import train
        dataset = train.FLAGS.dataset
    except:
        dataset = 'cora'  # 默认值
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    perf_dir = os.path.join('results', dataset, 'perfs', timestamp)
    os.makedirs(perf_dir, exist_ok=True)
    
    print(f"Created output directory: {perf_dir}")
    
    # 启动训练进程
    train_process = subprocess.Popen(['python', 'train.py'])
    pid = train_process.pid
    print(f"Training process started with PID: {pid}")
    
    try:
        # 等待训练进程启动
        time.sleep(5)
        
        if not is_process_running(pid):
            print("Training process failed to start")
            return
            
        # 收集各个阶段的访存统计信息
        stages = ['model_building', 'preprocessing', 'training', 'validation', 'testing']
        for stage in stages:
            if not is_process_running(pid):
                print(f"Training process ended before collecting {stage} stats")
                break
            stats_file = os.path.join(perf_dir, f'{stage}_memory_stats.txt')
            if collect_stage_memory_stats(pid, stage, stats_file):
                print(f"Successfully collected stats for {stage}")
                # 计算并添加命中率信息
                hit_rates = calculate_hit_rates(stats_file)
                if hit_rates:
                    with open(stats_file, 'a') as f:
                        f.write("\nHit Rates:\n")
                        for name, rate in hit_rates.items():
                            f.write(f"{name}: {rate:.2f}%\n")
            
        # 只收集前10个epoch的访存统计信息
        for epoch in range(10):
            if not is_process_running(pid):
                print(f"Training process ended at epoch {epoch}")
                break
                
            print(f"\nCollecting stats for epoch {epoch}")
            
            # 使用perf stat收集当前epoch的统计信息
            stats_file = os.path.join(perf_dir, f'epoch_{epoch}_memory_stats.txt')
            if collect_epoch_memory_stats(pid, epoch, stats_file):
                print(f"Successfully collected stats for epoch {epoch}")
            else:
                # 如果收集失败，检查进程是否还在运行
                if not is_process_running(pid):
                    print(f"Training process ended at epoch {epoch}")
                    break
            
        print("Memory stats collection completed successfully!")
        
    except Exception as e:
        print(f"Error during memory stats collection: {e}")
    finally:
        # 确保训练进程被正确终止
        if is_process_running(pid):
            train_process.terminate()
            train_process.wait()

if __name__ == "__main__":
    main()