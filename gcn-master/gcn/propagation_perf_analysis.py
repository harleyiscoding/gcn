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

def collect_propagation_stats(pid, phase_name, output_file):
    """收集前向或后向传播阶段的性能统计信息"""
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
            
        # 计算并添加命中率信息
        hit_rates = calculate_hit_rates(output_file)
        if hit_rates:
            with open(output_file, 'a') as f:
                f.write("\nHit Rates:\n")
                for name, rate in hit_rates.items():
                    f.write(f"{name}: {rate:.2f}%\n")
                    
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error collecting stats for {phase_name}: {e}")
        return False

def is_process_running(pid):
    """检查进程是否在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def monitor_propagation_phases(pid, forward_dir, backward_dir):
    """监控并收集前向和后向传播阶段的性能统计"""
    events = get_memory_events()
    signal_file = '/tmp/perf_stage_signal'
    
    # 确保信号文件存在
    if not os.path.exists(signal_file):
        os.mknod(signal_file)
    
    current_phase = None
    current_epoch = 0
    forward_stats = []
    backward_stats = []
    
    print("\nStarting to monitor propagation phases...")
    while is_process_running(pid):
        try:
            # 读取信号文件
            with open(signal_file, 'r') as f:
                signal = f.read().strip()
            
            if signal.startswith('PERF_STAGE_START'):
                phase = signal.split()[1]
                if phase == 'forward':
                    current_phase = 'forward'
                    print(f"\nEpoch {current_epoch}: Starting forward propagation")
                    # 开始收集前向传播统计
                    forward_file = os.path.join(forward_dir, f'forward_stats_epoch_{current_epoch}.txt')
                    perf_cmd = [
                        'perf', 'stat',
                        '-e', ','.join(events),
                        '-p', str(pid),
                        '-o', forward_file
                    ]
                    perf_process = subprocess.Popen(perf_cmd)
                    forward_stats.append(perf_process)
                    
                elif phase == 'backward':
                    current_phase = 'backward'
                    print(f"\nEpoch {current_epoch}: Starting backward propagation")
                    # 开始收集后向传播统计
                    backward_file = os.path.join(backward_dir, f'backward_stats_epoch_{current_epoch}.txt')
                    perf_cmd = [
                        'perf', 'stat',
                        '-e', ','.join(events),
                        '-p', str(pid),
                        '-o', backward_file
                    ]
                    perf_process = subprocess.Popen(perf_cmd)
                    backward_stats.append(perf_process)
                    
            elif signal.startswith('PERF_STAGE_END'):
                phase = signal.split()[1]
                if phase == 'forward' and current_phase == 'forward':
                    # 停止收集前向传播统计
                    if forward_stats:
                        perf_process = forward_stats.pop()
                        perf_process.terminate()
                        perf_process.wait()
                        print(f"Epoch {current_epoch}: Completed forward propagation stats collection")
                        
                elif phase == 'backward' and current_phase == 'backward':
                    # 停止收集后向传播统计
                    if backward_stats:
                        perf_process = backward_stats.pop()
                        perf_process.terminate()
                        perf_process.wait()
                        print(f"Epoch {current_epoch}: Completed backward propagation stats collection")
                        current_epoch += 1
                
                current_phase = None
                
        except Exception as e:
            print(f"Error monitoring propagation phases: {e}")
            
        time.sleep(0.1)  # 短暂休眠以减少CPU使用
        
    print(f"\nCollection completed:")
    print(f"- Collected forward propagation stats for epochs: {list(range(current_epoch))}")
    print(f"- Collected backward propagation stats for epochs: {list(range(current_epoch))}")

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    perf_dir = os.path.join('results', 'perfs', timestamp)
    os.makedirs(perf_dir, exist_ok=True)
    
    # 创建前向和后向传播的统计目录
    forward_dir = os.path.join(perf_dir, 'forward')
    backward_dir = os.path.join(perf_dir, 'backward')
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(backward_dir, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"- Forward propagation stats: {forward_dir}")
    print(f"- Backward propagation stats: {backward_dir}")
    
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
            
        # 监控并收集前向和后向传播阶段的性能统计
        monitor_propagation_phases(pid, forward_dir, backward_dir)
            
    except Exception as e:
        print(f"Error during propagation stats collection: {e}")
    finally:
        # 确保训练进程被正确终止
        if is_process_running(pid):
            train_process.terminate()
            train_process.wait()

if __name__ == "__main__":
    main() 