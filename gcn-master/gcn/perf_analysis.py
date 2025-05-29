#!/usr/bin/env python3
import os
import subprocess
import json
import time
from datetime import datetime

def run_perf_record(command, output_file, events=None):
    """使用perf record记录性能数据"""
    if events is None:
        events = [
            'cpu-clock',           # CPU时钟周期
            'cycles',             # CPU周期数
            'instructions',       # 指令数
            'cache-references',   # 缓存引用
            'cache-misses',       # 缓存未命中
            'branch-instructions', # 分支指令
            'branch-misses',      # 分支预测失败
            'L1-dcache-loads',    # L1数据缓存加载
            'L1-dcache-load-misses', # L1数据缓存加载未命中
            'L1-icache-load-misses', # L1指令缓存加载未命中
            'LLC-loads',          # 最后一级缓存加载
            'LLC-load-misses',    # 最后一级缓存加载未命中
            'dTLB-loads',         # 数据TLB加载
            'dTLB-load-misses',   # 数据TLB加载未命中
            'iTLB-loads',         # 指令TLB加载
            'iTLB-load-misses'    # 指令TLB加载未命中
        ]
    
    perf_cmd = ['perf', 'record', '-g', '-e', ','.join(events), '-o', output_file] + command
    try:
        subprocess.run(perf_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running perf record: {e}")
        return False

def run_perf_report(perf_data_file, output_file):
    """生成perf报告"""
    perf_cmd = ['perf', 'report', '-i', perf_data_file, '--stdio', '--sort', 'comm,dso,symbol']
    try:
        with open(output_file, 'w') as f:
            subprocess.run(perf_cmd, stdout=f, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running perf report: {e}")
        return False

def run_perf_stat(command, output_file, events=None):
    """使用perf stat收集统计信息"""
    if events is None:
        events = [
            'cpu-clock',
            'cycles',
            'instructions',
            'cache-references',
            'cache-misses',
            'branch-instructions',
            'branch-misses',
            'L1-dcache-loads',
            'L1-dcache-load-misses',
            'L1-icache-load-misses',
            'LLC-loads',
            'LLC-load-misses',
            'dTLB-loads',
            'dTLB-load-misses',
            'iTLB-loads',
            'iTLB-load-misses'
        ]
    
    perf_cmd = ['perf', 'stat', '-e', ','.join(events), '-o', output_file] + command
    try:
        subprocess.run(perf_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running perf stat: {e}")
        return False

def analyze_perf_data(perf_data_file, output_file):
    """分析perf数据并生成详细报告"""
    # 使用perf script生成详细的事件跟踪，并按阶段分类
    perf_cmd = ['perf', 'script', '-i', perf_data_file, '--time', '--cpu', '--event', '--sym']
    try:
        with open(output_file, 'w') as f:
            subprocess.run(perf_cmd, stdout=f, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing perf data: {e}")
        return False

def generate_stage_analysis(perf_data_file, output_dir):
    """生成各个训练阶段的性能分析报告"""
    stages = {
        'data_loading': 'Loading data',
        'preprocessing': 'Preprocessing',
        'model_building': 'Building model',
        'training': 'Training',
        'validation': 'Validation',
        'testing': 'Testing'
    }
    
    for stage_name, stage_desc in stages.items():
        # 为每个阶段生成性能报告
        stage_file = os.path.join(output_dir, f'{stage_name}_analysis.txt')
        perf_cmd = [
            'perf', 'report', '-i', perf_data_file,
            '--stdio',
            '--sort', 'comm,dso,symbol',
            '--symbol-filter', stage_desc
        ]
        try:
            with open(stage_file, 'w') as f:
                subprocess.run(perf_cmd, stdout=f, check=True)
            print(f"Generated analysis for {stage_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating analysis for {stage_name}: {e}")

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    perf_dir = os.path.join('results', 'perfs', timestamp)
    os.makedirs(perf_dir, exist_ok=True)

    # 定义要监控的性能事件
    events = [
        'cpu-clock',           # CPU时钟周期
        'cycles',             # CPU周期数
        'instructions',       # 指令数
        'cache-references',   # 缓存引用
        'cache-misses',       # 缓存未命中
        'branch-instructions', # 分支指令
        'branch-misses',      # 分支预测失败
        'L1-dcache-loads',    # L1数据缓存加载
        'L1-dcache-load-misses', # L1数据缓存加载未命中
        'L1-icache-load-misses', # L1指令缓存加载未命中
        'LLC-loads',          # 最后一级缓存加载
        'LLC-load-misses',    # 最后一级缓存加载未命中
        'dTLB-loads',         # 数据TLB加载
        'dTLB-load-misses',   # 数据TLB加载未命中
        'iTLB-loads',         # 指令TLB加载
        'iTLB-load-misses'    # 指令TLB加载未命中
    ]

    # 设置要分析的命令
    command = ['python', 'train.py']

    # 运行perf record和stat的组合命令
    print("Running perf record and stat...")
    perf_data_file = os.path.join(perf_dir, 'perf.data')
    stat_file = os.path.join(perf_dir, 'perf_stat.txt')
    
    # 组合perf record和stat命令
    perf_cmd = ['perf', 'record', '-g', '-e', ','.join(events), '-o', perf_data_file, '--', 'perf', 'stat', '-e', ','.join(events), '-o', stat_file] + command
    
    try:
        subprocess.run(perf_cmd, check=True)
        print(f"Perf data saved to {perf_data_file}")
        print(f"Perf statistics saved to {stat_file}")

        # 生成perf报告
        print("Generating perf report...")
        report_file = os.path.join(perf_dir, 'perf_report.txt')
        if run_perf_report(perf_data_file, report_file):
            print(f"Perf report saved to {report_file}")

        # 生成详细的事件跟踪
        print("Generating detailed event trace...")
        trace_file = os.path.join(perf_dir, 'perf_trace.txt')
        if analyze_perf_data(perf_data_file, trace_file):
            print(f"Detailed event trace saved to {trace_file}")
            
        # 生成各个阶段的性能分析
        print("Generating stage-wise performance analysis...")
        generate_stage_analysis(perf_data_file, perf_dir)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running perf: {e}")
        return

if __name__ == "__main__":
    main()