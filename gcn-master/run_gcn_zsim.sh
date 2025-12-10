#!/bin/bash
set -euo pipefail
PY=/opt/miniconda3/envs/hlgcn/bin/python
export PATH=/opt/miniconda3/envs/hlgcn/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/miniconda3/envs/hlgcn/lib
export PYTHONPATH=/workspace/hlexperience/gcnex/gcn-master:/workspace/hlexperience/gcnex/gcn-master/gcn
# 禁用 OpenMP CPU affinity 设置，避免在容器中权限错误
export OMP_PROC_BIND=false
export KMP_AFFINITY=disabled
# 设置 OpenMP 线程数：建议设置为 bank 数量的倍数或约数
# 对于 8/16/32 bank，设置为 8 可以均匀利用 bank 并行性
# 如果未设置，OpenMP 会使用系统默认值（通常是 CPU 核心数）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# 确保 Python 输出不被缓冲（实时输出）
export PYTHONUNBUFFERED=1
cd /workspace/hlexperience/gcnex/gcn-master
# 使用 -u 参数确保输出不被缓冲
exec "$PY" -u -m gcn.task_scheduler_distributed --dataset cora --epochs 1 --enable_roi_marking
