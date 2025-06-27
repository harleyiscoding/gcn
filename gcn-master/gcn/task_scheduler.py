import random
import time
import logging
import os
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf
from gcn.models import GCN, MLP
from gcn.utils import *
from partition_utils import (
    metis_partition, get_partition_masks, extract_all_partition_subgraphs,
    compress_csr_with_delta_varint, decompress_csr_with_delta_varint,
    auto_compress_features, auto_decompress_features
)

# === 自动插入TF1.x风格flags定义 ===
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
# === END ===

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def get_amir_threshold_kmeans(amir_values):
    amir_values = np.array(amir_values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(amir_values)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = (centers[0] + centers[1]) / 2
    return threshold, kmeans.labels_, centers

def get_cd_threshold_kmeans(cd_values):
    cd_values = np.array(cd_values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(cd_values)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = (centers[0] + centers[1]) / 2
    return threshold, kmeans.labels_, centers

# === 钩子+调度器集成（支持AMIR/CD分配执行单元） ===
class Scheduler:
    def __init__(self, amir_dir=None):
        self.history_amir = []
        self.history_cd = []
        self.amir_threshold = 5
        self.cd_threshold = 5
        self.amir_kmeans_labels = []
        self.amir_kmeans_centers = []
        self.cd_kmeans_labels = []
        self.cd_kmeans_centers = []
        self.amir_dir = amir_dir

    def update_thresholds(self):
        if len(self.history_amir) >= 20: 
            threshold, labels, centers = get_amir_threshold_kmeans(self.history_amir[-20:])
            self.amir_threshold = threshold
            self.amir_kmeans_labels = labels
            self.amir_kmeans_centers = centers
        if len(self.history_cd) >= 20: 
            threshold, labels, centers = get_cd_threshold_kmeans(self.history_cd[-20:])
            self.cd_threshold = threshold
            self.cd_kmeans_labels = labels
            self.cd_kmeans_centers = centers

    def schedule_task(self, phase, value):
        phase = phase.upper()
        assert phase in ('AGG', 'UPDATE'), f"Unsupported phase: {phase}"
        # 记录历史值
        if phase == 'AGG':
            self.history_amir.append(value)
        else:
            self.history_cd.append(value)
        self.update_thresholds()
        if phase == 'AGG':
            if value > self.amir_threshold:
                return 'PIM'
            else:
                return 'PNM'
        elif phase == 'UPDATE':
            if value > self.cd_threshold:
                return 'GPU'
            else:
                return 'PNM'

# 设备监控与回退
class DeviceMonitor:
    def __init__(self):
        self.device_status = {'GPU': 'normal', 'PIM': 'normal', 'PNM': 'normal'}
        self.overload_prob = {'GPU': 0.2, 'PIM': 0.1, 'PNM': 0.05}  # 模拟超载概率

    def update_status(self):
        for device in self.device_status:
            if random.random() < self.overload_prob[device]:
                self.device_status[device] = 'overloaded'
            else:
                self.device_status[device] = 'normal'
        logging.info(f"Device status: {self.device_status}")

    def is_overloaded(self, device):
        return self.device_status[device] == 'overloaded'

# 回退管理
class FallbackManager:
    def __init__(self, monitor):
        self.monitor = monitor

    def fallback(self, task):
        # 优先PIM, 其次PNM, 最后GPU
        candidates = ['PIM', 'PNM', 'GPU']
        for device in candidates:
            if not self.monitor.is_overloaded(device):
                old_device = task.device
                task.device = device
                task.status = 'fallback'
                logging.warning(f"Task {task.id} fallback from {old_device} to {device}")
                return
        logging.error(f"Task {task.id} cannot fallback, all devices overloaded!")

# 任务分发器
class Dispatcher:
    def dispatch(self, tasks):
        exec_time = {
            'PIM': 0.02,
            'PNM': 0.04,
            'GPU': 0.01
        }
        for task in tasks:
            # 日志区分回退任务
            if task.status == 'fallback':
                logging.info(f"Task {task.id} [fallback] dispatched to {task.device}")
            else:
                logging.info(f"Dispatching Task {task.id} to {task.device}")
            task.status = 'running'
            # 模拟小概率执行失败
            if random.random() < 0.05:
                task.status = 'failed'
                logging.error(f"Task {task.id} failed on {task.device}")
                continue
            # 模拟执行
            time.sleep(exec_time.get(task.device, 0.01))
            task.status = 'finished'
            logging.info(f"Task {task.id} finished on {task.device}")

# 读取memory_flops_epochs.txt所有epoch的AMIR和CD
def read_amir_cd_from_file(file_path):
    amir_list = []
    cd_list = []
    with open(file_path, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            try:
                agg_mem = float(parts[1]) + float(parts[2])  # layer1 + layer2 aggregation memory
                update_flops = float(parts[3]) + float(parts[4])  # layer1 + layer2 update flops
                if update_flops > 0 and agg_mem > 0:
                    amir = agg_mem / update_flops  # AMIR = MemoryAccess_aggregate / FLOPs_update
                    cd = update_flops / agg_mem    # CD = Update FLOPs / Aggregation Memory Accesses
                    amir_list.append(amir)
                    cd_list.append(cd)
            except Exception:
                continue
    return amir_list, cd_list

# 全局AMIR和CD聚类可视化
def plot_amir_cd_kmeans_full(amir_values, cd_values, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # AMIR聚类
    amir_values = np.array(amir_values).reshape(-1, 1)
    amir_kmeans = KMeans(n_clusters=2, random_state=0).fit(amir_values)
    amir_labels = amir_kmeans.labels_
    amir_centers = sorted(amir_kmeans.cluster_centers_.flatten())
    amir_threshold = (amir_centers[0] + amir_centers[1]) / 2
    
    ax1.scatter(range(len(amir_values)), amir_values, c=amir_labels, cmap='coolwarm', label='AMIR')
    ax1.axhline(amir_threshold, color='green', linestyle='--', label=f'Threshold={amir_threshold:.2f}')
    ax1.scatter([-1, -1], amir_centers, c='black', marker='x', s=100, label='Centers')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AMIR')
    ax1.set_title('K-means Clustering of All Epochs AMIR')
    ax1.legend()
    
    # CD聚类
    cd_values = np.array(cd_values).reshape(-1, 1)
    cd_kmeans = KMeans(n_clusters=2, random_state=0).fit(cd_values)
    cd_labels = cd_kmeans.labels_
    cd_centers = sorted(cd_kmeans.cluster_centers_.flatten())
    cd_threshold = (cd_centers[0] + cd_centers[1]) / 2
    
    ax2.scatter(range(len(cd_values)), cd_values, c=cd_labels, cmap='coolwarm', label='CD')
    ax2.axhline(cd_threshold, color='green', linestyle='--', label=f'Threshold={cd_threshold:.2f}')
    ax2.scatter([-1, -1], cd_centers, c='black', marker='x', s=100, label='Centers')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CD (Compute Density)')
    ax2.set_title('K-means Clustering of All Epochs CD')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"AMIR和CD聚类可视化已保存: {save_path}")

# 从memory_flops_epochs.txt读取真实AMIR和CD生成Task列表
def read_tasks_from_memory_flops(file_path):
    tasks = []
    with open(file_path, 'r') as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            try:
                # 计算总的聚合内存访问和更新FLOPs
                agg_mem = float(parts[1]) + float(parts[2])  # layer1 + layer2 aggregation memory
                update_flops = float(parts[3]) + float(parts[4])  # layer1 + layer2 update flops
                
                if update_flops > 0 and agg_mem > 0:
                    amir = agg_mem / update_flops  # AMIR = MemoryAccess_aggregate / FLOPs_update
                    cd = update_flops / agg_mem    # CD = Update FLOPs / Aggregation Memory Accesses
                    tasks.append(Task(task_id=i, AMIR=amir, CD=cd))
            except Exception:
                continue
    return tasks

def ensure_amir_dir(base_dir, dataset):
    amir_dir = os.path.join(base_dir, 'results', dataset, 'AMIR')
    os.makedirs(amir_dir, exist_ok=True)
    return amir_dir

# 主流程
if __name__ == "__main__":
    # 1. 数据加载与预处理
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # === 集成子图划分与压缩 ===
    num_parts = getattr(FLAGS, 'num_parts', 4)
    part_labels = metis_partition(adj, num_parts)
    partition_masks = get_partition_masks(part_labels, num_parts)
    subgraph_list = extract_all_partition_subgraphs(
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, partition_masks)

    # 2. 针对每个子图独立训练，调度、分发、回退等流程保持原有逻辑
    for part_id, subgraph in enumerate(subgraph_list):
        print(f"\n=== Training on Partition {part_id} ({subgraph['adj_sub'].shape[0]} nodes) ===")
        tf.reset_default_graph()
        adj_sub = subgraph['adj_sub']
        features_sub = subgraph['features_sub']
        y_train_sub = subgraph['y_train_sub']
        y_val_sub = subgraph['y_val_sub']
        y_test_sub = subgraph['y_test_sub']
        train_mask_sub = subgraph['train_mask_sub']
        val_mask_sub = subgraph['val_mask_sub']
        test_mask_sub = subgraph['test_mask_sub']

        # 压缩并解压邻接矩阵
        indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
        adj_sub_restored = decompress_csr_with_delta_varint(indptr, indices_bytes, data, adj_sub.shape)
        # 自动压缩特征
        compressed_features = auto_compress_features(features_sub)
        # 解压特征并预处理
        features_sub_restored = auto_decompress_features(compressed_features)
        features_sub_restored = preprocess_features(features_sub_restored)

        # ...后续调度、分发、回退、训练等流程与原task_scheduler.py一致...
        # 例如：
        # support = [preprocess_adj(adj_sub_restored)]
        # ...模型构建、占位符、训练、调度、分发、回退、日志...
        # ...全局推理/评估...

    # ...全局推理/评估代码...

