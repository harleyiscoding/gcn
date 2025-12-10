# -*- coding: utf-8 -*-
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

# 导入 zsim_hooks（可选）- 用于 ROI 标记
import sys

try:
    import zsim_hooks_python as zsim_hooks
    ZSIM_HOOKS_AVAILABLE = True
    logging.info("✓ zsim_hooks_python 已成功加载并可用")
except ImportError as e:
    ZSIM_HOOKS_AVAILABLE = False
    logging.warning(f"zsim_hooks_python not available, ROI marking disabled: {e}")

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
flags.DEFINE_integer('num_parts', 4, 'Number of partitions for metis_partition.')
flags.DEFINE_boolean('enable_roi_marking', False, 'Enable ROI marking with zsim_hooks for PIM/PNM tasks.')
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

def _ensure_np_array(indices):
    return np.array(indices, dtype=np.int64)

def map_features_to_global(features_tuple, part_nodes, num_nodes, num_features):
    """将子图特征三元组映射到全图形状"""
    coords, values, _ = features_tuple
    part_nodes = _ensure_np_array(part_nodes)
    if coords.size == 0:
        new_coords = coords.reshape(0, 2).astype(np.int64)
    else:
        new_coords = coords.copy().astype(np.int64)
        new_coords[:, 0] = part_nodes[new_coords[:, 0]]
    return (new_coords, values, (num_nodes, num_features))

def map_support_to_global(support_tuple, part_nodes, num_nodes):
    """将子图邻接三元组映射到全图形状"""
    coords, values, _ = support_tuple
    part_nodes = _ensure_np_array(part_nodes)
    if coords.size == 0:
        new_coords = coords.reshape(0, 2).astype(np.int64)
    else:
        new_coords = coords.copy().astype(np.int64)
        new_coords[:, 0] = part_nodes[new_coords[:, 0]]
        new_coords[:, 1] = part_nodes[new_coords[:, 1]]
    return (new_coords, values, (num_nodes, num_nodes))

def expand_labels_to_global(labels_sub, part_nodes, num_nodes):
    labels_global = np.zeros((num_nodes, labels_sub.shape[1]), dtype=labels_sub.dtype)
    labels_global[_ensure_np_array(part_nodes)] = labels_sub
    return labels_global

def expand_mask_to_global(mask_sub, part_nodes, num_nodes):
    mask_global = np.zeros(num_nodes, dtype=mask_sub.dtype)
    mask_global[_ensure_np_array(part_nodes)] = mask_sub
    return mask_global

# 主流程
if __name__ == "__main__":
    # 1. 数据加载与预处理
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # === 新增：子图划分与压缩 ===
    num_parts = FLAGS.num_parts if hasattr(FLAGS, 'num_parts') else 4
    part_labels = metis_partition(adj, num_parts)
    partition_masks = get_partition_masks(part_labels, num_parts)
    subgraph_list = extract_all_partition_subgraphs(
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, partition_masks)

    # 预处理 & 压缩子图（只做一次）
    for i, subgraph in enumerate(subgraph_list):
        # 压缩邻接矩阵
        adj_sub = subgraph['adj_sub']
        indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
        subgraph['adj_compressed'] = (indptr, indices_bytes, data, adj_sub.shape)
        # 压缩特征
        features_sub = subgraph['features_sub']
        subgraph['features_compressed'] = auto_compress_features(features_sub)
        # 预先计算映射到全图的三元组和标签/掩码
        part_nodes = _ensure_np_array(subgraph['part_nodes'])
        adj_sub_dec = decompress_csr_with_delta_varint(*subgraph['adj_compressed'])
        features_sub_dec = auto_decompress_features(subgraph['features_compressed'])
        features_sub_tuple = preprocess_features(features_sub_dec)
        support_sub = [preprocess_adj(adj_sub_dec)]
        features_global = map_features_to_global(features_sub_tuple, part_nodes, adj.shape[0], features.shape[1])
        support_global = [map_support_to_global(s, part_nodes, adj.shape[0]) for s in support_sub]
        y_train_global = expand_labels_to_global(subgraph['y_train_sub'], part_nodes, adj.shape[0])
        y_val_global = expand_labels_to_global(subgraph['y_val_sub'], part_nodes, adj.shape[0])
        y_test_global = expand_labels_to_global(subgraph['y_test_sub'], part_nodes, adj.shape[0])
        train_mask_global = expand_mask_to_global(subgraph['train_mask_sub'], part_nodes, adj.shape[0])
        val_mask_global = expand_mask_to_global(subgraph['val_mask_sub'], part_nodes, adj.shape[0])
        test_mask_global = expand_mask_to_global(subgraph['test_mask_sub'], part_nodes, adj.shape[0])
        subgraph['cached_global'] = {
            'features_global': features_global,
            'support_global': support_global,
            'y_train_global': y_train_global,
            'y_val_global': y_val_global,
            'y_test_global': y_test_global,
            'train_mask_global': train_mask_global,
            'val_mask_global': val_mask_global,
            'test_mask_global': test_mask_global,
            'part_nodes': part_nodes,
        }

    # 全局特征和support也预处理为三元组
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # 2. 占位符定义（用三元组shape，num_features_nonzero shape=[1]）
    num_nodes, num_features = features[2]
    num_classes = y_train.shape[1]
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, shape=(num_nodes, num_nodes)) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=(num_nodes, num_features)),
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes)),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32, shape=[1])
    }

    # 3. 模型构建
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    
    # === END ===

    # 4. （原先这里是全局 Session 和 feed_dict，现在改为在每个子图内部单独建图/Session）

    # 6. 解析memory_flops_epochs.txt，生成每层AGG/UPDATE的特征值
    base_dir = os.path.dirname(os.path.abspath(__file__))
    memory_flops_path = os.path.join(base_dir, 'results', FLAGS.dataset, 'l1_cache_analysis', 'memory_flops_epochs.txt')
    assert os.path.exists(memory_flops_path), f"{memory_flops_path} not found!"
    tasks_info = []
    with open(memory_flops_path, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            l1_agg_mem = float(parts[1])
            l2_agg_mem = float(parts[2])
            l1_update_flops = float(parts[3])
            l2_update_flops = float(parts[4])
            # AMIR用于AGG，CD用于UPDATE
            if l1_update_flops > 0:
                amir1 = l1_agg_mem / l1_update_flops
                cd1 = l1_update_flops / l1_agg_mem if l1_agg_mem > 0 else 1.0
            else:
                amir1 = 1.0
                cd1 = 1.0
            if l2_update_flops > 0:
                amir2 = l2_agg_mem / l2_update_flops
                cd2 = l2_update_flops / l2_agg_mem if l2_agg_mem > 0 else 1.0
            else:
                amir2 = 1.0
                cd2 = 1.0
            # L1 AGG 使用 AMIR
            tasks_info.append({'layer': 1, 'phase': 'UPDATE', 'value': cd1})
            # L1 UPDATE 使用 CD
            tasks_info.append({'layer': 1, 'phase': 'AGG', 'value': amir1})
            # L2 AGG 使用 AMIR
            tasks_info.append({'layer': 2, 'phase': 'UPDATE', 'value': cd2})
            # L2 UPDATE 使用 CD
            tasks_info.append({'layer': 2, 'phase': 'AGG', 'value': amir2})

    scheduler = Scheduler()
    current_epoch = [0]
    stage_counter = [0]
    stage_device_log = []
    
    # 检查 zsim_hooks 是否可用（用于 ROI 标记）
    if FLAGS.enable_roi_marking and not ZSIM_HOOKS_AVAILABLE:
        logging.warning("=" * 60)
        logging.warning("⚠️  警告: ROI marking 已启用但 zsim_hooks_python 不可用！")
        logging.warning("   这会导致 trace 文件为空（如果 only_offload=true）")
        logging.warning("   解决方案：")
        logging.warning("   1. 确保在 ZSim 环境下运行（通过 zsim 启动）")
        logging.warning("   2. 检查 zsim_hooks_python 是否正确安装")
        logging.warning("   3. 或者修改 gcn_host.cfg: only_offload = false（生成完整 trace）")
        logging.warning("=" * 60)
    elif FLAGS.enable_roi_marking and ZSIM_HOOKS_AVAILABLE:
        logging.info("✓ ROI marking 已启用，zsim_hooks 可用，将生成 ROI trace")
    else:
        logging.info("ℹ ROI marking 未启用（--enable_roi_marking=False）")

    def stage_hook(stage, layer_idx, info=None):
        # 只在每个子阶段BEGIN时调度
        if stage.endswith('BEGIN'):
            idx = current_epoch[0] * 4 + stage_counter[0]
            if idx < len(tasks_info):
                task = tasks_info[idx]
                device = scheduler.schedule_task(task['phase'], task['value'])
                print(f"[调度器] Epoch {current_epoch[0]+1} 子阶段{stage_counter[0]+1} (Layer {layer_idx}, {stage}): 值={task['value']:.4f}, 分配到 {device}")
                stage_device_log.append((current_epoch[0]+1, stage_counter[0]+1, layer_idx, stage, task['value'], None, device))
            stage_counter[0] += 1
            if stage_counter[0] == 4:
                stage_counter[0] = 0
                current_epoch[0] += 1

    # === ROI 包裹辅助函数 ===
    def _run_with_roi(func, device, task_id=None, *args, **kwargs):
        """
        参照 BFS 示例的 ROI 包裹方式：
        - 如果设备是 PIM/PNM 且启用了 ROI 标记，自动包裹 zsim_roi_begin/end
        - 否则直接执行
        """
        if device in ('PIM', 'PNM') and FLAGS.enable_roi_marking and ZSIM_HOOKS_AVAILABLE:
            if task_id:
                logging.info(f"[ROI BEGIN] {task_id} on {device}")
            zsim_hooks.zsim_roi_begin()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                zsim_hooks.zsim_roi_end()
                if task_id:
                    logging.info(f"[ROI END] {task_id} on {device}")
        else:
            # GPU 或其他设备，或未启用 ROI，直接执行
            return func(*args, **kwargs)

    def _exec_stage(layer_idx, phase, epoch, part_id, sess, model, placeholders, feed_dict, 
                    input_data=None):
        """
        统一执行一个阶段（Update 或 Aggregate），自动处理调度和 ROI 标记
        参照 BFS 示例的结构化方式
        """
        # 获取任务信息和调度决策
        idx = epoch * 4 + (0 if phase == 'UPDATE' and layer_idx == 1 else
                           1 if phase == 'AGG' and layer_idx == 1 else
                           2 if phase == 'UPDATE' and layer_idx == 2 else 3)
        
        if idx < len(tasks_info):
            task = tasks_info[idx]
            device = scheduler.schedule_task(task['phase'], task['value'])
            task_value = task['value']
        else:
            device = scheduler.schedule_task(phase, 1.0)
            task_value = 1.0
        
        print(f"[调度器] Layer {layer_idx} {phase}: 值={task_value:.4f}, 分配到 {device}")
        stage_device_log.append((epoch+1, idx % 4 + 1, layer_idx, phase, task_value, part_id, device))
        
        # 构建 task_id 用于日志
        task_id = f"L{layer_idx}_{phase}_P{part_id}_E{epoch}"
        
        # 执行计算（自动处理 ROI 包裹）
        def _compute():
            layer = model.layers[layer_idx - 1]  # layer_idx 从 1 开始
            if phase == 'UPDATE':
                # Update 阶段：第一个 layer 的 Update 使用 placeholders，后续使用前一个阶段的输出
                if input_data is None:
                    update_input = placeholders['features']
                else:
                    update_input = input_data
                return sess.run([layer._update(update_input)], feed_dict=feed_dict)
            else:  # AGG
                # Aggregate 阶段：总是使用前一个 Update 阶段的输出
                if input_data is None:
                    # 这种情况不应该发生，但为了安全起见
                    agg_input = placeholders['features']
                else:
                    agg_input = input_data
                return sess.run([layer._aggregate(agg_input)], feed_dict=feed_dict)
        
        result = _run_with_roi(_compute, device, task_id)
        print(f"[完成] Layer {layer_idx} {phase} 在 {device} 上完成 (Partition {part_id})")
        return result

    # 7. 子图尺寸模型 + 调度训练（每个子图单独建图和Session）
    all_subgraph_results = []
    for part_id, subgraph in enumerate(subgraph_list):
        print(f"\n=== 调度训练 Partition {part_id} ({subgraph['adj_sub'].shape[0]} nodes) ===")
        tf.reset_default_graph()

        # 子图数据
        adj_sub = subgraph['adj_sub']
        features_sub = subgraph['features_sub']
        y_train_sub = subgraph['y_train_sub']
        y_val_sub = subgraph['y_val_sub']
        y_test_sub = subgraph['y_test_sub']
        train_mask_sub = subgraph['train_mask_sub']
        val_mask_sub = subgraph['val_mask_sub']
        test_mask_sub = subgraph['test_mask_sub']
        part_nodes = _ensure_np_array(subgraph['part_nodes'])

        # 压缩 / 解压 邻接
        indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
        adj_sub_restored = decompress_csr_with_delta_varint(indptr, indices_bytes, data, adj_sub.shape)

        # 压缩 / 解压 特征，并预处理
        compressed_features = auto_compress_features(features_sub)
        features_sub_restored = auto_decompress_features(compressed_features)
        features_sub_restored = preprocess_features(features_sub_restored)

        # 支持集（按子图尺寸）
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj_sub_restored)]
            num_supports_sub = 1
            model_func_sub = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj_sub_restored, FLAGS.max_degree)
            num_supports_sub = 1 + FLAGS.max_degree
            model_func_sub = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj_sub_restored)]
            num_supports_sub = 1
            model_func_sub = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # 占位符（子图尺寸）
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports_sub)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_sub_restored[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train_sub.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

        # 模型 & Session
        model = model_func_sub(placeholders, input_dim=features_sub_restored[2][1], logging=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 子图评估函数（使用 utils.construct_feed_dict）
        def evaluate(features, support_local, labels, mask, placeholders_local):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support_local, labels, mask, placeholders_local)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)

        cost_val = []
        early_stop = False
        for epoch in range(FLAGS.epochs):
            t = time.time()
            print(f"\n=== Partition {part_id} Epoch {epoch + 1} ===")

            feed_dict = construct_feed_dict(features_sub_restored, support, y_train_sub, train_mask_sub, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # 每个epoch开始时重置子阶段计数
            stage_counter[0] = 0
            current_epoch[0] = epoch

            # Layer 1 Update（使用统一的 _exec_stage 函数，参照 BFS 示例）
            updated = _exec_stage(
                layer_idx=1, phase='UPDATE', epoch=epoch, part_id=part_id,
                sess=sess, model=model, placeholders=placeholders, feed_dict=feed_dict
            )

            # Layer 1 Aggregate
            aggregated = _exec_stage(
                layer_idx=1, phase='AGG', epoch=epoch, part_id=part_id,
                sess=sess, model=model, placeholders=placeholders, feed_dict=feed_dict,
                input_data=updated[0]
            )

            # Layer 2 Update
            updated = _exec_stage(
                layer_idx=2, phase='UPDATE', epoch=epoch, part_id=part_id,
                sess=sess, model=model, placeholders=placeholders, feed_dict=feed_dict,
                input_data=aggregated[0]
            )

            # Layer 2 Aggregate
            outputs = _exec_stage(
                layer_idx=2, phase='AGG', epoch=epoch, part_id=part_id,
                sess=sess, model=model, placeholders=placeholders, feed_dict=feed_dict,
                input_data=updated[0]
            )

            # 损失和准确率 + 反向传播
            loss_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
            sess.run([model.opt_op], feed_dict=feed_dict)

            cost, acc, duration = evaluate(features_sub_restored, support, y_val_sub, val_mask_sub, placeholders)
            cost_val.append(cost)
            print(f"Partition {part_id} Epoch: {epoch+1:04d} train_loss={loss_acc[0]:.5f} train_acc={loss_acc[1]:.5f} val_loss={cost:.5f} val_acc={acc:.5f} time={time.time()-t:.5f}")
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                early_stop = True
                break

        print("Optimization Finished for Partition", part_id)

        # 测试推理，收集预测（子图尺寸）
        feed_dict_test = construct_feed_dict(features_sub_restored, support, y_test_sub, test_mask_sub, placeholders)
        y_pred_sub = sess.run(model.outputs, feed_dict=feed_dict_test)
        all_subgraph_results.append((part_nodes, y_pred_sub))

    print("\n=== 调度器阶段分配日志 ===")
    pim_count = 0
    pnm_count = 0
    gpu_count = 0
    for log in stage_device_log:
        device = log[6]
        if device == 'PIM':
            pim_count += 1
        elif device == 'PNM':
            pnm_count += 1
        elif device == 'GPU':
            gpu_count += 1
        print(f"Epoch {log[0]} 阶段{log[1]} (Layer {log[2]} {log[3]}): 值={log[4]:.4f}, 分配到 {device}")
    
    total_tasks = len(stage_device_log)
    pim_pnm_count = pim_count + pnm_count
    logging.info(f"任务分配统计: 总任务数={total_tasks}, PIM={pim_count}, PNM={pnm_count}, GPU={gpu_count}, PIM+PNM={pim_pnm_count}")

    # 只保留子图分区推理结果（all_subgraph_results 已在上面收集）
    print("\n=== 子图分区推理与评估 ===")
    print("子图推理结果已收集，总分区数：", len(all_subgraph_results))
    
    # 保存任务信息到文件，供后续独立生成 trace 使用
    pim_pnm_tasks = [log for log in stage_device_log if log[6] in ('PIM', 'PNM')]
    if pim_pnm_tasks:
        # 创建 trace 目录：results/{dataset}/trace/
        trace_dir = os.path.join(base_dir, 'results', FLAGS.dataset, 'trace')
        os.makedirs(trace_dir, exist_ok=True)
        
        # 保存任务信息到 trace 目录
        trace_tasks_file = os.path.join(trace_dir, 'trace_tasks.json')
        
        import json
        trace_tasks_data = []
        for log in pim_pnm_tasks:
            task_id = f"L{log[2]}_{log[3]}_P{log[5]}_E{log[0]}"
            trace_tasks_data.append({
                'epoch': log[0],
                'stage': log[1],
                'layer': log[2],
                'phase': log[3],
                'value': log[4],
                'partition_id': log[5],
                'device': log[6],
                'task_id': task_id,
                # 添加预期的 trace 文件路径信息
                'trace_file': os.path.join(trace_dir, f"gcn_{log[6]}_{log[3]}_{task_id}.out"),
                'stats_file': os.path.join(trace_dir, f"ramulator_stats_{task_id}.stats")
            })
        
        with open(trace_tasks_file, 'w') as f:
            json.dump(trace_tasks_data, f, indent=2)
        print(f"\n任务信息已保存到: {trace_tasks_file}")
        print(f"Trace 目录: {trace_dir}")
        print(f"需要 trace 的任务数: {len(pim_pnm_tasks)} (PIM/PNM 任务)")
        print("提示：可以使用独立脚本在 ZSim 环境下生成 trace，然后使用 Ramulator 分析")
        print(f"     生成的 trace 文件将保存到: {trace_dir}")

    # 8. 全局AMIR和CD聚类分析与可视化
    amir_dir = os.path.join(base_dir, 'results', FLAGS.dataset, 'AMIR')
    os.makedirs(amir_dir, exist_ok=True)
    amir_list, cd_list = read_amir_cd_from_file(memory_flops_path)
    save_path = os.path.join(amir_dir, 'amir_cd_kmeans_full.png')
    plot_amir_cd_kmeans_full(amir_list, cd_list, save_path)

