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

    # 2. 占位符定义
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # 3. 模型构建
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    
    # === END ===

    # 4. Session初始化
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 5. feed_dict构造
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

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

    # 7. 标准端到端训练与验证流程（与train_normal.py一致）
    cost_val = []
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # 每个epoch开始时重置子阶段计数（防止断点/早停后错乱）
        stage_counter[0] = 0
        current_epoch[0] = epoch

        # === 新的训练步骤：分阶段执行update和aggregate ===
        print(f"\n=== Epoch {epoch + 1} 训练开始 ===")
        
        # Layer 1 Update
        print(f"Layer 1 Update - 调度任务...")
        idx = epoch * 4 + 0  # Layer 1 Update
        if idx < len(tasks_info):
            task = tasks_info[idx]
            device = scheduler.schedule_task(task['phase'], task['value'])
            print(f"[调度器] Layer 1 Update: 值={task['value']:.4f}, 分配到 {device}")
            stage_device_log.append((epoch+1, 1, 1, 'UPDATE', task['value'], None, device))
            print(f"[完成] Layer 1 Update 在 {device} 上完成")
            updated = sess.run([model.layers[0]._update(placeholders['features'])], feed_dict=feed_dict)
        else:
            device = scheduler.schedule_task('UPDATE', 1.0)
            print(f"[调度器] Layer 1 Update: 值=1.0000, 分配到 {device}")
            stage_device_log.append((epoch+1, 1, 1, 'UPDATE', 1.0, None, device))
            print(f"[完成] Layer 1 Update 在 {device} 上完成")
            updated = sess.run([model.layers[0]._update(placeholders['features'])], feed_dict=feed_dict)
        
        # Layer 1 Aggregate
        print(f"Layer 1 Aggregate - 调度任务...")
        idx = epoch * 4 + 1  # Layer 1 Aggregate
        if idx < len(tasks_info):
            task = tasks_info[idx]
            device = scheduler.schedule_task(task['phase'], task['value'])
            print(f"[调度器] Layer 1 Aggregate: 值={task['value']:.4f}, 分配到 {device}")
            stage_device_log.append((epoch+1, 2, 1, 'AGG', task['value'], None, device))
            print(f"[完成] Layer 1 Aggregate 在 {device} 上完成")
            aggregated = sess.run([model.layers[0]._aggregate(updated[0])], feed_dict=feed_dict)
        else:
            device = scheduler.schedule_task('AGG', 1.0)
            print(f"[调度器] Layer 1 Aggregate: 值=1.0000, 分配到 {device}")
            stage_device_log.append((epoch+1, 2, 1, 'AGG', 1.0, None, device))
            print(f"[完成] Layer 1 Aggregate 在 {device} 上完成")
            aggregated = sess.run([model.layers[0]._aggregate(updated[0])], feed_dict=feed_dict)
        
        # Layer 2 Update
        print(f"Layer 2 Update - 调度任务...")
        idx = epoch * 4 + 2  # Layer 2 Update
        if idx < len(tasks_info):
            task = tasks_info[idx]
            device = scheduler.schedule_task(task['phase'], task['value'])
            print(f"[调度器] Layer 2 Update: 值={task['value']:.4f}, 分配到 {device}")
            stage_device_log.append((epoch+1, 3, 2, 'UPDATE', task['value'], None, device))
            print(f"[完成] Layer 2 Update 在 {device} 上完成")
            updated = sess.run([model.layers[1]._update(aggregated[0])], feed_dict=feed_dict)
        else:
            device = scheduler.schedule_task('UPDATE', 1.0)
            print(f"[调度器] Layer 2 Update: 值=1.0000, 分配到 {device}")
            stage_device_log.append((epoch+1, 3, 2, 'UPDATE', 1.0, None, device))
            print(f"[完成] Layer 2 Update 在 {device} 上完成")
            updated = sess.run([model.layers[1]._update(aggregated[0])], feed_dict=feed_dict)
        
        # Layer 2 Aggregate
        print(f"Layer 2 Aggregate - 调度任务...")
        idx = epoch * 4 + 3  # Layer 2 Aggregate
        if idx < len(tasks_info):
            task = tasks_info[idx]
            device = scheduler.schedule_task(task['phase'], task['value'])
            print(f"[调度器] Layer 2 Aggregate: 值={task['value']:.4f}, 分配到 {device}")
            stage_device_log.append((epoch+1, 4, 2, 'AGG', task['value'], None, device))
            print(f"[完成] Layer 2 Aggregate 在 {device} 上完成")
            outputs = sess.run([model.layers[1]._aggregate(updated[0])], feed_dict=feed_dict)
        else:
            device = scheduler.schedule_task('AGG', 1.0)
            print(f"[调度器] Layer 2 Aggregate: 值=1.0000, 分配到 {device}")
            stage_device_log.append((epoch+1, 4, 2, 'AGG', 1.0, None, device))
            print(f"[完成] Layer 2 Aggregate 在 {device} 上完成")
            outputs = sess.run([model.layers[1]._aggregate(updated[0])], feed_dict=feed_dict)
        
        # 计算损失和准确率
        loss_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        
        # 执行反向传播更新参数
        sess.run([model.opt_op], feed_dict=feed_dict)

        # 验证集评估
        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_acc[0]),
              "train_acc=", "{:.5f}".format(loss_acc[1]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # Early stopping
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # 输出调度日志
    print("\n=== 调度器阶段分配日志 ===")
    for log in stage_device_log:
        print(f"Epoch {log[0]} 阶段{log[1]} (Layer {log[2]} {log[3]}): 值={log[4]:.4f}, 分配到 {log[6]}")

    # 测试集评估
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # 8. 全局AMIR和CD聚类分析与可视化
    amir_dir = os.path.join(base_dir, 'results', FLAGS.dataset, 'AMIR')
    os.makedirs(amir_dir, exist_ok=True)
    amir_list, cd_list = read_amir_cd_from_file(memory_flops_path)
    save_path = os.path.join(amir_dir, 'amir_cd_kmeans_full.png')
    plot_amir_cd_kmeans_full(amir_list, cd_list, save_path)

    # 8. 可视化、聚类等分析（可复用原有代码）
    # ... 
