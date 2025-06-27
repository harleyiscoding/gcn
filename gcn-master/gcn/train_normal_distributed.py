from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss

from gcn.utils import *
from gcn.models import GCN, MLP
from partition_utils import metis_partition, get_partition_masks, extract_all_partition_subgraphs, compress_csr_with_delta_varint, decompress_csr_with_delta_varint, auto_compress_features, auto_decompress_features

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
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
flags.DEFINE_integer('num_parts', 4, 'Number of partitions.')

# 1. 加载全局数据并分区（保留全局变量）
adj_full, features_full, y_train_full, y_val_full, y_test_full, train_mask_full, val_mask_full, test_mask_full = load_data(FLAGS.dataset)
part_labels = metis_partition(adj_full, FLAGS.num_parts)
partition_masks = get_partition_masks(part_labels, FLAGS.num_parts)
subgraph_list = extract_all_partition_subgraphs(
    adj_full, features_full, y_train_full, y_val_full, y_test_full, train_mask_full, val_mask_full, test_mask_full, partition_masks)

# 2. 针对每个子图独立训练
all_subgraph_results = []
for part_id, subgraph in enumerate(subgraph_list):
    print(f"\n=== Training on Partition {part_id} ({subgraph['adj_sub'].shape[0]} nodes) ===")
    tf.reset_default_graph()  # 每个子图重置计算图

    # 数据准备（只用子图变量，不覆盖全局）
    adj_sub = subgraph['adj_sub']
    features_sub = subgraph['features_sub']
    y_train_sub = subgraph['y_train_sub']
    y_val_sub = subgraph['y_val_sub']
    y_test_sub = subgraph['y_test_sub']
    train_mask_sub = subgraph['train_mask_sub']
    val_mask_sub = subgraph['val_mask_sub']
    test_mask_sub = subgraph['test_mask_sub']
    part_nodes = partition_masks[part_id]

    # === 压缩并解压邻接矩阵 ===
    indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
    adj_sub_restored = decompress_csr_with_delta_varint(indptr, indices_bytes, data, adj_sub.shape)

    # === 自动压缩特征 ===
    compressed_features = auto_compress_features(features_sub)

    # === 解压特征并预处理 ===
    features_sub_restored = auto_decompress_features(compressed_features)
    features_sub_restored = preprocess_features(features_sub_restored)

    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj_sub_restored)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj_sub_restored, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj_sub_restored)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # 占位符
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_sub_restored[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train_sub.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # 创建模型
    model = model_func(placeholders, input_dim=features_sub_restored[2][1], logging=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 评估函数
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    cost_val = []
    # 训练
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features_sub_restored, support, y_train_sub, train_mask_sub, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        cost, acc, duration = evaluate(features_sub_restored, support, y_val_sub, val_mask_sub, placeholders)
        cost_val.append(cost)
        print(f"Partition {part_id} Epoch: {epoch+1:04d} train_loss={outs[1]:.5f} train_acc={outs[2]:.5f} val_loss={cost:.5f} val_acc={acc:.5f} time={time.time()-t:.5f}")
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
    print("Optimization Finished for Partition", part_id)
    # 测试推理，收集预测
    feed_dict_test = construct_feed_dict(features_sub_restored, support, y_test_sub, test_mask_sub, placeholders)
    y_pred_sub = sess.run(model.outputs, feed_dict=feed_dict_test)  # shape: [num_nodes_in_sub, num_classes]
    all_subgraph_results.append((part_nodes, y_pred_sub))

# === 合并所有子图预测，做全局评估 ===
print("\n=== 全局推理/评估 ===")
start_time = time.time()
num_nodes = adj_full.shape[0]  # 全图节点数
num_classes = y_test_full.shape[1]
y_pred_global = np.zeros((num_nodes, num_classes))
for part_nodes, y_pred_sub in all_subgraph_results:
    y_pred_global[part_nodes] = y_pred_sub
test_idx = np.where(test_mask_full)[0]
y_true = y_test_full[test_idx].argmax(1)
y_pred = y_pred_global[test_idx].argmax(1)
acc = accuracy_score(y_true, y_pred)
cost = log_loss(y_test_full[test_idx], y_pred_global[test_idx])
global_infer_time = time.time() - start_time
print("Test set results:", "cost=", "{:.5f}".format(cost),
      "accuracy=", "{:.5f}".format(acc), "time=", "{:.5f}".format(global_infer_time)) 