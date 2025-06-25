from __future__ import division
from __future__ import print_function

import time
import os
import json
import numpy as np
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('results_dir', './results/pubmed', 'Directory to save results.')

# Generate a timestamp for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Create directories if they don't exist
flops_dir = os.path.join(FLAGS.results_dir, 'tf-flops', timestamp)
os.makedirs(flops_dir, exist_ok=True)

# Load data
print("Loading data...")
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
print("Preprocessing data...")
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
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
print("Building model...")
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# 计算FLOPs的函数 - 使用提供的公式
def calculate_flops_using_formula(adj, features, hidden_dim):
    """
    使用公式计算FLOPs:
    - MemoryAccess_agg ≈ adj.nnz × feature_dim
    - FLOPs_update ≈ num_nodes × feature_dim × hidden_dim
    """
    num_nodes = adj.shape[0]  # 节点数量
    feature_dim = features[2][1]  # 特征维度
    adj_nnz = adj.nnz  # 邻接矩阵非零元素数量
    
    # 计算各层的FLOPs
    # Layer 1
    layer1_update_flops = num_nodes * feature_dim * hidden_dim
    layer1_aggregate_flops = adj_nnz * feature_dim
    
    # Layer 2 (输出层)
    output_dim = y_train.shape[1]  # 输出维度（类别数）
    layer2_update_flops = num_nodes * hidden_dim * output_dim
    layer2_aggregate_flops = adj_nnz * hidden_dim
    
    return {
        'layer1_update': layer1_update_flops,
        'layer1_aggregate': layer1_aggregate_flops,
        'layer2_update': layer2_update_flops,
        'layer2_aggregate': layer2_aggregate_flops
    }

# 计算AMIR (Memory Access Intensity Ratio)
def calculate_amir(flops_dict):
    """
    计算AMIR = MemoryAccess_aggregate / FLOPs_update
    """
    total_aggregate_flops = flops_dict['layer1_aggregate'] + flops_dict['layer2_aggregate']
    total_update_flops = flops_dict['layer1_update'] + flops_dict['layer2_update']
    
    amir = total_aggregate_flops / total_update_flops if total_update_flops > 0 else 0
    return amir

# 计算数据集的基本信息
print("Calculating dataset statistics...")
dataset_stats = {
    'num_nodes': adj.shape[0],
    'feature_dim': features[2][1],
    'hidden_dim': FLAGS.hidden1,
    'output_dim': y_train.shape[1],
    'adj_nnz': adj.nnz
}
print(f"Dataset stats: {dataset_stats}")

# Train model and calculate FLOPs
print("Starting training and FLOPs calculation...")
for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # 执行前向传播
    outputs = sess.run([model.outputs], feed_dict=feed_dict)
    
    # 计算损失和准确率
    loss_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)

    # 执行反向传播，更新参数
    sess.run([model.opt_op], feed_dict=feed_dict)

    # 计算当前epoch的FLOPs
    flops_epoch = calculate_flops_using_formula(adj, features, FLAGS.hidden1)
    
    # 计算AMIR
    amir = calculate_amir(flops_epoch)
    
    # 添加AMIR到结果中
    flops_epoch['amir'] = amir

    # 保存FLOPs数据
    with open(os.path.join(flops_dir, f'flops_epoch_{epoch+1}.json'), 'w') as f:
        json.dump(flops_epoch, f, indent=2)

    # 打印当前 epoch 结果
    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(loss_acc[0]),
          "train_acc=", "{:.5f}".format(loss_acc[1]),
          "AMIR=", "{:.4f}".format(amir),
          "time=", "{:.5f}".format(time.time() - t))

# 保存数据集统计信息
with open(os.path.join(flops_dir, 'dataset_stats.json'), 'w') as f:
    json.dump(dataset_stats, f, indent=2)

# 计算并保存总体统计
total_flops = calculate_flops_using_formula(adj, features, FLAGS.hidden1)
total_amir = calculate_amir(total_flops)

summary = {
    'total_flops': total_flops,
    'total_amir': total_amir,
    'dataset_stats': dataset_stats
}

with open(os.path.join(flops_dir, 'flops_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("FLOPs calculation completed. Results saved to:", flops_dir)
print(f"Total AMIR: {total_amir:.4f}")
print(f"Layer 1 Update FLOPs: {total_flops['layer1_update']:,}")
print(f"Layer 1 Aggregate FLOPs: {total_flops['layer1_aggregate']:,}")
print(f"Layer 2 Update FLOPs: {total_flops['layer2_update']:,}")
print(f"Layer 2 Aggregate FLOPs: {total_flops['layer2_aggregate']:,}") 