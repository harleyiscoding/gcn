from __future__ import division
from __future__ import print_function

import time
import os
import json
import numpy as np
import tensorflow as tf
import subprocess

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
perf_dir = os.path.join(FLAGS.results_dir, 'perfs', timestamp)
os.makedirs(perf_dir, exist_ok=True)

def start_perf(stage, epoch, perf_dir):
    """启动perf统计"""
    stage_dir = os.path.join(perf_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    
    perf_file = os.path.join(stage_dir, f'{stage}_stats_{epoch}.txt')
    events = [
        'branch-misses',
        'cache-misses',
        'cache-references',
        'cpu-cycles',
        'instructions',
        'L1-dcache-load-misses',
        'L1-dcache-loads',
        'L1-dcache-stores',
        'LLC-loads',
        'LLC-load-misses',
        'branch-load-misses',
        'branch-loads',
        'dTLB-load-misses',
        'dTLB-loads',
        'fp_arith_inst_retired.scalar_single',
        'fp_arith_inst_retired.scalar_double',
        'fp_arith_inst_retired.128b_packed_single',
        'fp_arith_inst_retired.128b_packed_double'
    ]
    
    return subprocess.Popen([
        'perf', 'stat',
        '-e', ','.join(events),
        '-o', perf_file,
        '-p', str(os.getpid())
    ])

def stop_perf(perf_proc):
    """停止perf统计"""
    perf_proc.send_signal(subprocess.signal.SIGINT)
    perf_proc.wait()

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

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
print("Starting training...")
for epoch in range(FLAGS.epochs):
    t = time.time()
    
    # === Step 1: 构造输入 ===
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    
    # === Step 2: 第1层 GCN - 聚合邻居特征 ===
    print(f"\nEpoch {epoch + 1} - Layer 1:")
    
    # 特征变换（更新）
    print("Layer 1 - Update Phase:")
    perf_proc = start_perf('layer1_update', epoch, perf_dir)
    updated = sess.run([model.layers[0]._update(placeholders['features'])], feed_dict=feed_dict)
    stop_perf(perf_proc)
    
    # 邻居信息聚合
    print("Layer 1 - Aggregation Phase:")
    perf_proc = start_perf('layer1_aggregation', epoch, perf_dir)
    aggregated = sess.run([model.layers[0]._aggregate(updated[0])], feed_dict=feed_dict)
    stop_perf(perf_proc)
    
    # === Step 3: 第2层 GCN - 聚合+分类输出 ===
    print(f"Epoch {epoch + 1} - Layer 2:")
    
    # 特征变换（更新）
    print("Layer 2 - Update Phase:")
    perf_proc = start_perf('layer2_update', epoch, perf_dir)
    updated = sess.run([model.layers[1]._update(aggregated[0])], feed_dict=feed_dict)
    stop_perf(perf_proc)
    
    # 邻居信息聚合
    print("Layer 2 - Aggregation Phase:")
    perf_proc = start_perf('layer2_aggregation', epoch, perf_dir)
    outputs = sess.run([model.layers[1]._aggregate(updated[0])], feed_dict=feed_dict)
    stop_perf(perf_proc)

    # === Step 4: 计算当前 batch 的损失和准确率 ===
    loss_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    
    # === Step 5: 执行反向传播，更新参数 ===
    sess.run([model.opt_op], feed_dict=feed_dict)

    # === Step 6: 验证集评估 ===
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # === Step 7: 打印当前 epoch 结果 ===
    print("Epoch:", '%04d' % (epoch + 1), 
          "train_loss=", "{:.5f}".format(loss_acc[0]),
          "train_acc=", "{:.5f}".format(loss_acc[1]), 
          "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), 
          "time=", "{:.5f}".format(time.time() - t))

    # === Step 8: 提前停止判断 ===
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

# Testing
print("\nRunning final test...")
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))