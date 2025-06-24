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

# Helper to提取scope FLOPs
# 新增：遍历所有 scope，累加以 prefix 开头的 FLOPs
def collect_flops_by_prefix(profile, prefixes):
    flops_dict = {prefix: 0 for prefix in prefixes}
    def traverse(node):
        for prefix in prefixes:
            if node.name.startswith(prefix):
                flops_dict[prefix] += node.total_float_ops
        for child in getattr(node, 'children', []):
            traverse(child)
    traverse(profile)
    return flops_dict

# 随便跑一次前向传播
sess.run(model.outputs, feed_dict=construct_feed_dict(features, support, y_train, train_mask, placeholders))

# 打印所有 scope 和 FLOPs
opts = tf.profiler.ProfileOptionBuilder.float_operation()
profile = tf.compat.v1.profiler.profile(sess.graph, options=opts, cmd='scope')
for node in profile.children:
    print(f"{node.name}: {node.total_float_ops}")

# Train model and collect FLOPs
print("Starting training and FLOPs profiling...")
for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # === Step 2: 第1层 GCN - 聚合邻居特征 ===
    # 特征变换（更新）
    run_meta = tf.RunMetadata()
    updated = sess.run([model.layers[0]._update(placeholders['features'])], feed_dict=feed_dict,
                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    profile1 = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='scope', options=opts)

    # 邻居信息聚合
    aggregated = sess.run([model.layers[0]._aggregate(updated[0])], feed_dict=feed_dict)

    # === Step 3: 第2层 GCN - 聚合+分类输出 ===
    # 特征变换（更新）
    run_meta2 = tf.RunMetadata()
    updated2 = sess.run([model.layers[1]._update(aggregated[0])], feed_dict=feed_dict,
                        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta2)
    profile2 = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta2, cmd='scope', options=opts)

    # 邻居信息聚合
    outputs = sess.run([model.layers[1]._aggregate(updated2[0])], feed_dict=feed_dict)

    # === Step 4: 计算当前 batch 的损失和准确率 ===
    loss_acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)

    # === Step 5: 执行反向传播，更新参数 ===
    sess.run([model.opt_op], feed_dict=feed_dict)

    # === Step 6: 验证集评估 ===
    # 可选：如需val loss/acc可加

    # === Step 7: 打印当前 epoch 结果 ===
    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(loss_acc[0]),
          "train_acc=", "{:.5f}".format(loss_acc[1]),
          "time=", "{:.5f}".format(time.time() - t))

    # 统计FLOPs
    prefixes = ['layer1_update', 'layer2_update']
    flops1 = collect_flops_by_prefix(profile1, prefixes)
    flops2 = collect_flops_by_prefix(profile2, prefixes)
    aggr_flops_layer1 = adj.nnz * features[2][1]
    aggr_flops_layer2 = adj.nnz * FLAGS.hidden1

    flops_epoch = {
        'layer1_update': flops1['layer1_update'],
        'layer1_aggregate': aggr_flops_layer1,
        'layer2_update': flops2['layer2_update'],
        'layer2_aggregate': aggr_flops_layer2,
    }
    with open(os.path.join(flops_dir, f'flops_epoch_{epoch+1}.json'), 'w') as f:
        json.dump(flops_epoch, f, indent=2)

print("FLOPs profiling completed. Results saved to:", flops_dir) 