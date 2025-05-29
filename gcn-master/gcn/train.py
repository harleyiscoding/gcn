from __future__ import division
from __future__ import print_function

import time
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler import profiler
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

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
flags.DEFINE_string('profile_dir', './profiles', 'Directory to save profiling data.')
flags.DEFINE_string('results_dir', './results', 'Directory to save results.')
flags.DEFINE_string('json_base_dir', './results/json', 'Base directory for JSON results.')
flags.DEFINE_string('profile_reports_dir', './results/profile_reports', 'Directory to save profile reports.')

# Generate a timestamp for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Create directories if they don't exist
json_run_dir = os.path.join(FLAGS.json_base_dir, timestamp)
profile_reports_run_dir = os.path.join(FLAGS.profile_reports_dir, timestamp)

for directory in [FLAGS.profile_dir, FLAGS.results_dir, FLAGS.json_base_dir, FLAGS.profile_reports_dir, json_run_dir, profile_reports_run_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 定义性能统计阶段
class PerfStage:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def start(self):
        self.start_time = time.time()
        # 触发perf记录开始
        os.system(f'echo "PERF_STAGE_START {self.name}" > /dev/null')

    def end(self):
        self.end_time = time.time()
        # 触发perf记录结束
        os.system(f'echo "PERF_STAGE_END {self.name}" > /dev/null')
        self.metrics['duration'] = self.end_time - self.start_time

# Load data
print("Loading data...")
data_load_stage = PerfStage('data_loading')
data_load_stage.start()
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
data_load_stage.end()
print(f"Data loading time: {data_load_stage.metrics['duration']:.2f} seconds")

# Some preprocessing
print("Preprocessing data...")
preprocess_stage = PerfStage('preprocessing')
preprocess_stage.start()
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
preprocess_stage.end()
print(f"Preprocessing time: {preprocess_stage.metrics['duration']:.2f} seconds")

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
model_build_stage = PerfStage('model_building')
model_build_stage.start()
model = model_func(placeholders, input_dim=features[2][1], logging=True)
model_build_stage.end()
print(f"Model building time: {model_build_stage.metrics['duration']:.2f} seconds")

# Initialize session
sess = tf.Session()

# Initialize profiler
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    eval_stage = PerfStage('evaluation')
    eval_stage.start()
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], 
                       feed_dict=feed_dict_val,
                       options=run_options,
                       run_metadata=run_metadata)
    eval_stage.end()
    return outs_val[0], outs_val[1], eval_stage.metrics['duration']

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Training statistics
train_times = []
val_times = []
total_times = []
epoch_stats = []

# Train model
print("Starting training...")
for epoch in range(FLAGS.epochs):
    epoch_stage = PerfStage(f'epoch_{epoch}')
    epoch_stage.start()
    
    # Training step
    train_stage = PerfStage(f'epoch_{epoch}_training')
    train_stage.start()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([model.opt_op, model.loss, model.accuracy], 
                   feed_dict=feed_dict,
                   options=run_options,
                   run_metadata=run_metadata)
    train_stage.end()
    train_times.append(train_stage.metrics['duration'])

    # Validation
    val_stage = PerfStage(f'epoch_{epoch}_validation')
    val_stage.start()
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    val_stage.end()
    val_times.append(val_stage.metrics['duration'])
    
    cost_val.append(cost)
    epoch_stage.end()
    total_times.append(epoch_stage.metrics['duration'])

    # Store epoch statistics
    epoch_stats.append({
        'epoch': epoch + 1,
        'train_loss': float(outs[1]),
        'train_acc': float(outs[2]),
        'val_loss': float(cost),
        'val_acc': float(acc),
        'train_time': float(train_stage.metrics['duration']),
        'val_time': float(val_stage.metrics['duration']),
        'total_time': float(epoch_stage.metrics['duration'])
    })

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), 
          "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), 
          "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), 
          "train_time=", "{:.5f}".format(train_stage.metrics['duration']),
          "val_time=", "{:.5f}".format(val_stage.metrics['duration']),
          "total_time=", "{:.5f}".format(epoch_stage.metrics['duration']))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

# Testing
print("\nRunning final test...")
test_stage = PerfStage('final_test')
test_stage.start()
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
test_stage.end()
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_stage.metrics['duration']))

# Generate profiling report
print("\nGenerating profiling report...")
opts = option_builder.ProfileOptionBuilder.time_and_memory()
opts['min_bytes'] = 0
opts['min_micros'] = 0
opts['output'] = f'file:outfile={os.path.join(profile_reports_run_dir, "profile_report.txt")}'
opts['select'] = ['float_ops', 'occurrence', 'device', 'op_types']

profiler.profile(
    tf.get_default_graph(),
    run_meta=run_metadata,
    cmd='op',
    options=opts
)

# Prepare results dictionary
results = {
    'model_config': {
        'dataset': FLAGS.dataset,
        'model': FLAGS.model,
        'learning_rate': FLAGS.learning_rate,
        'epochs': FLAGS.epochs,
        'hidden1': FLAGS.hidden1,
        'dropout': FLAGS.dropout,
        'weight_decay': FLAGS.weight_decay,
        'early_stopping': FLAGS.early_stopping,
        'max_degree': FLAGS.max_degree
    },
    'timing_stats': {
        'data_loading_time': float(data_load_stage.metrics['duration']),
        'preprocessing_time': float(preprocess_stage.metrics['duration']),
        'model_building_time': float(model_build_stage.metrics['duration']),
        'average_training_time': float(np.mean(train_times)),
        'average_validation_time': float(np.mean(val_times)),
        'average_epoch_time': float(np.mean(total_times)),
        'total_training_time': float(np.sum(total_times)),
        'test_time': float(test_stage.metrics['duration'])
    },
    'performance_metrics': {
        'final_test_loss': float(test_cost),
        'final_test_accuracy': float(test_acc)
    },
    'epoch_details': epoch_stats
}

# Save results to JSON file
results_file = os.path.join(json_run_dir, f'results_{FLAGS.model}_{FLAGS.dataset}.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {results_file}")
print(f"Profiling data saved to {FLAGS.profile_dir}")
print(f"Profile report saved to {os.path.join(profile_reports_run_dir, 'profile_report.txt')}")