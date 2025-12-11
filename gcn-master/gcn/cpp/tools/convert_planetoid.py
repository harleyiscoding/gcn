#!/usr/bin/env python3
# Convert Planetoid binary data (cora/citeseer/pubmed) to simple text format
# Format is aligned with cpp DataLoader:
#   *_meta.txt:        num_nodes num_features num_classes
#   *_adj.coo:         rows cols nnz \n row col val ...
#   *_features.txt:    rows cols \n feature_row ...
#   *_labels.txt:      rows cols \n onehot_row ...
#   *_train_mask.txt:  count idx1 idx2 ...
#   *_val_mask.txt:    count idx1 idx2 ...
#   *_test_mask.txt:   count idx1 idx2 ...

import argparse
import os
import numpy as np
import scipy.sparse as sp


def sample_mask(idx, length):
    mask = np.zeros(length, dtype=np.int64)
    mask[idx] = 1
    return mask


def parse_index_file(filename):
    with open(filename) as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_planetoid(dataset_str, data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open(os.path.join(data_dir, f"ind.{dataset_str}.{name}"), 'rb') as f:
            obj = np.load(f, allow_pickle=True, encoding='latin1')
            objects.append(obj)
    x, y, tx, ty, allx, ally, graph = objects
    # graph may already be a dict/defaultdict after numpy load; only call .item() if needed
    if not isinstance(graph, dict):
        graph = graph.item()

    test_idx_reorder = parse_index_file(os.path.join(data_dir, f"ind.{dataset_str}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer test set (as in Kipf code)
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # Build adjacency
    adj = sp.coo_matrix(([], ([], [])), shape=(labels.shape[0], labels.shape[0]))
    row = []
    col = []
    data = []
    for i, neighs in graph.items():
        for j in neighs:
            row.append(i)
            col.append(j)
            data.append(1.0)
    adj = sp.coo_matrix((data, (row, col)), shape=(labels.shape[0], labels.shape[0]))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # symmetrize

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = train_mask[:, None] * labels
    y_val = val_mask[:, None] * labels
    y_test = test_mask[:, None] * labels

    return adj.tocsr(), features.tocsr(), y_train, y_val, y_test, train_mask, val_mask, test_mask


def save_meta(path, num_nodes, num_features, num_classes):
    with open(path, 'w') as f:
        f.write(f"{num_nodes} {num_features} {num_classes}\n")


def save_adj(path, adj):
    coo = adj.tocoo()
    with open(path, 'w') as f:
        f.write(f"{coo.shape[0]} {coo.shape[1]} {coo.nnz}\n")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            f.write(f"{int(r)} {int(c)} {float(v)}\n")


def save_dense(path, mat):
    arr = np.asarray(mat, dtype=np.float32)
    rows, cols = arr.shape
    with open(path, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for i in range(rows):
            row = arr[i].ravel()
            f.write(" ".join(str(float(x)) for x in row) + "\n")


def save_mask(path, mask):
    idx = np.nonzero(mask)[0].tolist()
    with open(path, 'w') as f:
        f.write(str(len(idx)))
        for i in idx:
            f.write(f" {i}")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--data_dir", default="gcn/data", help="Path to raw Planetoid data (ind.* files)")
    parser.add_argument("--out_dir", default="gcn/data/processed", help="Output directory for processed text")
    args = parser.parse_args()

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_planetoid(
        args.dataset, args.data_dir
    )

    os.makedirs(args.out_dir, exist_ok=True)

    labels = y_train + y_val + y_test  # combine masks
    num_nodes, num_features = features.shape
    num_classes = labels.shape[1]

    save_meta(os.path.join(args.out_dir, f"{args.dataset}_meta.txt"), num_nodes, num_features, num_classes)
    save_adj(os.path.join(args.out_dir, f"{args.dataset}_adj.coo"), adj)
    save_dense(os.path.join(args.out_dir, f"{args.dataset}_features.txt"), features.todense())
    save_dense(os.path.join(args.out_dir, f"{args.dataset}_labels.txt"), labels)
    save_mask(os.path.join(args.out_dir, f"{args.dataset}_train_mask.txt"), train_mask)
    save_mask(os.path.join(args.out_dir, f"{args.dataset}_val_mask.txt"), val_mask)
    save_mask(os.path.join(args.out_dir, f"{args.dataset}_test_mask.txt"), test_mask)

    print(f"Saved processed data to {args.out_dir}")


if __name__ == "__main__":
    main()

