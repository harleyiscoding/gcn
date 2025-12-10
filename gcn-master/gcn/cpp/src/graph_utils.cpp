/*
 * 图处理工具实现
 */

#include "../include/graph_utils.h"
#include <algorithm>
#include <cmath>

SparseMatrix<float> GraphUtils::normalize_adj(const SparseMatrix<float>& adj) {
    int n = adj.rows();
    
    // 计算度矩阵 D
    VectorXf degrees = adj * VectorXf::Ones(n);
    
    // D^(-1/2)
    for (int i = 0; i < n; i++) {
        if (degrees(i) > 0) {
            degrees(i) = 1.0f / std::sqrt(degrees(i));
        }
    }
    
    // D^(-1/2) * A * D^(-1/2)
    // 使用三元组构建稀疏矩阵
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(adj.nonZeros());
    
    for (int k = 0; k < adj.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(adj, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            float val = it.value() * degrees(i) * degrees(j);
            triplets.push_back(Eigen::Triplet<float>(i, j, val));
        }
    }
    
    SparseMatrix<float> adj_norm(n, n);
    adj_norm.setFromTriplets(triplets.begin(), triplets.end());
    adj_norm.makeCompressed();
    
    return adj_norm;
}

MatrixXf GraphUtils::preprocess_features(const MatrixXf& features) {
    // 行归一化
    MatrixXf normalized = features;
    
    for (int i = 0; i < normalized.rows(); i++) {
        float rowsum = normalized.row(i).sum();
        if (rowsum > 0) {
            normalized.row(i) /= rowsum;
        }
    }
    
    return normalized;
}

GraphData::SparseTuple GraphUtils::sparse_to_tuple(const SparseMatrix<float>& sparse_mx) {
    GraphData::SparseTuple tuple;
    
    int nnz = sparse_mx.nonZeros();
    tuple.coords = MatrixXi(nnz, 2);
    tuple.values = VectorXf(nnz);
    
    int idx = 0;
    for (int k = 0; k < sparse_mx.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(sparse_mx, k); it; ++it) {
            tuple.coords(idx, 0) = it.row();
            tuple.coords(idx, 1) = it.col();
            tuple.values(idx) = it.value();
            idx++;
        }
    }
    
    tuple.shape = std::make_pair(sparse_mx.rows(), sparse_mx.cols());
    return tuple;
}

SparseMatrix<float> GraphUtils::tuple_to_sparse(const GraphData::SparseTuple& tuple) {
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(tuple.coords.rows());
    
    for (int i = 0; i < tuple.coords.rows(); i++) {
        triplets.push_back(Eigen::Triplet<float>(
            tuple.coords(i, 0),
            tuple.coords(i, 1),
            tuple.values(i)
        ));
    }
    
    SparseMatrix<float> sparse(tuple.shape.first, tuple.shape.second);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();
    
    return sparse;
}

SparseMatrix<float> GraphUtils::add_self_loops(const SparseMatrix<float>& adj) {
    int n = adj.rows();
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(adj.nonZeros() + n);
    
    // 添加原有边
    for (int k = 0; k < adj.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(adj, k); it; ++it) {
            triplets.push_back(Eigen::Triplet<float>(it.row(), it.col(), it.value()));
        }
    }
    
    // 添加自环
    for (int i = 0; i < n; i++) {
        triplets.push_back(Eigen::Triplet<float>(i, i, 1.0f));
    }
    
    SparseMatrix<float> adj_with_loops(n, n);
    adj_with_loops.setFromTriplets(triplets.begin(), triplets.end());
    adj_with_loops.makeCompressed();
    
    return adj_with_loops;
}

std::vector<std::vector<int>> PartitionUtils::adj_to_metis(const SparseMatrix<float>& adj) {
    int n = adj.rows();
    std::vector<std::vector<int>> neighbors(n);
    
    for (int k = 0; k < adj.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(adj, k); it; ++it) {
            neighbors[it.row()].push_back(it.col());
        }
    }
    
    return neighbors;
}

std::vector<int> PartitionUtils::metis_partition(const SparseMatrix<float>& adj, int num_parts) {
    // TODO: 调用 METIS C 库进行分区
    // 这里先返回简单的划分（按顺序）
    int n = adj.rows();
    std::vector<int> part_labels(n);
    
    for (int i = 0; i < n; i++) {
        part_labels[i] = i % num_parts;
    }
    
    return part_labels;
}

std::vector<std::vector<int>> PartitionUtils::get_partition_masks(
    const std::vector<int>& part_labels, int num_partitions) {
    
    std::vector<std::vector<int>> masks(num_partitions);
    
    for (size_t i = 0; i < part_labels.size(); i++) {
        masks[part_labels[i]].push_back(i);
    }
    
    return masks;
}

SubgraphData PartitionUtils::extract_subgraph(
    const GraphData& graph_data,
    const std::vector<int>& part_nodes) {
    
    SubgraphData subgraph;
    int num_sub_nodes = part_nodes.size();
    
    // 提取子图邻接矩阵
    std::vector<Eigen::Triplet<float>> triplets;
    
    // 创建节点映射：全局索引 -> 子图索引
    std::vector<int> global_to_local(graph_data.num_nodes, -1);
    for (size_t i = 0; i < part_nodes.size(); i++) {
        global_to_local[part_nodes[i]] = i;
    }
    
    // 提取边
    for (int k = 0; k < graph_data.adj.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(graph_data.adj, k); it; ++it) {
            int global_i = it.row();
            int global_j = it.col();
            
            int local_i = global_to_local[global_i];
            int local_j = global_to_local[global_j];
            
            if (local_i >= 0 && local_j >= 0) {
                triplets.push_back(Eigen::Triplet<float>(local_i, local_j, it.value()));
            }
        }
    }
    
    subgraph.adj_sub = SparseMatrix<float>(num_sub_nodes, num_sub_nodes);
    subgraph.adj_sub.setFromTriplets(triplets.begin(), triplets.end());
    subgraph.adj_sub.makeCompressed();
    
    // 提取特征
    subgraph.features_sub = MatrixXf(num_sub_nodes, graph_data.num_features);
    for (size_t i = 0; i < part_nodes.size(); i++) {
        subgraph.features_sub.row(i) = graph_data.features.row(part_nodes[i]);
    }
    
    // 提取标签和掩码
    subgraph.y_train_sub = MatrixXf(num_sub_nodes, graph_data.num_classes);
    subgraph.y_val_sub = MatrixXf(num_sub_nodes, graph_data.num_classes);
    subgraph.y_test_sub = MatrixXf(num_sub_nodes, graph_data.num_classes);
    subgraph.train_mask_sub = VectorXi(num_sub_nodes);
    subgraph.val_mask_sub = VectorXi(num_sub_nodes);
    subgraph.test_mask_sub = VectorXi(num_sub_nodes);
    
    for (size_t i = 0; i < part_nodes.size(); i++) {
        int global_idx = part_nodes[i];
        subgraph.y_train_sub.row(i) = graph_data.labels.row(global_idx);
        subgraph.y_val_sub.row(i) = graph_data.labels.row(global_idx);
        subgraph.y_test_sub.row(i) = graph_data.labels.row(global_idx);
        subgraph.train_mask_sub(i) = graph_data.train_mask(global_idx);
        subgraph.val_mask_sub(i) = graph_data.val_mask(global_idx);
        subgraph.test_mask_sub(i) = graph_data.test_mask(global_idx);
    }
    
    // 保存全局节点索引
    subgraph.part_nodes = VectorXi(num_sub_nodes);
    for (size_t i = 0; i < part_nodes.size(); i++) {
        subgraph.part_nodes(i) = part_nodes[i];
    }
    
    return subgraph;
}

