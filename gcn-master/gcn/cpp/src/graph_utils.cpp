/*
 * 图处理工具实现
 */

#include "../include/graph_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

#ifdef HAVE_METIS
#include <metis.h>
#endif

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
    int n = adj.rows();
    std::vector<int> part_labels(n);
    
    // 如果只有一个分区，直接返回
    if (num_parts <= 1) {
        std::fill(part_labels.begin(), part_labels.end(), 0);
        return part_labels;
    }
    
#ifdef HAVE_METIS
    // 使用真实的 METIS 库进行分区
    // 将 Eigen 稀疏矩阵转换为 METIS 需要的 CSR 格式
    // METIS 需要：xadj (索引指针数组) 和 adjncy (邻接节点数组)
    std::vector<idx_t> xadj(n + 1);
    std::vector<idx_t> adjncy;
    
    xadj[0] = 0;
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (SparseMatrix<float>::InnerIterator it(adj, i); it; ++it) {
            // 只添加非零边（METIS 不需要自环，但可以包含）
            if (it.value() != 0.0f) {
                adjncy.push_back(static_cast<idx_t>(it.col()));
                count++;
            }
        }
        xadj[i + 1] = xadj[i] + count;
    }
    
    // METIS 参数
    idx_t nvtxs = static_cast<idx_t>(n);           // 节点数
    idx_t ncon = 1;                                // 约束数（通常为 1）
    idx_t* xadj_ptr = xadj.data();                 // CSR 索引指针
    idx_t* adjncy_ptr = adjncy.data();             // CSR 邻接节点
    idx_t* vwgt = nullptr;                          // 节点权重（可选）
    idx_t* vsize = nullptr;                         // 节点大小（可选）
    idx_t* adjwgt = nullptr;                        // 边权重（可选）
    idx_t nparts = static_cast<idx_t>(num_parts);   // 分区数
    real_t* tpwgts = nullptr;                       // 目标分区权重（可选）
    real_t* ubvec = nullptr;                        // 不平衡容忍度（可选）
    idx_t options[METIS_NOPTIONS];                  // 选项数组
    METIS_SetDefaultOptions(options);               // 设置默认选项
    // 设置分区质量选项
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // 最小化边割（edge cuts）
    options[METIS_OPTION_NCUTS] = 1;                     // 尝试次数（可以增加以获得更好结果）
    options[METIS_OPTION_NITER] = 10;                    // 迭代次数（默认 10）
    options[METIS_OPTION_UFACTOR] = 1;                   // 不平衡因子（1% 不平衡容忍度）
    idx_t objval;                                   // 输出：分区质量指标
    idx_t* part = new idx_t[n];                     // 输出：分区标签
    
    // 调用 METIS_PartGraphKway 进行 K-way 分区
    int ret = METIS_PartGraphKway(
        &nvtxs,      // 节点数
        &ncon,       // 约束数
        xadj_ptr,    // CSR 索引指针
        adjncy_ptr,  // CSR 邻接节点
        vwgt,        // 节点权重
        vsize,       // 节点大小
        adjwgt,      // 边权重
        &nparts,     // 分区数
        tpwgts,      // 目标分区权重
        ubvec,       // 不平衡容忍度
        options,     // 选项数组
        &objval,     // 输出：分区质量
        part         // 输出：分区标签
    );
    
    if (ret == METIS_OK) {
        // 转换结果
        for (int i = 0; i < n; i++) {
            part_labels[i] = static_cast<int>(part[i]);
        }
        
        // 验证分区质量：计算实际的边割数
        int actual_edge_cuts = 0;
        for (int i = 0; i < n; i++) {
            for (SparseMatrix<float>::InnerIterator it(adj, i); it; ++it) {
                int j = it.col();
                if (part_labels[i] != part_labels[j]) {
                    actual_edge_cuts++;
                }
            }
        }
        // 每条边被计算两次（i->j 和 j->i），所以除以 2
        actual_edge_cuts /= 2;
        
        // 计算每个分区的大小
        std::vector<int> part_sizes(num_parts, 0);
        for (int i = 0; i < n; i++) {
            part_sizes[part_labels[i]]++;
        }
        
        std::cout << "[METIS] Graph partitioned into " << num_parts 
                  << " parts (objval=" << objval << ", actual_edge_cuts=" 
                  << actual_edge_cuts << ")" << std::endl;
        std::cout << "[METIS] Partition sizes: ";
        for (int p = 0; p < num_parts; p++) {
            std::cout << "P" << p << "=" << part_sizes[p];
            if (p < num_parts - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
    } else {
        // METIS 失败，回退到简单分区
        std::cerr << "[METIS] Partition failed (ret=" << ret 
                  << "), using simple partition" << std::endl;
        for (int i = 0; i < n; i++) {
            part_labels[i] = i % num_parts;
        }
    }
    
    delete[] part;
    
#else
    // 如果没有 METIS 库，使用简单分区（按顺序）
    std::cout << "[METIS] METIS library not available, using simple partition" << std::endl;
    for (int i = 0; i < n; i++) {
        part_labels[i] = i % num_parts;
    }
#endif
    
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

