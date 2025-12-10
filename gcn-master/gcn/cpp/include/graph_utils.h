/*
 * 图处理工具
 * 图预处理、归一化、分区等操作
 */

#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include "types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// ==================== 图预处理 ====================

class GraphUtils {
public:
    // 归一化邻接矩阵：D^(-1/2) * A * D^(-1/2)
    static SparseMatrix<float> normalize_adj(const SparseMatrix<float>& adj);
    
    // 预处理特征：行归一化
    static MatrixXf preprocess_features(const MatrixXf& features);
    
    // 将稀疏矩阵转换为三元组表示
    static GraphData::SparseTuple sparse_to_tuple(const SparseMatrix<float>& sparse_mx);
    
    // 从三元组创建稀疏矩阵
    static SparseMatrix<float> tuple_to_sparse(const GraphData::SparseTuple& tuple);
    
    // 添加自环：A = A + I
    static SparseMatrix<float> add_self_loops(const SparseMatrix<float>& adj);
};

// ==================== 图分区工具 ====================

class PartitionUtils {
public:
    // 使用 METIS 进行图分区
    // 返回：每个节点的分区标签
    static std::vector<int> metis_partition(const SparseMatrix<float>& adj, int num_parts);
    
    // 获取分区掩码
    static std::vector<std::vector<int>> get_partition_masks(
        const std::vector<int>& part_labels, int num_partitions);
    
    // 提取子图
    static SubgraphData extract_subgraph(
        const GraphData& graph_data,
        const std::vector<int>& part_nodes);
    
    // 将邻接矩阵转换为 METIS 格式
    static std::vector<std::vector<int>> adj_to_metis(const SparseMatrix<float>& adj);
};

#endif // GRAPH_UTILS_H

