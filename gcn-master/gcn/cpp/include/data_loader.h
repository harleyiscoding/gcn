/*
 * 数据加载模块
 * 从预处理后的文本文件加载图数据（Cora/Citeseer/Pubmed）
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "types.h"
#include <string>
#include <memory>
#include <vector>

// ==================== 数据加载器 ====================

class DataLoader {
public:
    // 加载数据集
    // dataset: "cora", "citeseer", "pubmed"
    // data_dir: 数据目录路径（默认 gcn/data）
    static GraphData load_data(const std::string& dataset, 
                               const std::string& data_dir = "data");
    
private:
    // 寻找可用的数据目录（支持 ./data, ../data, ../../data 等）
    static std::string resolve_data_dir(const std::string& data_dir);

    // 读取元信息
    static void read_meta(const std::string& path, int& num_nodes, int& num_features, int& num_classes);

    // 读取稀疏矩阵（COO：首行 rows cols nnz，后续行 row col value）
    static SparseMatrix<float> read_sparse_coo(const std::string& path, int rows, int cols);

    // 读取稠密矩阵（首行 rows cols，后续每行空格分隔）
    static MatrixXf read_dense_matrix(const std::string& path, int rows, int cols);

    // 读取掩码（首行 count，后续为索引列表）
    static VectorXi read_mask(const std::string& path, int length);
};

#endif // DATA_LOADER_H

