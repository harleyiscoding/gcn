/*
 * 数据加载实现
 * 流程与 Python 版 utils.load_data 对齐，但采用预处理后的文本格式
 * 需要先运行 python 脚本将 Planetoid 原始二进制转换为文本：
 *   python3 gcn/cpp/tools/convert_planetoid.py --dataset cora
 */

#include "../include/data_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;

namespace {
std::vector<std::string> candidate_data_dirs(const std::string& data_dir) {
    return {
        data_dir + "/processed",
        data_dir,
        "./data/processed",
        "./data",
        "../data/processed",
        "../data",
        "../../data/processed",
        "../../data"
    };
}
}

std::string DataLoader::resolve_data_dir(const std::string& data_dir) {
    for (const auto& dir : candidate_data_dirs(data_dir)) {
        if (fs::exists(fs::path(dir))) {
            return dir;
        }
    }
    throw std::runtime_error("Cannot locate data directory. Tried variants of: " + data_dir);
}

void DataLoader::read_meta(const std::string& path, int& num_nodes, int& num_features, int& num_classes) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open meta file: " + path);
    }
    fin >> num_nodes >> num_features >> num_classes;
    if (!fin) throw std::runtime_error("Meta file format error: " + path);
}

SparseMatrix<float> DataLoader::read_sparse_coo(const std::string& path, int rows, int cols) {
    std::ifstream fin(path);
    if (!fin.is_open()) throw std::runtime_error("Cannot open sparse file: " + path);
    int file_rows, file_cols, nnz;
    if (!(fin >> file_rows >> file_cols >> nnz)) {
        throw std::runtime_error("Sparse header format error: " + path);
    }
    if (file_rows != rows || file_cols != cols) {
        throw std::runtime_error("Sparse shape mismatch in " + path);
    }
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(nnz);
    int r, c;
    float v;
    while (fin >> r >> c >> v) {
        triplets.emplace_back(r, c, v);
    }
    SparseMatrix<float> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();
    return mat;
}

MatrixXf DataLoader::read_dense_matrix(const std::string& path, int rows, int cols) {
    std::ifstream fin(path);
    if (!fin.is_open()) throw std::runtime_error("Cannot open dense file: " + path);
    int file_rows, file_cols;
    if (!(fin >> file_rows >> file_cols)) {
        throw std::runtime_error("Dense header format error: " + path);
    }
    if (file_rows != rows || file_cols != cols) {
        throw std::runtime_error("Dense shape mismatch in " + path);
    }
    MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(fin >> mat(i, j))) {
                throw std::runtime_error("Dense data format error: " + path);
            }
        }
    }
    return mat;
}

VectorXi DataLoader::read_mask(const std::string& path, int length) {
    std::ifstream fin(path);
    if (!fin.is_open()) throw std::runtime_error("Cannot open mask file: " + path);
    int count;
    if (!(fin >> count)) throw std::runtime_error("Mask header format error: " + path);
    VectorXi mask = VectorXi::Zero(length);
    for (int i = 0; i < count; ++i) {
        int idx;
        if (!(fin >> idx)) throw std::runtime_error("Mask data format error: " + path);
        if (idx >= 0 && idx < length) mask(idx) = 1;
    }
    return mask;
}

GraphData DataLoader::load_data(const std::string& dataset, const std::string& data_dir) {
    GraphData data;

    std::string base = resolve_data_dir(data_dir);
    // 如果 base 已经是 processed 目录则直接用，否则附加 processed
    fs::path base_path(base);
    if (base_path.filename() != "processed") {
        base_path /= "processed";
    }

    // 文件路径
    fs::path meta_path     = base_path / (dataset + "_meta.txt");
    fs::path adj_path      = base_path / (dataset + "_adj.coo");
    fs::path feat_path     = base_path / (dataset + "_features.txt");
    fs::path labels_path   = base_path / (dataset + "_labels.txt");
    fs::path train_mask_p  = base_path / (dataset + "_train_mask.txt");
    fs::path val_mask_p    = base_path / (dataset + "_val_mask.txt");
    fs::path test_mask_p   = base_path / (dataset + "_test_mask.txt");

    if (!fs::exists(meta_path)) {
        throw std::runtime_error(
            "Preprocessed files not found. Run: python3 gcn/cpp/tools/convert_planetoid.py --dataset " + dataset);
    }

    int num_nodes = 0, num_features = 0, num_classes = 0;
    read_meta(meta_path.string(), num_nodes, num_features, num_classes);

    data.num_nodes = num_nodes;
    data.num_features = num_features;
    data.num_classes = num_classes;

    data.adj = read_sparse_coo(adj_path.string(), num_nodes, num_nodes);
    data.features = read_dense_matrix(feat_path.string(), num_nodes, num_features);
    data.labels = read_dense_matrix(labels_path.string(), num_nodes, num_classes);
    data.train_mask = read_mask(train_mask_p.string(), num_nodes);
    data.val_mask = read_mask(val_mask_p.string(), num_nodes);
    data.test_mask = read_mask(test_mask_p.string(), num_nodes);

    // 预处理特征（行归一化）与邻接（D^-1/2 A D^-1/2）在 graph_utils 处理
    return data;
}
