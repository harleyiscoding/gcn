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
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace Eigen;

namespace {
// 检查路径是否存在（目录或文件）
static bool path_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// 检查是否为目录
static bool is_directory(const std::string& path) {
    struct stat buffer;
    if (stat(path.c_str(), &buffer) != 0) return false;
    return S_ISDIR(buffer.st_mode);
}

// 路径拼接辅助函数
static std::string join_path(const std::string& base, const std::string& part) {
    if (base.empty()) return part;
    if (base.back() == '/') return base + part;
    return base + "/" + part;
}

// 获取路径的文件名部分
static std::string get_filename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

std::vector<std::string> candidate_data_dirs(const std::string& data_dir) {
    std::vector<std::string> candidates = {
        join_path(data_dir, "processed"),
        data_dir,
        "./data/processed",
        "./data",
        "../data/processed",
        "../data",
        "../../data/processed",
        "../../data",
        "../../../data/processed",
        "../../../data"
    };
    
    // 尝试从环境变量获取基础路径
    const char* base_path = std::getenv("HLEXPERIENCE_ROOT");
    if (base_path) {
        std::string base(base_path);
        candidates.push_back(base + "/gcnex/gcn-master/gcn/data/processed");
        candidates.push_back(base + "/gcnex/gcn-master/gcn/data");
    }
    
    // Docker 和宿主机常见路径
    candidates.push_back("/workspace/hlexperience/gcnex/gcn-master/gcn/data/processed");
    candidates.push_back("/workspace/hlexperience/gcnex/gcn-master/gcn/data");
    candidates.push_back("/home/wanyu/hlexperience/gcnex/gcn-master/gcn/data/processed");
    candidates.push_back("/home/wanyu/hlexperience/gcnex/gcn-master/gcn/data");
    
    return candidates;
}
}

std::string DataLoader::resolve_data_dir(const std::string& data_dir) {
    std::cout << "[DataLoader] 查找数据目录，初始路径: " << data_dir << std::endl;
    for (const auto& dir : candidate_data_dirs(data_dir)) {
        if (path_exists(dir) && is_directory(dir)) {
            std::cout << "[DataLoader] 找到数据目录: " << dir << std::endl;
            return dir;
        }
    }
    std::cerr << "[DataLoader] 错误: 无法找到数据目录，尝试的路径:" << std::endl;
    for (const auto& dir : candidate_data_dirs(data_dir)) {
        std::cerr << "  - " << dir << (path_exists(dir) ? " (存在)" : " (不存在)") << std::endl;
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
    
    std::cout << "[DataLoader] 读取稀疏矩阵: " << rows << " x " << cols 
              << ", " << nnz << " 非零元素" << std::endl;
    
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(nnz);
    int r, c;
    float v;
    const int progress_interval = std::max(10000, nnz / 10);  // 每 10% 或每 10000 个元素输出一次
    int count = 0;
    while (fin >> r >> c >> v) {
        triplets.emplace_back(r, c, v);
        count++;
        if (count % progress_interval == 0 || count == nnz) {
            std::cout << "[DataLoader] 读取进度: " << count << " / " << nnz 
                      << " 元素 (" << (100 * count / nnz) << "%)" << std::endl;
            std::cout.flush();
        }
    }
    std::cout << "[DataLoader] 构建稀疏矩阵..." << std::endl;
    SparseMatrix<float> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();
    std::cout << "[DataLoader] 稀疏矩阵读取完成" << std::endl;
    return mat;
}

MatrixXf DataLoader::read_dense_matrix(const std::string& path, int rows, int cols) {
    std::ifstream fin(path);
    if (!fin.is_open()) throw std::runtime_error("Cannot open dense file: " + path);
    
    // 设置更大的缓冲区以提高 I/O 性能
    const size_t buffer_size = 1024 * 1024;  // 1MB 缓冲区
    char* buffer = new char[buffer_size];
    fin.rdbuf()->pubsetbuf(buffer, buffer_size);
    
    int file_rows, file_cols;
    if (!(fin >> file_rows >> file_cols)) {
        delete[] buffer;
        throw std::runtime_error("Dense header format error: " + path);
    }
    if (file_rows != rows || file_cols != cols) {
        delete[] buffer;
        throw std::runtime_error("Dense shape mismatch in " + path);
    }
    
    // 对于大文件，添加进度输出
    const int progress_interval = std::max(100, rows / 10);  // 每 10% 或每 100 行输出一次
    const size_t total_elements = static_cast<size_t>(rows) * cols;
    const size_t total_size_mb = (total_elements * sizeof(float)) / (1024 * 1024);
    std::cout << "[DataLoader] 读取密集矩阵: " << rows << " x " << cols 
              << " (约 " << total_size_mb << " MB, " << total_elements << " 个元素)" << std::endl;
    std::cout << "[DataLoader] 注意: 在 ZSim 仿真中，大文件读取会非常慢，请耐心等待..." << std::endl;
    std::cout.flush();
    
    MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(fin >> mat(i, j))) {
                delete[] buffer;
                throw std::runtime_error("Dense data format error: " + path);
            }
        }
        // 进度输出
        if ((i + 1) % progress_interval == 0 || i == rows - 1) {
            int percent = (100 * (i + 1)) / rows;
            std::cout << "[DataLoader] 读取进度: " << (i + 1) << " / " << rows 
                      << " 行 (" << percent << "%)" << std::endl;
            std::cout.flush();  // 强制刷新输出
        }
    }
    delete[] buffer;
    std::cout << "[DataLoader] 密集矩阵读取完成" << std::endl;
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

    std::cout << "[DataLoader] 加载数据集: " << dataset << std::endl;
    std::string base = resolve_data_dir(data_dir);
    // 如果 base 已经是 processed 目录则直接用，否则附加 processed
    std::string base_path = base;
    if (get_filename(base_path) != "processed") {
        base_path = join_path(base_path, "processed");
    }

    // 文件路径
    std::string meta_path     = join_path(base_path, dataset + "_meta.txt");
    std::string adj_path      = join_path(base_path, dataset + "_adj.coo");
    std::string feat_path     = join_path(base_path, dataset + "_features.txt");
    std::string labels_path   = join_path(base_path, dataset + "_labels.txt");
    std::string train_mask_p  = join_path(base_path, dataset + "_train_mask.txt");
    std::string val_mask_p    = join_path(base_path, dataset + "_val_mask.txt");
    std::string test_mask_p   = join_path(base_path, dataset + "_test_mask.txt");

    std::cout << "[DataLoader] 检查元数据文件: " << meta_path << std::endl;
    if (!path_exists(meta_path)) {
        throw std::runtime_error(
            "Preprocessed files not found. Run: python3 gcn/cpp/tools/convert_planetoid.py --dataset " + dataset);
    }

    int num_nodes = 0, num_features = 0, num_classes = 0;
    std::cout << "[DataLoader] 读取元数据..." << std::endl;
    read_meta(meta_path, num_nodes, num_features, num_classes);
    std::cout << "[DataLoader] 元数据: " << num_nodes << " 节点, " << num_features << " 特征, " << num_classes << " 类" << std::endl;

    data.num_nodes = num_nodes;
    data.num_features = num_features;
    data.num_classes = num_classes;

    std::cout << "[DataLoader] 读取邻接矩阵..." << std::endl;
    data.adj = read_sparse_coo(adj_path, num_nodes, num_nodes);
    std::cout << "[DataLoader] 读取特征矩阵..." << std::endl;
    data.features = read_dense_matrix(feat_path, num_nodes, num_features);
    std::cout << "[DataLoader] 读取标签..." << std::endl;
    data.labels = read_dense_matrix(labels_path, num_nodes, num_classes);
    std::cout << "[DataLoader] 读取掩码..." << std::endl;
    data.train_mask = read_mask(train_mask_p, num_nodes);
    data.val_mask = read_mask(val_mask_p, num_nodes);
    data.test_mask = read_mask(test_mask_p, num_nodes);
    std::cout << "[DataLoader] 数据加载完成" << std::endl;

    // 预处理特征（行归一化）与邻接（D^-1/2 A D^-1/2）在 graph_utils 处理
    return data;
}
