/*
 * 数据加载模块
 * 从 pickle 文件加载图数据（Cora/Citeseer/Pubmed）
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "types.h"
#include <string>
#include <memory>

// ==================== 数据加载器 ====================

class DataLoader {
public:
    // 加载数据集
    // dataset: "cora", "citeseer", "pubmed"
    // data_dir: 数据目录路径
    static GraphData load_data(const std::string& dataset, 
                               const std::string& data_dir = "data");
    
private:
    // 从 pickle 文件加载（需要实现 pickle 解析或使用 Python 脚本预处理）
    // 或者直接从文本格式加载
    static void load_from_pickle(const std::string& filepath, GraphData& data);
    
    // 解析索引文件
    static std::vector<int> parse_index_file(const std::string& filepath);
    
    // 创建掩码
    static VectorXi sample_mask(const std::vector<int>& indices, int length);
};

#endif // DATA_LOADER_H

