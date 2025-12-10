/*
 * GCN C++ 主程序
 * 训练逻辑与 Python 版本完全一致
 */

#include "../include/types.h"
#include "../include/trainer.h"
#include <iostream>
#include <cstring>
#include <omp.h>

// 解析命令行参数
GCNConfig parse_args(int argc, char* argv[]) {
    GCNConfig config;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            config.dataset = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning_rate") == 0 && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--hidden1") == 0 && i + 1 < argc) {
            config.hidden1 = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--dropout") == 0 && i + 1 < argc) {
            config.dropout = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--weight_decay") == 0 && i + 1 < argc) {
            config.weight_decay = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--early_stopping") == 0 && i + 1 < argc) {
            config.early_stopping = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--num_parts") == 0 && i + 1 < argc) {
            config.num_parts = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model = argv[++i];
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    // 解析配置
    GCNConfig config = parse_args(argc, argv);
    
    // 设置 OpenMP 线程数
    omp_set_num_threads(4);
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
    
    // 打印配置
    std::cout << "=== GCN C++ Training ===" << std::endl;
    std::cout << "Dataset: " << config.dataset << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "Hidden dim: " << config.hidden1 << std::endl;
    std::cout << "Num partitions: " << config.num_parts << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << std::endl;
    
    // 创建训练器并训练
    Trainer trainer(config);
    
    try {
        trainer.initialize();
        trainer.train();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

