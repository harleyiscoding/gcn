/*
 * 训练器实现
 * 实现完整的训练循环，与 Python 版本逻辑完全一致
 */

#include "../include/trainer.h"
#include "zsim_hooks.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

using namespace Eigen;
using namespace std;

// ==================== Trainer 实现 ====================

// 检查文件是否存在的辅助函数（兼容旧版 GCC）
static bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static std::string resolve_memory_flops_path(const std::string& dataset) {
    std::vector<std::string> candidates = {
        "results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../../results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../../../results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../../gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt",
        "../../../gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt"
    };
    
    // 尝试从环境变量获取基础路径
    const char* base_path = std::getenv("HLEXPERIENCE_ROOT");
    if (base_path) {
        std::string base(base_path);
        candidates.push_back(base + "/gcnex/gcn-master/gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt");
    }
    
    // Docker 和宿主机常见路径
    candidates.push_back("/workspace/hlexperience/gcnex/gcn-master/gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt");
    candidates.push_back("/home/wanyu/hlexperience/gcnex/gcn-master/gcn/results/" + dataset + "/l1_cache_analysis/memory_flops_epochs.txt");
    
    std::cout << "[TaskInfo] 查找任务信息文件: " << dataset << std::endl;
    for (const auto& p : candidates) {
        if (file_exists(p)) {
            std::cout << "[TaskInfo] 找到文件: " << p << std::endl;
            return p;
        }
    }
    std::cerr << "[TaskInfo] 警告: 未找到任务信息文件，尝试的路径:" << std::endl;
    for (const auto& p : candidates) {
        std::cerr << "  - " << p << (file_exists(p) ? " (存在)" : " (不存在)") << std::endl;
    }
    return "";
}

void Trainer::load_tasks_info(const std::string& dataset) {
    std::string memory_flops_path = resolve_memory_flops_path(dataset);
    if (memory_flops_path.empty()) {
        std::cerr << "[TaskInfo] 警告: memory_flops_epochs.txt not found for dataset " << dataset
                  << ". Using default task value 1.0 (all PNM)." << std::endl;
        return;
    }

    std::cout << "[TaskInfo] 读取任务信息文件: " << memory_flops_path << std::endl;
    std::ifstream file(memory_flops_path);
    if (!file.is_open()) {
        std::cerr << "[TaskInfo] 警告: Cannot open " << memory_flops_path << std::endl;
        return;
    }
    
    std::string line;
    std::getline(file, line);  // 跳过表头
    std::cout << "[TaskInfo] 表头: " << line << std::endl;
    
    int loaded_count = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> parts;
        
        while (std::getline(iss, token, '\t')) {
            parts.push_back(token);
        }
        
        if (parts.size() < 5) continue;
        
        try {
            float l1_agg_mem = std::stof(parts[1]);
            float l2_agg_mem = std::stof(parts[2]);
            float l1_update_flops = std::stof(parts[3]);
            float l2_update_flops = std::stof(parts[4]);
            
            // 计算 AMIR 和 CD
            float amir1 = (l1_update_flops > 0) ? l1_agg_mem / l1_update_flops : 1.0f;
            float cd1 = (l1_agg_mem > 0) ? l1_update_flops / l1_agg_mem : 1.0f;
            float amir2 = (l2_update_flops > 0) ? l2_agg_mem / l2_update_flops : 1.0f;
            float cd2 = (l2_agg_mem > 0) ? l2_update_flops / l2_agg_mem : 1.0f;
            
            // 按照 Python 版本的顺序添加任务
            tasks_info.push_back({1, "UPDATE", cd1});
            tasks_info.push_back({1, "AGG", amir1});
            tasks_info.push_back({2, "UPDATE", cd2});
            tasks_info.push_back({2, "AGG", amir2});
            loaded_count++;
            
            std::cout << "[TaskInfo] Epoch " << loaded_count << ": "
                      << "L1_UPDATE(CD=" << cd1 << "), L1_AGG(AMIR=" << amir1 << "), "
                      << "L2_UPDATE(CD=" << cd2 << "), L2_AGG(AMIR=" << amir2 << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[TaskInfo] 解析错误: " << e.what() << " (行: " << line << ")" << std::endl;
            continue;
        }
    }
    
    std::cout << "[TaskInfo] 成功加载 " << loaded_count << " 个 epoch 的任务信息，共 " 
              << tasks_info.size() << " 个任务" << std::endl;
}

void Trainer::initialize() {
    // 1. 加载数据
    std::cout << "[初始化] 开始加载数据..." << std::endl;
    std::string data_dir = "../data";  // 初始路径，DataLoader 会尝试多个候选路径
    graph_data = DataLoader::load_data(config.dataset, data_dir);
    
    std::cout << "[初始化] 数据加载完成: " << graph_data.num_nodes << " 节点, " 
              << graph_data.num_features << " 特征, " << graph_data.num_classes << " 类" << std::endl;
    
    if (graph_data.num_nodes == 0) {
        throw std::runtime_error("Failed to load data. Please implement data loading or use preprocessed data.");
    }
    
    // 2. 图分区
    std::cout << "[初始化] 开始图分区 (METIS)..." << std::endl;
    std::vector<int> part_labels = PartitionUtils::metis_partition(
        graph_data.adj, config.num_parts);
    std::cout << "[初始化] 图分区完成" << std::endl;
    
    std::vector<std::vector<int>> partition_masks = 
        PartitionUtils::get_partition_masks(part_labels, config.num_parts);
    std::cout << "[初始化] 分区掩码生成完成: " << partition_masks.size() << " 个分区" << std::endl;
    
    // 3. 提取子图
    std::cout << "[初始化] 开始提取子图..." << std::endl;
    for (size_t i = 0; i < partition_masks.size(); i++) {
        const auto& mask = partition_masks[i];
        std::cout << "[初始化] 提取分区 " << i << " (" << mask.size() << " 节点)..." << std::endl;
        SubgraphData subgraph = PartitionUtils::extract_subgraph(graph_data, mask);
        
        // 预处理子图
        subgraph.adj_sub = GraphUtils::add_self_loops(subgraph.adj_sub);
        subgraph.adj_sub = GraphUtils::normalize_adj(subgraph.adj_sub);
        subgraph.features_sub = GraphUtils::preprocess_features(subgraph.features_sub);
        subgraph.support.push_back(subgraph.adj_sub);
        
        subgraph_list.push_back(subgraph);
    }
    std::cout << "[初始化] 子图提取完成: " << subgraph_list.size() << " 个子图" << std::endl;
    
    // 4. 加载任务信息
    std::cout << "[初始化] 加载任务信息..." << std::endl;
    load_tasks_info(config.dataset);
    std::cout << "[初始化] 初始化完成，准备开始训练" << std::endl;
}

MatrixXf Trainer::exec_stage(
    int layer_idx, const std::string& phase, int epoch, int part_id,
    GCNModel& model, const SparseMatrix<float>& adj_norm,
    const MatrixXf& input_data) {
    
    // 获取任务信息和调度决策
    int idx = epoch * 4 + (phase == "UPDATE" && layer_idx == 1 ? 0 :
                           phase == "AGG" && layer_idx == 1 ? 1 :
                           phase == "UPDATE" && layer_idx == 2 ? 2 : 3);
    
    float task_value = 1.0f;
    std::string device = "PNM";
    
    if (idx < static_cast<int>(tasks_info.size())) {
        const TaskInfo& task = tasks_info[idx];
        device = scheduler.schedule_task(task.phase, task.value);
        task_value = task.value;
    } else {
        device = scheduler.schedule_task(phase, 1.0f);
    }
    
    std::cout << "[调度器] Layer " << layer_idx << " " << phase 
              << ": 值=" << std::fixed << std::setprecision(4) << task_value 
              << ", 分配到 " << device << std::endl;
    
    // 记录日志
    StageLog log;
    log.epoch = epoch + 1;
    log.stage = (idx % 4) + 1;
    log.layer = layer_idx;
    log.phase = phase;
    log.value = task_value;
    log.partition_id = part_id;
    log.device = device;
    stage_device_log.push_back(log);
    
    // 执行计算（为 PIM/PNM 任务添加 ZSim 跟踪）
    MatrixXf result;
    
    // 如果是 PIM 或 PNM 任务，使用 ZSim hooks 跟踪
    bool is_pim_pnm = (device == "PIM" || device == "PNM");
    if (is_pim_pnm) {
        zsim_PIM_function_begin();
    }
    
    if (phase == "UPDATE") {
        if (layer_idx == 1) {
            result = model.layer1_update(input_data);
        } else {
            result = model.layer2_update(input_data);
        }
    } else {  // AGG
        if (layer_idx == 1) {
            result = model.layer1_aggregate(adj_norm, input_data);
        } else {
            result = model.layer2_aggregate(adj_norm, input_data);
        }
    }
    
    // 结束 PIM/PNM 任务跟踪
    if (is_pim_pnm) {
        zsim_PIM_function_end();
    }
    
    std::cout << "[完成] Layer " << layer_idx << " " << phase 
              << " 在 " << device << " 上完成 (Partition " << part_id << ")" << std::endl;
    
    return result;
}

TrainingStats Trainer::evaluate(
    GCNModel& model,
    const SubgraphData& subgraph,
    const SparseMatrix<float>& adj_norm) {
    
    model.set_training(false);
    MatrixXf logits = model.forward(adj_norm, subgraph.features_sub);
    model.set_training(true);
    
    TrainingStats stats;
    stats.val_loss = LossFunctions::masked_softmax_cross_entropy(
        logits, subgraph.y_val_sub, subgraph.val_mask_sub);
    stats.val_acc = LossFunctions::masked_accuracy(
        logits, subgraph.y_val_sub, subgraph.val_mask_sub);
    
    return stats;
}

void Trainer::train() {
    std::vector<std::pair<VectorXi, MatrixXf>> all_subgraph_results;
    
    // 对每个分区进行训练
    for (size_t part_id = 0; part_id < subgraph_list.size(); part_id++) {
        SubgraphData& subgraph = subgraph_list[part_id];
        
        std::cout << "\n=== 调度训练 Partition " << part_id 
                  << " (" << subgraph.num_nodes() << " nodes) ===" << std::endl;
        
        // 预处理子图数据
        SparseMatrix<float> adj_norm = subgraph.support[0];
        MatrixXf features = subgraph.features_sub;
        
        // 初始化模型
        GCNModel model(
            features.cols(),              // input_dim
            config.hidden1,               // hidden_dim
            subgraph.y_train_sub.cols(),  // output_dim
            config.dropout,
            static_cast<unsigned int>(config.seed)
        );
        
        // 初始化优化器
        AdamOptimizer optimizer(config.learning_rate);
        optimizer.initialize_state(0, features.cols(), config.hidden1);  // W1
        optimizer.initialize_state(1, config.hidden1, 1);  // b1
        optimizer.initialize_state(2, config.hidden1, subgraph.y_train_sub.cols());  // W2
        optimizer.initialize_state(3, subgraph.y_train_sub.cols(), 1);  // b2
        
        // 训练循环
        std::vector<float> cost_val;
        for (int epoch = 0; epoch < config.epochs; epoch++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::cout << "\n=== Partition " << part_id << " Epoch " << (epoch + 1) << " ===" << std::endl;
            
            // 每 10 个 epoch 输出一次调度器统计信息
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                scheduler.print_statistics();
            }
            
            model.set_training(true);
            
            // 前向传播阶段（包含 PIM/PNM 任务，由 zsim_PIM_function_begin/end 标记）
            // Layer 1 Update
            MatrixXf updated = exec_stage(
                1, "UPDATE", epoch, static_cast<int>(part_id), model, adj_norm, features);
            
            // Layer 1 Aggregate
            MatrixXf aggregated = exec_stage(
                1, "AGG", epoch, static_cast<int>(part_id), model, adj_norm, updated);
            
            // Layer 2 Update
            updated = exec_stage(
                2, "UPDATE", epoch, static_cast<int>(part_id), model, adj_norm, aggregated);
            
            // Layer 2 Aggregate
            MatrixXf outputs = exec_stage(
                2, "AGG", epoch, static_cast<int>(part_id), model, adj_norm, updated);
            
            // 计算损失和准确率
            float train_loss = LossFunctions::masked_softmax_cross_entropy(
                outputs, subgraph.y_train_sub, subgraph.train_mask_sub);
            float train_acc = LossFunctions::masked_accuracy(
                outputs, subgraph.y_train_sub, subgraph.train_mask_sub);
            
            // 添加 L2 正则化
            // 与原 TF 版本保持一致：仅对第一层权重做 L2 正则（embedding 层）
            train_loss += LossFunctions::l2_loss(
                model.get_layer1()->get_weight(), config.weight_decay);
            
            // 反向传播
            MatrixXf grad_output = BackwardPropagator::compute_loss_gradient(
                outputs, subgraph.y_train_sub, subgraph.train_mask_sub);
            
            // Layer 2 反向传播
            MatrixXf grad_layer2 = BackwardPropagator::backward_layer2_aggregate(
                grad_output, adj_norm);
            MatrixXf grad_layer1_agg;
            BackwardPropagator::backward_layer2_update(
                grad_layer2, aggregated, model.get_layer2(), 
                grad_layer1_agg, optimizer, 2);
            
            // Layer 1 反向传播
            MatrixXf grad_layer1 = BackwardPropagator::backward_layer1_aggregate(
                grad_layer1_agg, adj_norm);
            MatrixXf grad_features;
            BackwardPropagator::backward_layer1_update(
                grad_layer1, features, model.get_layer1(), 
                grad_features, optimizer, 0, config.weight_decay);
            
            // 验证
            TrainingStats val_stats = evaluate(model, subgraph, adj_norm);
            cost_val.push_back(val_stats.val_loss);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end_time - start_time).count();
            
            std::cout << "Partition " << part_id << " Epoch: " 
                      << std::setw(4) << std::setfill('0') << (epoch + 1)
                      << " train_loss=" << std::fixed << std::setprecision(5) << train_loss
                      << " train_acc=" << std::setprecision(5) << train_acc
                      << " val_loss=" << std::setprecision(5) << val_stats.val_loss
                      << " val_acc=" << std::setprecision(5) << val_stats.val_acc
                      << " time=" << std::setprecision(5) << duration << std::endl;
            
            // Early stopping
            // 与 Python 版本完全一致：
            // Python: if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1])
            // 逻辑：从第 (early_stopping + 1) 个 epoch 开始，如果当前验证损失大于最近 early_stopping 个 epoch 的平均值，则停止
            if (epoch > config.early_stopping && 
                cost_val.size() > static_cast<size_t>(config.early_stopping + 1)) {
                // cost_val[-(early_stopping+1):-1] 表示从倒数第 (early_stopping+1) 个到倒数第 2 个
                // 即最近 early_stopping 个 epoch（不包括当前 epoch）
                float recent_avg = std::accumulate(
                    cost_val.end() - config.early_stopping - 1, 
                    cost_val.end() - 1, 0.0f) / config.early_stopping;
                
                if (cost_val.back() > recent_avg) {
                    std::cout << "Early stopping..." << std::endl;
                    break;
                }
            }
        }
        
        std::cout << "Optimization Finished for Partition " << part_id << std::endl;
        
        // 测试推理
        model.set_training(false);
        MatrixXf y_pred_sub = model.forward(adj_norm, features);
        all_subgraph_results.push_back({subgraph.part_nodes, y_pred_sub});
    }
    
    // print_statistics(); // 暂时屏蔽调度日志输出，避免刷屏
}

void Trainer::print_statistics() const {
    std::cout << "\n=== 调度器阶段分配日志 ===" << std::endl;
    
    int pim_count = 0, pnm_count = 0, gpu_count = 0;
    
    for (const auto& log : stage_device_log) {
        if (log.device == "PIM") pim_count++;
        else if (log.device == "PNM") pnm_count++;
        else if (log.device == "GPU") gpu_count++;
        
        std::cout << "Epoch " << log.epoch << " 阶段" << log.stage 
                  << " (Layer " << log.layer << " " << log.phase 
                  << "): 值=" << std::fixed << std::setprecision(4) << log.value
                  << ", 分配到 " << log.device << std::endl;
    }
    
    int total_tasks = stage_device_log.size();
    int pim_pnm_count = pim_count + pnm_count;
    
    std::cout << "\n任务分配统计:" << std::endl;
    std::cout << "  总任务数: " << total_tasks << std::endl;
    std::cout << "  PIM: " << pim_count << " (" 
              << (total_tasks > 0 ? 100.0f * pim_count / total_tasks : 0.0f) << "%)" << std::endl;
    std::cout << "  PNM: " << pnm_count << " (" 
              << (total_tasks > 0 ? 100.0f * pnm_count / total_tasks : 0.0f) << "%)" << std::endl;
    std::cout << "  GPU: " << gpu_count << " (" 
              << (total_tasks > 0 ? 100.0f * gpu_count / total_tasks : 0.0f) << "%)" << std::endl;
    std::cout << "  PIM+PNM: " << pim_pnm_count << " (" 
              << (total_tasks > 0 ? 100.0f * pim_pnm_count / total_tasks : 0.0f) << "%)" << std::endl;
}

