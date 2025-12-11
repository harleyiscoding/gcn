/*
 * 训练器头文件
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "types.h"
#include "gcn_layer.h"
#include "scheduler.h"
#include "optimizer.h"
#include "loss.h"
#include "backward.h"
#include "graph_utils.h"
#include "data_loader.h"
#include <vector>
#include <string>

// ==================== 训练器类 ====================

class Trainer {
private:
    GCNConfig config;
    GraphData graph_data;
    std::vector<SubgraphData> subgraph_list;
    std::vector<TaskInfo> tasks_info;
    Scheduler scheduler;
    std::vector<StageLog> stage_device_log;
    
    // 从文件读取任务信息（AMIR/CD 值）
    void load_tasks_info(const std::string& dataset);
    
    // 执行一个阶段（Update 或 Aggregate）
    MatrixXf exec_stage(
        int layer_idx, const std::string& phase, int epoch, int part_id,
        GCNModel& model, const SparseMatrix<float>& adj_norm,
        const MatrixXf& input_data);
    
    // 评估函数
    TrainingStats evaluate(
        GCNModel& model,
        const SubgraphData& subgraph,
        const SparseMatrix<float>& adj_norm);
    
public:
    Trainer(const GCNConfig& cfg) : config(cfg) {}
    
    // 初始化：加载数据和分区
    void initialize();
    
    // 训练主循环
    void train();
    
    // 获取统计信息
    void print_statistics() const;
};

#endif // TRAINER_H

