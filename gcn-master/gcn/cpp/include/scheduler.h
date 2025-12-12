/*
 * 任务调度器
 * 根据 AMIR/CD 值将任务调度到不同设备（PIM/PNM/GPU）
 */

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ==================== 调度器类 ====================

class Scheduler {
private:
    std::vector<float> history_amir;        // AMIR 历史值
    std::vector<float> history_cd;           // CD 历史值
    float amir_threshold = 5.0f;
    float cd_threshold = 5.0f;
    
    // 阈值计算方法
    float compute_percentile_threshold(const std::vector<float>& values, float percentile = 0.75f);
    float compute_weighted_threshold(const std::vector<float>& values);
    float compute_adaptive_threshold(const std::vector<float>& values);
    float compute_kmeans_threshold(const std::vector<float>& values);
    
    // 计算趋势（斜率）
    float compute_trend(const std::vector<float>& values);
    
public:
    Scheduler() = default;
    
    // 调度任务
    // phase: "AGG" 或 "UPDATE"
    // value: AMIR 值（对于 AGG）或 CD 值（对于 UPDATE）
    // 返回: "PIM", "PNM", 或 "GPU"
    std::string schedule_task(const std::string& phase, float value);
    
    // 更新阈值（自适应方法）
    void update_thresholds();
    
    // 获取当前阈值
    float get_amir_threshold() const { return amir_threshold; }
    float get_cd_threshold() const { return cd_threshold; }
    
    // 获取统计信息（用于调试）
    void print_statistics() const;
};

#endif // SCHEDULER_H

