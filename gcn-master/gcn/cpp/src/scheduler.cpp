/*
 * 任务调度器实现
 */

#include "../include/scheduler.h"
#include <algorithm>
#include <numeric>
#include <cmath>

float Scheduler::compute_kmeans_threshold(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 5.0f;  // 默认阈值
    }
    
    // 简化的 K-means（2 聚类）
    // 使用中位数作为初始中心
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    int mid = sorted_values.size() / 2;
    float center1 = sorted_values[mid / 2];
    float center2 = sorted_values[mid + (sorted_values.size() - mid) / 2];
    
    // 迭代几次（简化版，实际应该迭代到收敛）
    for (int iter = 0; iter < 10; iter++) {
        float sum1 = 0.0f, sum2 = 0.0f;
        int count1 = 0, count2 = 0;
        
        for (float val : sorted_values) {
            float dist1 = std::abs(val - center1);
            float dist2 = std::abs(val - center2);
            
            if (dist1 < dist2) {
                sum1 += val;
                count1++;
            } else {
                sum2 += val;
                count2++;
            }
        }
        
        if (count1 > 0) center1 = sum1 / count1;
        if (count2 > 0) center2 = sum2 / count2;
    }
    
    // 返回两个中心的平均值作为阈值
    return (center1 + center2) / 2.0f;
}

std::string Scheduler::schedule_task(const std::string& phase, float value) {
    if (phase == "AGG") {
        history_amir.push_back(value);
        update_thresholds();
        return (value > amir_threshold) ? "PIM" : "PNM";
    } else if (phase == "UPDATE") {
        history_cd.push_back(value);
        update_thresholds();
        return (value > cd_threshold) ? "GPU" : "PNM";
    } else {
        return "PNM";  // 默认
    }
}

void Scheduler::update_thresholds() {
    // 只使用最近 20 个值
    const int window_size = 20;
    
    if (history_amir.size() >= window_size) {
        std::vector<float> recent_amir(
            history_amir.end() - window_size, 
            history_amir.end()
        );
        amir_threshold = compute_kmeans_threshold(recent_amir);
    }
    
    if (history_cd.size() >= window_size) {
        std::vector<float> recent_cd(
            history_cd.end() - window_size, 
            history_cd.end()
        );
        cd_threshold = compute_kmeans_threshold(recent_cd);
    }
}

