import random
import time
import logging
import os
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 任务结构体
class Task:
    def __init__(self, task_id, AMIR, CD):
        self.id = task_id
        self.AMIR = AMIR
        self.CD = CD
        self.device = None
        self.status = 'pending'  # pending, running, finished, fallback, failed

    def __repr__(self):
        return f"Task(id={self.id}, AMIR={self.AMIR}, CD={self.CD}, device={self.device}, status={self.status})"

# 工具函数：解析performance_analysis_report.txt获取LLC_loads
def parse_llc_loads(perf_report_path):
    if not os.path.exists(perf_report_path):
        return None
    with open(perf_report_path, 'r') as f:
        content = f.read()
    # 尝试匹配"LLC Loads"或"LLC-loads"，或直接用"LLC-loads"字段
    match = re.search(r'Average LLC Loads per Epoch:\s*([\deE+\.-]+)', content)
    if match:
        return float(match.group(1))
    # 兼容旧格式，尝试直接找LLC-loads
    match = re.search(r'LLC-loads:\s*([\deE+\.-]+)', content)
    if match:
        return float(match.group(1))
    # 兼容直接找LLC Misses（如果没有Loads）
    match = re.search(r'Average LLC Misses per Epoch:\s*([\deE+\.-]+)', content)
    if match:
        return float(match.group(1))
    return None

# 工具函数：解析flops_analysis_report.txt获取FLOPs
# 返回dict: {'layer1_update':..., 'layer1_aggregate':..., ...}
def parse_flops(flops_report_path):
    if not os.path.exists(flops_report_path):
        return None
    flops = {}
    with open(flops_report_path, 'r') as f:
        for line in f:
            m = re.match(r'(layer\d+_\w+):\s*([\deE+]+)', line)
            if m:
                flops[m.group(1)] = float(m.group(2))
    return flops if flops else None

# 任务生成器（支持真实AMIR计算）
def generate_tasks(num_tasks, perf_report_path=None, flops_report_path=None):
    tasks = []
    llc_loads = None
    flops = None
    if perf_report_path and flops_report_path:
        llc_loads = parse_llc_loads(perf_report_path)
        flops = parse_flops(flops_report_path)
    for i in range(num_tasks):
        # 默认随机
        AMIR = round(random.uniform(1, 10), 2)
        CD = round(random.uniform(0.5, 15), 2)
        # 若有真实数据，优先用真实AMIR
        if llc_loads and flops:
            # 以layer1_aggregate和layer1_update为例
            memory_access_agg = llc_loads * 64  # 字节
            flops_update = flops.get('layer1_update', None)
            if memory_access_agg and flops_update and flops_update > 0:
                AMIR = memory_access_agg / flops_update
            # CD可用FLOPs/MemoryAccess（如有需要可扩展）
            flops_agg = flops.get('layer1_aggregate', None)
            if flops_agg and memory_access_agg > 0:
                CD = flops_agg / memory_access_agg
        tasks.append(Task(task_id=i, AMIR=AMIR, CD=CD))
    return tasks

# 调度器
class Scheduler:
    def __init__(self):
        pass

    def schedule_task(self, task):
        if task.AMIR > 5 and task.CD < 2:
            return 'PIM'
        elif task.CD > 10:
            return 'GPU'
        else:
            return 'PNM'

    def schedule_batch(self, tasks):
        for task in tasks:
            task.device = self.schedule_task(task)
            logging.info(f"Task {task.id} assigned to {task.device} (AMIR={task.AMIR}, CD={task.CD})")

# 设备监控与回退
class DeviceMonitor:
    def __init__(self):
        self.device_status = {'GPU': 'normal', 'PIM': 'normal', 'PNM': 'normal'}
        self.overload_prob = {'GPU': 0.2, 'PIM': 0.1, 'PNM': 0.05}  # 模拟超载概率

    def update_status(self):
        for device in self.device_status:
            if random.random() < self.overload_prob[device]:
                self.device_status[device] = 'overloaded'
            else:
                self.device_status[device] = 'normal'
        logging.info(f"Device status: {self.device_status}")

    def is_overloaded(self, device):
        return self.device_status[device] == 'overloaded'

# 回退管理
class FallbackManager:
    def __init__(self, monitor):
        self.monitor = monitor

    def fallback(self, task):
        # 优先PIM, 其次PNM, 最后GPU
        candidates = ['PIM', 'PNM', 'GPU']
        for device in candidates:
            if not self.monitor.is_overloaded(device):
                old_device = task.device
                task.device = device
                task.status = 'fallback'
                logging.warning(f"Task {task.id} fallback from {old_device} to {device}")
                return
        logging.error(f"Task {task.id} cannot fallback, all devices overloaded!")

# 任务分发器
class Dispatcher:
    def dispatch(self, tasks):
        exec_time = {
            'PIM': 0.02,
            'PNM': 0.04,
            'GPU': 0.01
        }
        for task in tasks:
            # 日志区分回退任务
            if task.status == 'fallback':
                logging.info(f"Task {task.id} [fallback] dispatched to {task.device}")
            else:
                logging.info(f"Dispatching Task {task.id} to {task.device}")
            task.status = 'running'
            # 模拟小概率执行失败
            if random.random() < 0.05:
                task.status = 'failed'
                logging.error(f"Task {task.id} failed on {task.device}")
                continue
            # 模拟执行
            time.sleep(exec_time.get(task.device, 0.01))
            task.status = 'finished'
            logging.info(f"Task {task.id} finished on {task.device}")

# 主流程
if __name__ == "__main__":
    # 可指定数据集路径
    dataset = 'cora'  # 可修改为cora/citeseer等
    base_dir = os.path.dirname(os.path.abspath(__file__))
    perf_report_path = os.path.join(base_dir, 'results', dataset, 'l1_cache_analysis', 'performance_analysis_report.txt')
    flops_report_path = os.path.join(base_dir, 'results', dataset, 'tf-flops', sorted(os.listdir(os.path.join(base_dir, 'results', dataset, 'tf-flops')))[-1], 'flops_analysis_report.txt') if os.path.exists(os.path.join(base_dir, 'results', dataset, 'tf-flops')) else None

    num_tasks = 20
    tasks = generate_tasks(num_tasks, perf_report_path, flops_report_path)
    scheduler = Scheduler()
    monitor = DeviceMonitor()
    fallback_manager = FallbackManager(monitor)
    dispatcher = Dispatcher()

    # 1. 调度
    scheduler.schedule_batch(tasks)

    # 2. 设备状态监控与回退
    monitor.update_status()
    for task in tasks:
        if monitor.is_overloaded(task.device):
            fallback_manager.fallback(task)

    # 3. 分发任务
    dispatcher.dispatch(tasks)

    # 4. 输出最终任务状态
    for task in tasks:
        print(task) 