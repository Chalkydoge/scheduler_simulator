"""
Main application
"""
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from mock import dataset_generation
from schedule import SchedulerService


def plot_scheduling_strategies_performance(data_dict, problem_sizes):
    """
    绘制不同调度策略在不同问题规模下的性能对比柱状图。

    参数:
    data_dict (defaultdict of list): 调度策略名称到性能值列表的映射。
    problem_sizes (list): 问题规模大小列表，对应x轴。
    """

    # 设置柱状图宽度
    bar_width = 0.1
    index = np.arange(len(problem_sizes))
    fig, ax = plt.subplots(figsize=(10, 6))
    opacity = 0.8

    for i, (strategy, values) in enumerate(data_dict.items()):
        ax.bar(index + i * bar_width, values, bar_width,
               alpha=opacity,
               label=strategy)

    ax.set_xlabel('Number of Nodes, CNF chain size = 2 * number_nodes')
    ax.set_ylabel('Latency Performance')  # 这里根据实际意义可以替换为'时间'、'成本'等
    ax.set_title('Comparison')
    ax.set_xticks(index + bar_width * (len(data_dict) - 1) / 2)
    ax.set_xticklabels(problem_sizes)
    ax.legend()

    plt.tight_layout()
    plt.savefig("cmp1.png")


if __name__ == '__main__':
    # 1. 生成数据 [2, 3, 5, 7, 10, 17, 23, 29]
    sample = [2, 3, 5, 7, 10, 17, 23, 29, 37]
    dataset_generation(sample)
    result = defaultdict(list)
    # 2. 每组数据上运行调度策略
    for node_size in sample:
        for sc in ["RandomSchedulingStrategy", "LeastResourceSchedulingStrategy", "DelayAwareSchedulingStrategy",
                   "InterferenceAwareSchedulingStrategy", "NetMarksSchedulingStrategy", "IPlaceSchedulingStrategy"]:
            # 初始化调度服务
            scheduler_service = SchedulerService('./data/setup{}_{}.yaml'.format(node_size, sc[:5]))

            # 获取调度的指标，不额外指定就根据yaml内的配置进行测试
            metrics = scheduler_service.print_scheduler_metrics()
            data = metrics['response_time'][0] / 1000.0
            # 输出调度结果
            print("{} scheduled Sample results: {}s.".format(sc, data))
            result[sc].append(data)

    # 3. 统计结果
    print(result)
    plot_scheduling_strategies_performance(result, sample)

    # 建议：让GPT写
