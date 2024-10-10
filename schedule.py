from config_loader import ConfigLoader
from scheduler import Scheduler


class SchedulerService:
    """
    SchedulerService类用于初始化集群、创建调度器、调度用户工作负载，并提供调度指标。

    通过提供配置文件路径，该类会自动加载集群节点、RTT表、Pods、依赖关系以及调度策略，
    并使用这些信息执行调度工作负载的任务，返回调度的结果（例如响应时间）。
    """
    def __init__(self, config_path):
        """
        构造函数，初始化SchedulerService类。
        Args:
            config_path (str): 指定配置文件的路径，该文件定义了集群、RTT表、Pods等信息。
        """
        # 初始化配置加载器
        self.config_loader = ConfigLoader(config_path)

        # 根据配置文件创建集群节点
        self.node_list = self.config_loader.create_nodes()

        # 创建RTT表
        self.rtt_table = self.config_loader.create_rtt_table()

        # 初始化Pod对象
        self.pods = self.config_loader.create_pods()

        # 创建Pod依赖关系
        self.dependencies = self.config_loader.create_dependencies(self.pods)

        # 选择调度策略
        self.strategy = self.config_loader.select_strategy()

        # 创建调度器
        self.scheduler = Scheduler(init_node_list=self.node_list, global_rtt_table=self.rtt_table,
                                   used_strategy=self.strategy)

        # 创建用户工作负载
        self.workload = self.config_loader.create_workload(self.pods, self.dependencies)

    def run_scheduler(self, strategy_name=None):
        """
        执行调度工作负载并计算期望响应时间。

        该方法会调度用户的工作负载，记录调度结果，并打印当前集群状态。最终返回调度任务的期望响应时间。

        Returns:
            float: 调度任务的期望响应时间（单位为毫秒）。
        """

        # 按照新选择的策略进行覆盖
        if strategy_name is not None:
            self.strategy = self.config_loader.assign_strategy(strategy_name)
            self.scheduler = Scheduler(init_node_list=self.node_list, global_rtt_table=self.rtt_table,
                                       used_strategy=self.strategy)

        # 调度用户工作负载
        self.scheduler.schedule_workload(self.workload)

        # 获取调度记录
        self.scheduler.get_scheduling_record()

        # 打印集群状态（可选：如果你希望在调用时打印出来）
        self.scheduler.print_cluster_state()

        # 计算响应时间
        mx_resp = self.scheduler.calculate_response_time(user_workload=self.workload)
        return mx_resp

    def print_scheduler_metrics(self, strategy_name=None):
        """调用此方法来返回调度的相关指标，供外部使用"""
        # 运行调度并获取期望响应时间
        response_time = self.run_scheduler(strategy_name)

        # 返回调度的指标，例如响应时间
        m = {
            'response_time': response_time
        }

        # 你可以根据需求扩展返回的其他调度指标
        return m


if __name__ == '__main__':
    # 初始化调度服务
    scheduler_service = SchedulerService('./data/setup1.yaml')

    # 获取调度的指标，不额外指定就根据yaml内的配置进行测试
    metrics = scheduler_service.print_scheduler_metrics()

    # 输出调度结果
    print("Sample results: ", metrics)
