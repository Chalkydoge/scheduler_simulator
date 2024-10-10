import yaml
from models import Node, Pod, UserWorkload
from scheduling_strategies import RandomSchedulingStrategy, LeastResourceSchedulingStrategy, \
    DelayAwareSchedulingStrategy


class ConfigLoader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        with open(self.config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def create_nodes(self):
        """根据配置文件初始化Node对象"""
        nodes = [Node(**node) for node in self.config['nodes']]
        return nodes

    def create_rtt_table(self):
        """根据配置文件创建RTT表"""
        rtt_table = {}
        for node_name, rtt_values in self.config['rtt_table'].items():
            rtt_table[node_name] = {key: value for key, value in rtt_values.items()}
        return rtt_table

    def create_pods(self):
        """根据配置文件初始化Pod对象"""
        pods = {pod['name']: Pod(**pod) for pod in self.config['pods']}
        return pods

    def create_dependencies(self, pods):
        """根据配置文件创建Pod的依赖关系"""
        dependencies = {}
        for dependent, dependencies_list in self.config['dependencies'].items():
            dependencies[pods[dependent]] = [pods[dep] for dep in dependencies_list]
        return dependencies

    def select_strategy(self):
        """根据配置文件选择调度策略"""
        strategy = self.config['strategy']
        if strategy == "RandomSchedulingStrategy":
            return RandomSchedulingStrategy()
        elif strategy == "LeastResourceSchedulingStrategy":
            return LeastResourceSchedulingStrategy()
        elif strategy == "DelayAwareSchedulingStrategy":
            return DelayAwareSchedulingStrategy()
        else:
            raise ValueError(f"未知的调度策略: {strategy}")

    def assign_strategy(self, strategy_name):
        """根据外部指定的strategy来显式的选择运行的策略"""
        if strategy_name == "RandomSchedulingStrategy":
            return RandomSchedulingStrategy()
        elif strategy_name == "LeastResourceSchedulingStrategy":
            return LeastResourceSchedulingStrategy()
        elif strategy_name == "DelayAwareSchedulingStrategy":
            return DelayAwareSchedulingStrategy()
        else:
            raise ValueError(f"未知的调度策略: {strategy_name}")

    def create_workload(self, pods, dependencies):
        """根据配置文件创建用户工作负载"""
        workload = UserWorkload(
            name="user1",
            pod_list=list(pods.values()),
            num_pods=len(pods),
            work_dependencies=dependencies
        )
        return workload
