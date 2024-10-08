from config_loader import ConfigLoader
from scheduler import Scheduler

"""
    Single test
"""
if __name__ == '__main__':
    # 初始化配置加载器
    config_loader = ConfigLoader('./data/setup1.yaml')

    # 根据配置文件创建集群节点
    node_list = config_loader.create_nodes()

    # 创建RTT表
    rtt_table = config_loader.create_rtt_table()

    # 初始化Pod对象
    pods = config_loader.create_pods()

    # 创建Pod依赖关系
    dependencies = config_loader.create_dependencies(pods)

    # 选择调度策略
    strategy = config_loader.select_strategy()

    # 创建调度器
    scheduler = Scheduler(init_node_list=node_list, global_rtt_table=rtt_table, used_strategy=strategy)

    # 创建用户工作负载
    workload = config_loader.create_workload(pods, dependencies)

    # 调度用户工作负载
    scheduler.schedule_workload(workload)

    # 获取调度记录并打印集群状态
    scheduler.get_scheduling_record()
    scheduler.print_cluster_state()

    # 计算并打印响应时间
    mx_resp = scheduler.calculate_response_time(user_workload=workload)
    print("User workload的期望响应时间 = {}ms".format(mx_resp))

