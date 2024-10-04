from scheduler import Scheduler
from scheduling_strategies import RandomSchedulingStrategy, LeastResourceSchedulingStrategy, DelayAwareSchedulingStrategy
from models import Pod, UserWorkload, Node


if __name__ == '__main__':
    # 创建三个节点，假设它们的资源情况如下
    node1 = Node(name="Node-1", cpu_capacity=16, mem_capacity=32000, band_capacity=1000)
    node2 = Node(name="Node-2", cpu_capacity=8, mem_capacity=16000, band_capacity=500)
    node3 = Node(name="Node-3", cpu_capacity=32, mem_capacity=64000, band_capacity=2000)

    # 集群节点列表
    node_list = [node1, node2, node3]

    # RTT 列表，假设 RTT 分别为 20 ms, 30 ms, 10 ms
    rtt_list = [20, 30, 10]

    # 定义 RTT 表，存储每个节点到其他节点的 RTT
    rtt_table = {
        node1: {node1: 0, node2: 10, node3: 100},  # 从 node1 到其他节点的 RTT
        node2: {node1: 10, node2: 0, node3: 20},  # 从 node2 到其他节点的 RTT
        node3: {node1: 100, node2: 20, node3: 0}  # 从 node3 到其他节点的 RTT
    }

    # 选择调度策略
    strategy = RandomSchedulingStrategy()  # 使用随机调度策略
    # strategy = LeastResourceSchedulingStrategy()  # 使用最少资源调度策略

    # 创建调度器，掌握所有节点及其 RTT
    scheduler = Scheduler(init_node_list=node_list, global_rtt_table=rtt_table, used_strategy=strategy)

    # 输出每个节点的剩余资源
    for node in node_list:
        print(
            f"初始节点 {node.name} 的剩余资源: CPU: {node.cpu_capacity} vCPUs, 内存: {node.mem_capacity} MB, 带宽: {node.band_capacity} Mbps")

    # 创建 Pods
    pod_a = Pod(name="Pod-A", cpu_resource=2, mem_resource=2048, band_resource=200, setup_time=30)
    pod_b = Pod(name="Pod-B", cpu_resource=2, mem_resource=2048, band_resource=200, setup_time=50)
    pod_c = Pod(name="Pod-C", cpu_resource=4, mem_resource=4096, band_resource=300, setup_time=60)

    # 创建依赖关系，Pod C 依赖于 Pod A 和 Pod B
    dependencies = {
        pod_c: [pod_a, pod_b]
    }

    workload = UserWorkload(name="user1", pod_list=[pod_a, pod_b, pod_c], num_pods=3, work_dependencies=dependencies)

    scheduler.schedule_workload(workload)
    scheduler.get_scheduling_record()
    scheduler.print_cluster_state()
    mx_resp = scheduler.calculate_response_time(user_workload=workload)
    print("User workload的期望响应时间 = {}ms".format(mx_resp))
