from collections import deque, defaultdict
from abc import ABC, abstractmethod
import random


class SchedulingStrategy(ABC):
    @abstractmethod
    def select_node(self, pod, candidate_nodes):
        """从候选节点中选择一个节点来调度 Pod"""
        pass


class RandomSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes):
        return random.choice(candidate_nodes)


class LeastResourceSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes):
        return max(candidate_nodes, key=lambda n: (n.cpu_capacity, n.mem_capacity, n.band_capacity))


class DelayAwareSchedulingStrategy(SchedulingStrategy):
    def select_node(self, pod, candidate_nodes):
        alpha = 0.1
        zeta_i = 0
        lambda_threshold = 114514
        delays = {}

        def calculate_delay_processing(u, p, n):
            dp = 0.0
            return dp

        def calculate_delay_network(u, p, n):
            dn = 0.0
            return dn

        # 计算每个节点的延迟
        for ni in candidate_nodes:
            # 计算传播延迟和网络延迟
            Dp = calculate_delay_processing(uk, pj, ni)
            Dn = calculate_delay_network(uk, pj, ni, preferred_node)

            # 计算综合延迟 a_i,j
            a_i_j = alpha * Dp + (1 - alpha) * Dn

            # 如果 ζi > 0 ，添加 bias
            if zeta_i > 0:
                a_i_j += bias

            # 保存延迟结果
            delays[ni] = a_i_j

        # 选择最小延迟的节点
        n_ref = min(delays, key=delays.get)
        node_selected = [n_ref]

        # 计算 n_ref 的平均延迟
        a_ref = delays[node_selected]

        # 检查其他节点的延迟是否在阈值范围内
        for ni in candidate_nodes:
            if ni != n_ref and delays[ni] <= a_ref + lambda_threshold:
                node_selected.append(ni)

        # 如果多个节点在选集 N_SEL 中，选择具有最大 min(rCPU, rMEM) 的节点
        if len(node_selected) > 1:
            return max(node_selected, key=lambda ni: min(ni.rCPU, ni.rMEM))

        # 如果只有一个节点，直接返回 n_ref
        return n_ref


class Pod:
    def __init__(self, name, cpu_resource, mem_resource, band_resource, setup_time):
        self.name = name  # Pod 名称
        self.cpu_resource = cpu_resource  # CPU 需求(mCPU)
        self.mem_resource = mem_resource  # 内存需求（MB)
        self.band_resource = band_resource  # 带宽需求(Mbps)
        self.setup_time = setup_time  # 启动时间，单位是秒(average seconds)
        self.scheduled = False  # Pod 是否已经被调度

    def start_pod(self):
        print(f"Pod {self.name} 启动中，预计启动时间为 {self.setup_time} 秒...")
        print(f"Pod {self.name} 启动完成")
        self.scheduled = True

    # 实现 __str__ 方法，方便打印 Pod 对象时提供更友好的信息
    def __str__(self):
        return (f"Pod(name={self.name}, "
                f"CPU={self.cpu_resource}, "
                f"Memory={self.mem_resource}, "
                f"Bandwidth={self.band_resource}, "
                f"Setup Time={self.setup_time}s)")

    # 提供更有用的字符串表示，以便调试和打印时使用
    def __repr__(self):
        return (f"Pod(name={self.name}, "
                f"CPU={self.cpu_resource}, "
                f"Memory={self.mem_resource}, "
                f"Bandwidth={self.band_resource}, "
                f"Setup Time={self.setup_time}s)")


class UserWorkload:
    def __init__(self, name, num_pods, pod_list, work_dependencies):
        self.name = name  # 用户工作负载的名称（应用程序名称）
        self.num_pods = num_pods  # 该工作负载运行的 Pod 数量
        self.pods = pod_list  # 存放所有 Pod 的列表
        self.dependencies = work_dependencies  # Pod 的依赖关系

    # 构建依赖图，并进行拓扑排序
    def resolve_dependencies(self):
        graph = defaultdict(list)  # 依赖图
        ind = {pod: 0 for pod in self.pods}  # 每个 Pod 的入度（依赖数量）

        # 构建图和计算每个 Pod 的入度
        for pod, deps in self.dependencies.items():
            for dep in deps:
                graph[dep].append(pod)
                ind[pod] += 1

        # 拓扑排序（Kahn 算法）
        topo_order = []
        q = deque([pod for pod in self.pods if ind[pod] == 0])

        while q:
            current = q.popleft()
            topo_order.append(current)

            # 遍历当前 Pod 的依赖
            for neighbor in graph[current]:
                ind[neighbor] -= 1
                if ind[neighbor] == 0:
                    q.append(neighbor)

        # 检查是否存在环
        if len(topo_order) != len(self.pods):
            raise ValueError("依赖图中存在循环依赖，无法完成拓扑排序")

        return topo_order


class Node:
    def __init__(self, name: str, cpu_capacity: int, mem_capacity: int, band_capacity: int):
        self.name: str = name  # 节点名称
        self.cpu_capacity: int = cpu_capacity  # 节点的 CPU 容量
        self.mem_capacity: int = mem_capacity  # 节点的内存容量
        self.band_capacity: int = band_capacity  # 节点的带宽容量
        self.pods = []  # 节点上运行的 Pods 列表

    # 为节点分配 Pod
    def assign_pod(self, pod):
        self.pods.append(pod)
        self.cpu_capacity -= pod.cpu_resource
        self.mem_capacity -= pod.mem_resource
        self.band_capacity -= pod.band_resource
        print(f"Pod {pod.name} 成功调度到节点 {self.name}")


class Scheduler:
    def __init__(self, init_node_list, global_rtt_table: dict[Node, dict[Node, int]],
                 used_strategy: SchedulingStrategy):
        self.node_list = init_node_list  # 集群中的所有节点
        # self.master = init_node_list[0]  # 主节点
        self.rtt_table = global_rtt_table  # 主节点 到达每个节点的 RTT（往返时间）

        self.record = {}  # 记录单次调度的结果
        self.strategy = used_strategy  # 调度策略

    # 根据workload的dependencies按照要求调度pod
    def schedule_workload(self, user: UserWorkload):
        # user为UserWorkload
        topo = []
        path = defaultdict(list)
        ind = [0] * user.num_pods
        mapping = {}
        for idx, pod in enumerate(user.pods):
            mapping[pod] = idx

        for _from, v in user.dependencies.items():
            for _to in v:
                path[mapping[_to]].append(mapping[_from])
                ind[mapping[_from]] += 1

        q = deque([])
        for i in range(user.num_pods):
            if ind[i] == 0:
                q.append(i)
        while q:
            cur = q.popleft()
            topo.append(cur)
            for nxt in path[cur]:
                ind[nxt] -= 1
                if ind[nxt] == 0:
                    q.append(nxt)

        if len(topo) != user.num_pods:
            raise ValueError("Workload依赖图中存在循环依赖，无法完成拓扑排序")

        for order in topo:
            self.schedule_pod(user.pods[order])

    # 根据资源和 RTT 来调度 Pod
    def schedule_pod(self, pod):
        # 筛选出可被调度的节点 (即能够满足资源要求的节点)
        candidate_nodes = [
            n for n in self.node_list
            if pod.cpu_resource <= n.cpu_capacity and
               pod.mem_resource <= n.mem_capacity and
               pod.band_resource <= n.band_capacity
        ]

        if not candidate_nodes:
            print(f"没有可用的节点来调度 Pod {pod.name}")
            return

        # 从候选节点中随机选择一个节点
        selected_node = self.strategy.select_node(pod=pod, candidate_nodes=candidate_nodes)
        selected_node.assign_pod(pod)
        pod.start_pod()
        self.record[pod.name] = selected_node

    # 打印调度结果
    def get_scheduling_record(self):
        print("\n调度记录:")
        for pod_name, target_node in self.record.items():
            print(f"Pod {pod_name} 被调度到 Node {target_node.name}")

    # 打印当前集群内所有节点状态
    def print_cluster_state(self):
        print("\n当前集群状态:")
        for current_node in self.node_list:
            print(f"节点 {current_node.name} 状态:")
            print(f"  剩余 CPU: {current_node.cpu_capacity} vCPUs")
            print(f"  剩余内存: {current_node.mem_capacity} MB")
            print(f"  剩余带宽: {current_node.band_capacity} Mbps")
            print(f"  已调度的 Pods: {[pod.name for pod in current_node.pods]}")
        print()

    # 获取给定 Pod 调度到的节点的 RTT
    def get_pod_rtt(self, _from: Pod, _to: Pod):
        _from_node = self.record.get(_from.name)
        _to_node = self.record.get(_to.name)
        # 根据pod所在的节点获取节点之间的rtt
        if _from_node in self.rtt_table and _to_node in self.rtt_table[_from_node]:
            return self.rtt_table[_from_node][_to_node]
        else:
            return float('inf')  # 如果没有找到 RTT，则返回无穷大

    # 计算完成调度之后user workload的期望response time（根据已有rtt进行估计）
    def calculate_response_time(self, user_workload: UserWorkload):
        g = defaultdict(list)  # 依赖图
        ind = {pod: 0 for pod in user_workload.pods}  # 每个 Pod 的入度（依赖数量）
        # 初始化 DP 表，每个 Pod 的最大响应时间
        dp = {pod: float('-inf') for pod in user_workload.pods}

        # 构建图和计算每个 Pod 的入度
        for pod, deps in user_workload.dependencies.items():
            for dep in deps:
                g[dep].append(pod)
                ind[pod] += 1

        q = deque([])
        for pod in user_workload.pods:
            if ind[pod] == 0:
                q.append(pod)
                dp[pod] = 0  # 初始pod 没有影响

        while q:
            cur = q.popleft()
            # 遍历当前 Pod 的依赖
            for nxt in g[cur]:
                ind[nxt] -= 1
                if ind[nxt] == 0:
                    q.append(nxt)
                # 只要存在依赖关系的都可能是依赖项 进行dp
                w = self.get_pod_rtt(_from=cur, _to=nxt)
                if dp[cur] + w > dp[nxt]:
                    dp[nxt] = dp[cur] + w

        # 最终的期望响应时间是 dp 表中的最大值
        max_response_time = max(dp.values())
        return max_response_time


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
