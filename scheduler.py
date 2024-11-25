from collections import deque, defaultdict

from models import UserWorkload, Node, Pod
from scheduling_strategies import SchedulingStrategy


class Scheduler:
    """
    Scheduler类用于管理Pod的调度，负责根据不同调度策略、节点资源和RTT（Round-Trip Time）等因素来调度Pod

    Attributes:
        node_list (list[Node]): 集群中的所有节点。
        rtt_table (dict[str, dict[str, int]]): 节点之间的RTT表，记录每个节点到其他节点的RTT。
        record (dict): 记录每个Pod被调度到的节点。
        strategy (SchedulingStrategy): 调度策略实例，用于选择节点。

    Methods:
        schedule_workload(user: UserWorkload) -> None:
            根据用户的Pod依赖关系调度所有Pod。

        schedule_pod(pod: Pod) -> None:
            根据节点资源和RTT调度单个Pod。

        get_scheduling_record() -> None:
            打印调度结果，显示每个Pod被调度到的节点。

        print_cluster_state() -> None:
            打印当前集群的所有节点的资源使用情况。

        get_pod_rtt(_from: Pod, _to: Pod) -> int:
            获取两个Pod所在节点之间的RTT值。

        calculate_response_time(user_workload: UserWorkload) -> float:
            计算给定用户工作负载的期望响应时间（根据Pod调度的RTT）。
    """

    def __init__(self, init_node_list, global_rtt_table: dict[str, dict[str, int]],
                 used_strategy: SchedulingStrategy):
        self.node_list = init_node_list  # 集群中的所有节点
        self.rtt_table = global_rtt_table  # 节点对之间 到达每个节点的 RTT（往返时间）

        self.record = {}  # 记录单次调度的结果
        self.strategy = used_strategy  # 调度策略
        self.assigned_node = []  # 记录被调度过的node的顺序

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
            if (pod.cpu_resource <= n.cpu_capacity and
                pod.mem_resource <= n.mem_capacity and
                pod.band_resource <= n.band_capacity)
        ]

        if not candidate_nodes:
            print(f"没有可用的节点来调度 Pod {pod.name}")
            return

        # 上一个被选择的节点
        prev = None if len(self.assigned_node) == 0 else self.assigned_node[-1]
        # zeta表示了之前所有的调度之后 节点上的workload大小
        zeta = defaultdict(int)
        for ni in self.assigned_node:
            zeta[ni] += 1

        # 从候选节点中随机选择一个节点
        selected_node: Node = self.strategy.select_node(
            pod=pod,
            candidate_nodes=candidate_nodes,
            rtt=self.rtt_table,
            prev_pod_node=prev,
            node_workload=zeta,
        )

        selected_node.assign_pod(pod)
        pod.start_pod()
        self.record[pod.name] = selected_node
        self.assigned_node.append(selected_node.name)

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
        _from_node: Node = self.record.get(_from.name)
        _to_node: Node = self.record.get(_to.name)

        _from_node_name = _from_node.name
        _to_node_name = _to_node.name
        # 根据pod所在的节点获取节点之间的rtt
        if _from_node_name in self.rtt_table and _to_node_name in self.rtt_table[_from_node_name]:
            return self.rtt_table[_from_node_name][_to_node_name]
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


    """
        workload generator -> firewall -> NAT -> IDS -> cache -> Load balancer ->  receiver
        圆圈(空心) -> 星星 -> 三角 -> 菱形 -> 五边形 -> 正方形 -> 圆圈(实心)
    """