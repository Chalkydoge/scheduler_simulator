from collections import deque, defaultdict

from models import UserWorkload, Node, Pod
from scheduling_strategies import SchedulingStrategy


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

