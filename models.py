from collections import defaultdict, deque


class Pod:
    def __init__(self, name, cpu_resource, mem_resource, band_resource, setup_time, cnf_type=None):
        valid_cnf_types = ["Traffic Monitor", "Firewall", "NAT", "Load Balancer", "Cache",
                           "Intrusion Detection", "Video Transcoder", "WAN Optimizer", "Mqtt Broker"]

        if cnf_type is not None and cnf_type not in valid_cnf_types:
            raise ValueError(f"Invalid cnf_type. Expected one of {valid_cnf_types}")

        self.cnf_type = cnf_type
        self.name = name  # Pod 名称
        self.cpu_resource = cpu_resource  # CPU 需求(mCPU)
        self.mem_resource = mem_resource  # 内存需求（MB)
        self.band_resource = band_resource  # 带宽需求(Mbps)
        self.setup_time = setup_time    # 启动时间，单位是秒(average seconds)
        self.cnf_type = cnf_type        # CNF 类型
        self.scheduled = False          # Pod 是否已经被调度

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
        # 响应User工作负载的 CNF类型 只需要放在Pod上面就行

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

    # 模拟资源竞争导致的延迟
    # 改进为 根据Pod的类型 产生数据处理的延迟 data_processing_delay, 以及 deployment_delay
    # cpu/band的delay本身就有 不属于额外的贡献
    def resource_contention_delay(self, alpha: float, beta: float, enable_drop=False, enable_jitter=False):
        """
        根据目前节点上已经存在的CNF pod的关系 {cnf1: num1, cnf2: num2, ...}
        计算将cnf_new 放入带来的延迟大小 data_processing_delay, deployment_delay

        :param alpha:
        :param beta:
        :param enable_drop:
        :param enable_jitter:
        :return:
        """


        # 计算 CPU 竞争引发的延迟
        total_cpu_usage = sum(pod.cpu_resource for pod in self.pods)
        total_band_usage = sum(pod.band_resource for pod in self.pods)

        # CPU 超负荷的情况下增加处理延迟 (按 alpha 系数)
        cpu_delay = 0
        if total_cpu_usage > self.cpu_capacity:
            cpu_overload = total_cpu_usage - self.cpu_capacity
            cpu_delay = alpha * cpu_overload  # 按比例增加的 CPU 处理延迟

        # 带宽竞争导致的传输延迟 (按 beta 系数)
        band_delay = 0
        if total_band_usage > self.band_capacity:
            band_overload = total_band_usage - self.band_capacity
            band_delay = beta * band_overload  # 按比例增加的网络传输延迟

        # 计算丢包率 (可选)
        drop_rate = 0
        if enable_drop and total_band_usage > self.band_capacity:
            drop_rate = (total_band_usage - self.band_capacity) / total_band_usage  # 丢包率

        # 计算抖动 (可选)
        jitter = 0
        if enable_jitter:
            jitter = band_delay * 0.1  # 假设抖动为延迟的 10%

        # 打印结果
        print(f"资源竞争导致的 CPU 延迟: {cpu_delay:.2f} ms")
        print(f"资源竞争导致的网络延迟: {band_delay:.2f} ms")
        if enable_drop:
            print(f"丢包率: {drop_rate:.2%}")
        if enable_jitter:
            print(f"网络抖动: {jitter:.2f} ms")



        return {
            "cpu_delay": cpu_delay,
            "band_delay": band_delay,
            "drop_rate": drop_rate if enable_drop else None,
            "jitter": jitter if enable_jitter else None
        }