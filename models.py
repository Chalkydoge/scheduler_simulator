from collections import defaultdict, deque


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

