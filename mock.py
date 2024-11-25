import random
import yaml


def generate_random_topology(num_nodes=10):
    nodes = []
    rtt_table = {}
    pods = []
    dependencies = {}

    # Generate nodes
    for i in range(1, num_nodes + 1):
        node_name = f"Node_{i}"
        node = {
            "name": node_name,
            "cpu_capacity": random.randint(4, 32),  # Random CPU capacity between 4 and 32
            "mem_capacity": random.randint(8000, 64000),  # Random memory between 8GB to 64GB
            "band_capacity": random.randint(500, 2000)  # Random bandwidth between 500 and 2000
        }
        nodes.append(node)
        rtt_table[node_name] = {}

    # Generate RTT values
    for i in range(1, num_nodes + 1):
        from_node = f"Node_{i}"
        for j in range(1, num_nodes + 1):
            to_node = f"Node_{j}"
            if i == j:
                rtt_table[from_node][to_node] = 0  # RTT from a node to itself is 0
            else:
                rtt_table[from_node][to_node] = random.randint(5, 100)  # Random RTT between 5ms and 100ms

    # Generate pods (for simplicity, we'll generate 5 pods)
    for i in range(1, 6):
        pod_name = f"Pod_{chr(64 + i)}"  # Generates names like Pod_A, Pod_B, etc.
        pod = {
            "name": pod_name,
            "cpu_resource": random.randint(1, 4),  # Random CPU resource between 1 and 4
            "mem_resource": random.randint(512, 4096),  # Random memory between 512MB and 4GB
            "band_resource": random.randint(100, 500),  # Random bandwidth between 100 and 500
            "setup_time": random.randint(10, 60)  # Random setup time between 10 and 60
        }
        pods.append(pod)

    # Generate random dependencies (for simplicity, let's make Pod_B depend on Pod_A, etc.)
    for i in range(2, 6):
        dependent_pod = f"Pod_{chr(64 + i)}"
        dependencies[dependent_pod] = [f"Pod_{chr(64 + i - 1)}"]  # Pod_B depends on Pod_A, etc.

    # Choose a random strategy
    strategies = ["RandomSchedulingStrategy", "LeastResourceSchedulingStrategy", "DelayAwareSchedulingStrategy"]
    strategy = random.choice(strategies)

    # Create the final topology structure
    topology = {
        "nodes": nodes,
        "rtt_table": rtt_table,
        "pods": pods,
        "dependencies": dependencies,
        "strategy": strategy
    }

    return topology


def generate_cnf_chains():
    # 定义 Pod 名称和依赖关系顺序
    pod_sequence = ["Producer", "Firewall", "NAT", "IDS", "Cache", "LB", "Traffic_shaper", "Receiver"]

    pods = []
    dependencies = {}

    # 生成 pods
    for i, pod_type in enumerate(pod_sequence):
        pod_name = f"{pod_type}_1"  # 生成名称如 Producer_1, Firewall_1 等
        pod = {
            "name": pod_name,
            "cpu_resource": random.randint(1, 4),  # 随机 CPU 资源在 1 到 4 之间
            "mem_resource": random.randint(512, 4096),  # 随机内存资源在 512MB 到 4GB 之间
            "band_resource": random.randint(100, 500),  # 随机带宽在 100 到 500 之间
            "setup_time": random.randint(10, 60)  # 随机设置时间在 10 到 60 之间
        }
        pods.append(pod)

    # 生成依赖关系
    for i in range(1, len(pod_sequence)):
        dependent_pod = f"{pod_sequence[i]}_1"
        dependencies[pod_sequence[i - 1]] = [f"{dependent_pod}_1"]  # 每个 pod 依赖于前一个 pod

    # 输出 pods 和 dependencies
    print("Pods:", pods)
    print("Dependencies:", dependencies)


# Print the topology to the console
# print(yaml.dump(topology, default_flow_style=False))
if __name__ == '__main__':
    # Generate the random topology
    # topology = generate_random_topology(10)

    generate_cnf_chains()

    # Write to a YAML file
    # with open("data/random_topology.yaml", "w") as file:
    #     yaml.dump(topology, file)
