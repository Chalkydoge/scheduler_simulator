import matplotlib.pyplot as plt
import networkx as nx

# 定义 Pod 名称和依赖关系顺序
pod_sequence = ["Producer", "Firewall", "NAT", "IDS", "Cache", "LB", "TC", "Receiver"]
pod_shapes = ["o", "*", "^", "D", "p", "s", "h", "d"]  # 各种形状：圆圈(空心)、星星、三角形、菱形、五边形、正方形、圆圈(实心)
solid_fill = [False, False, False, False, False, False, True, True]  # 最后两个是实心形状
pod_colors = ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']  # 空心/实心颜色

pods = []
dependencies = {}

# 生成 Pod 名称和依赖关系
for i in range(len(pod_sequence)):
    pod_name = f"{pod_sequence[i]}_1"
    pods.append(pod_name)

    # 生成依赖关系
    if i > 0:
        dependencies[pod_name] = [f"{pod_sequence[i - 1]}_1"]

# 创建有向图
G = nx.DiGraph()

# 添加节点和依赖关系
for pod in pods:
    G.add_node(pod)

for pod, dep in dependencies.items():
    G.add_edge(dep[0], pod)

# 绘制图形
plt.figure(figsize=(10, 8))

# 获取图中的节点位置
pos = nx.spring_layout(G)

# 根据 Pod 类型为每个节点分配不同的形状和颜色
for i, pod in enumerate(pods):
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[pod],
        node_shape=pod_shapes[i],
        node_color=pod_colors[i],
        edgecolors='black',  # 使用黑色边框
        node_size=1500,
        linewidths=2
    )

# 绘制边
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20, edge_color='gray')

# 添加标签
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# 显示图形
plt.axis('off')
plt.savefig("pod_dependency_graph.png")
