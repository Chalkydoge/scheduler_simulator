from flask import Flask, render_template, jsonify, request
from schedule import SchedulerService
import yaml
import networkx as nx

app = Flask(__name__)


# 加载 YAML 文件，解析为节点和边信息
def load_topology_from_yaml():
    with open("data/random_topology.yaml", "r") as file:
        config = yaml.safe_load(file)

    G = nx.Graph()

    # 添加节点
    for node in config["nodes"]:
        G.add_node(node["name"])

    # 添加边和 RTT
    rtt_table = config["rtt_table"]
    for from_node, connections in rtt_table.items():
        for to_node, rtt in connections.items():
            if from_node != to_node and rtt > 0:
                G.add_edge(from_node, to_node, weight=rtt)

    return G


# 将图数据转换为前端可以解析的 JSON 格式
def graph_to_json(G):
    nodes = [{"id": node} for node in G.nodes()]
    edges = [{"source": u, "target": v, "weight": G[u][v]['weight']} for u, v in G.edges()]
    return {"nodes": nodes, "edges": edges}


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/handle_selection', methods=['POST'])
def handle_selection():
    # 从前端接收数据
    data = request.json
    # 输出接收到的数据到控制台，以确认收到了数据
    print(f"Received data: {data}")

    selected_option = data.get('selected_option')

    # 初始化调度服务
    scheduler_service = SchedulerService('data/random_topology.yaml')

    # 处理接收到的选项，比如根据选项返回不同的结果
    if selected_option == '1':
        strategy_name = "RandomSchedulingStrategy"
    elif selected_option == '2':
        strategy_name = "LeastResourceSchedulingStrategy"
    elif selected_option == '3':
        strategy_name = "DelayAwareSchedulingStrategy"
    else:
        strategy_name = None

    # 获取调度的指标
    metrics = scheduler_service.print_scheduler_metrics(strategy_name=strategy_name)

    # 返回 JSON 响应
    return jsonify({'response_time': metrics['response_time']})


@app.route('/graph')
def get_graph():
    G = load_topology_from_yaml()
    graph_json = graph_to_json(G)
    return jsonify(graph_json)


if __name__ == '__main__':
    app.run(debug=True)