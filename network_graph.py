import yaml
import networkx as nx
import matplotlib.pyplot as plt

# Mocking the YAML content
yaml_content = """
nodes:
  - name: "Node_1"
    cpu_capacity: 16
    mem_capacity: 32000
    band_capacity: 1000
  - name: "Node_2"
    cpu_capacity: 8
    mem_capacity: 16000
    band_capacity: 500
  - name: "Node_3"
    cpu_capacity: 32
    mem_capacity: 64000
    band_capacity: 2000

rtt_table:
  Node_1:
    Node_1: 0
    Node_2: 10
    Node_3: 30
  Node_2:
    Node_1: 10
    Node_2: 0
    Node_3: 20
  Node_3:
    Node_1: 30
    Node_2: 20
    Node_3: 0
"""

# Simulating reading from YAML
config = yaml.safe_load(yaml_content)

# Creating a graph
G = nx.Graph()

# Adding nodes
for node in config['nodes']:
    G.add_node(node['name'].split('_')[1])

# Adding edges with RTT as weight
rtt_table = config['rtt_table']
for from_node, connections in rtt_table.items():
    for to_node, rtt in connections.items():
        if from_node != to_node and rtt > 0:  # avoid self-loop and invalid connections
            G.add_edge(from_node, to_node, weight=rtt)

# Set the layout of the graph
pos = nx.spring_layout(G)

# Custom options for a simpler graph style
options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}

# Drawing the simplified graph with custom options
plt.figure(figsize=(8, 6))
nx.draw_networkx(G, pos, **options)

# Drawing edge labels with RTT values
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20)

# Show the simplified plot
plt.title("Simple Network Topology")
plt.axis("off")
plt.show()
