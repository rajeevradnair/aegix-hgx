import torch
import networkx as nx
import torch_geometric as tg
from torch_geometric.data import Data

G = nx.DiGraph()

G.add_node("alice", node_type="User", risk_score=0.10)
G.add_node("bob", node_type="User", risk_score=0.35)

G.add_node("laptop_01", node_type="Host", risk_score=0.20)
G.add_node("laptop_02", node_type="Host", risk_score=0.45)
G.add_node("server_01", node_type="Host", risk_score=0.70)

G.add_node("chrome.exe", node_type="Process", risk_score=0.10)
G.add_node("powershell.exe", node_type="Process", risk_score=0.55)
G.add_node("unknown.exe", node_type="Process", risk_score=0.95)

G.add_node("database_01", node_type="Host", risk_score=0.85)
G.add_node("external_ip_8.8.8.8", node_type="ExternalIP", risk_score=0.15)
G.add_node("external_ip_185.10.10.10", node_type="ExternalIP", risk_score=0.98)

G.add_node("payroll.csv", node_type="File", risk_score=0.90)

G.add_edge("alice", "laptop_01", relation="logs_into")
G.add_edge("laptop_01", "chrome.exe", relation="runs")
G.add_edge("chrome.exe", "external_ip_8.8.8.8", relation="connects_to")

G.add_edge("alice", "powershell.exe", relation="launches")
G.add_edge("powershell.exe", "external_ip_185.10.10.10", relation="connects_to")

G.add_edge("bob", "laptop_02", relation="logs_into")
G.add_edge("laptop_02", "unknown.exe", relation="runs")
G.add_edge("unknown.exe", "external_ip_185.10.10.10", relation="connects_to")
G.add_edge("unknown.exe", "payroll.csv", relation="writes_file")

G.add_edge("unknown.exe", "database_01", relation="queries")

G.add_edge("bob", "server_01", relation="accesses")
G.add_edge("unknown.exe", "server_01", relation="touches")

print("============ Id to Node ordering")
nodes = G.nodes()
node_to_id = {node: idx for idx, node in enumerate(nodes)}
id_to_node = {idx: node for idx, node in enumerate(nodes)}
for key, value in id_to_node.items():
    print(f"{key:2d}: {value}")

print(
    "\n============ Node features (One-hot encoded node_type is One-hot encoded followed by risk score (last column)) ============"
)
node_types = ["User", "Host", "Process", "ExternalIP", "File"]
print("Node Type One-hot Encoding: ", node_types)


def build_node_feature_vector(node_attr):
    node_type = node_attr["node_type"]
    risk_score = node_attr["risk_score"]
    one_hot_type = []

    for t in node_types:
        if node_type == t:
            one_hot_type.append(1.0)
        else:
            one_hot_type.append(0.0)

    return one_hot_type + [risk_score]


feature_rows = []
for node in list(nodes):
    feature_rows.append(build_node_feature_vector(G.nodes[node]))

x = torch.tensor(feature_rows, dtype=torch.float)
print(x)

print(
    "\n============ Edge index ============"
)
src_ids = []
dest_ids = []
for src, dest in G.edges():
    src_ids.append(node_to_id[src])
    dest_ids.append(node_to_id[dest])

edge_index=torch.tensor([src_ids, dest_ids], dtype=torch.long)

print(edge_index)

print(
    "\n============ Identify anomalous nodes ============"
)
anomalous_nodes = {
    "unknown.exe",
    "external_ip_185.10.10.10",
    "payroll.csv",
}

labels = []

for node in nodes:
    if node in anomalous_nodes:
        labels.append(1)
    else:
        labels.append(0)

y = torch.tensor(labels, dtype=torch.long)

print(y)

print(
    "\n============ Create pyg Data object ============"
)
data = Data(x=x, edge_index=edge_index, labels=y)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Node feature matrix shape: ", data.x.shape)
print("Edge index shape: ", data.edge_index.shape)
print("Labels shape: ", data.labels.shape)


from pathlib import Path
path = Path(__file__)
torch.save(data, path.parents[0] / "cyber_graph_data.pt")
print("\nSaved PyG graph object to cyber_graph_data.pt")