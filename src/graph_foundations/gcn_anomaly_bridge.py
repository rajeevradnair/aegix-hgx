import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


G = nx.DiGraph()

G.add_node("alice", node_type="User", risk_score=0.10)
G.add_node("bob", node_type="User", risk_score=0.35)

G.add_node("laptop_01", node_type="Host", risk_score=0.20)
G.add_node("laptop_02", node_type="Host", risk_score=0.45)
G.add_node("server_01", node_type="Host", risk_score=0.70)

G.add_node("chrome.exe", node_type="Process", risk_score=0.10)
G.add_node("powershell.exe", node_type="Process", risk_score=0.55)
G.add_node("unknown.exe", node_type="Process", risk_score=0.95)

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

G.add_edge("bob", "server_01", relation="accesses")
G.add_edge("unknown.exe", "server_01", relation="touches")


nodes = list(G.nodes())

node_to_id = {node: idx for idx, node in enumerate(nodes)}
id_to_node = {idx: node for node, idx in node_to_id.items()}

print("\n================ NODE ORDERING ================")
for idx, node in id_to_node.items():
    print(f"{idx:2d}: {node}")


node_types = ["User", "Host", "Process", "ExternalIP", "File"]

def build_node_feature_vector(node_attrs):
    node_type = node_attrs["node_type"]
    risk_score = node_attrs["risk_score"]

    one_hot_type = []

    for current_type in node_types:
        if node_type == current_type:
            one_hot_type.append(1.0)
        else:
            one_hot_type.append(0.0)

    return one_hot_type + [risk_score]


feature_rows = []

for node in nodes:
    attrs = G.nodes[node]
    feature_vector = build_node_feature_vector(attrs)
    feature_rows.append(feature_vector)

x = torch.tensor(feature_rows, dtype=torch.float)

print("\n================ NODE FEATURES ================")
print("x shape:", x.shape)
print(x)


source_ids = []
target_ids = []

for src, dst in G.edges():
    source_ids.append(node_to_id[src])
    target_ids.append(node_to_id[dst])

edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)

print("\n================ EDGE INDEX ================")
print("edge_index shape:", edge_index.shape)
print(edge_index)


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

print("\n================ LABELS ================")
for idx, label in enumerate(y):
    print(f"{idx:2d}: {id_to_node[idx]:28s} label={label.item()}")


data = Data(x=x, edge_index=edge_index, y=y)

print("\n================ PYG DATA OBJECT ================")
print(data)


class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data, return_embeddings=False):
        x = data.x
        edge_index = data.edge_index

        hidden = self.conv1(x, edge_index)
        hidden = F.relu(hidden)
        logits = self.conv2(hidden, edge_index)
        if return_embeddings:
            return logits, hidden
        return logits


input_dim = data.num_node_features
hidden_dim = 8
output_dim = 2

model = SimpleGCN(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ MODEL ================")
print(model)
print(model.conv1.lin.weight)


print("\n================ TRAINING ================")

for epoch in range(201):
    model.train()

    optimizer.zero_grad()

    logits = model(data)

    loss = criterion(logits, data.y)

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        predictions = logits.argmax(dim=1)
        correct = (predictions == data.y).sum().item()
        accuracy = correct / data.num_nodes

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.2f}"
        )


print("\n================ FINAL PREDICTIONS ================")

model.eval()

with torch.no_grad():
    logits, hidden_embedding = model(data, return_embeddings=True)
    probabilities = F.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)

#print(logits)
#print(probabilities)
#print(data.y)
#print(predictions)
print(data.num_node_features, hidden_embedding.shape)

print("\n================ NODE_FEATURES & NODE HIDDEN EMBEDDINGS ================")
for idx, node in enumerate(nodes):
    print("-" * 60)
    print(f"Node: {node}")
    print(f"Node feature:")
    print(f"  {data.x[idx]}")
    print(f"Hidden embedding:")
    print(f"  {hidden_embedding[idx]}")

for idx in range(data.num_nodes):
    node_name = id_to_node[idx]
    true_label = data.y[idx].item()
    pred_label = predictions[idx].item()
    anomaly_probability = probabilities[idx, 1].item()

    print(
        f"{idx:2d}: {node_name:28s} | "
        f"true={true_label} | pred={pred_label} | "
        f"anomaly_prob={anomaly_probability:.4f}"
    )


print("\n================ MANUAL CONTEXTUAL RISK ================")

manual_contextual_risk = {}

for node in nodes:
    own_risk = G.nodes[node]["risk_score"]
    outgoing_neighbors = list(G.successors(node))

    if len(outgoing_neighbors) == 0:
        contextual_risk = own_risk
    else:
        neighbor_risks = []

        for neighbor in outgoing_neighbors:
            neighbor_risk = G.nodes[neighbor]["risk_score"]
            neighbor_risks.append(neighbor_risk)

        avg_neighbor_risk = sum(neighbor_risks) / len(neighbor_risks)

        contextual_risk = 0.5 * own_risk + 0.5 * avg_neighbor_risk

    manual_contextual_risk[node] = contextual_risk

    print(
        f"{node:30s} | "
        f"own_risk={own_risk:.2f} | "
        f"contextual_risk={contextual_risk:.3f}"
    )

print("\n================ RISK COMPARISON REPORT ================")

model.eval()

with torch.no_grad():
    logits, hidden_embeddings = model(data, return_embeddings=True)
    probabilities = F.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)


for idx, node in id_to_node.items():
    own_risk = G.nodes[node]["risk_score"]
    contextual_risk = manual_contextual_risk[node]
    anomaly_probability = probabilities[idx, 1].item()
    true_label = data.y[idx].item()
    prediction = predictions[idx].item()
    
    print(
        f"{node:30s} | "
        f"own={own_risk:.2f} | "
        f"context={contextual_risk:.3f} | "
        f"gcn_prob={anomaly_probability:.3f} | "
        f"true={true_label} | pred={prediction}"
    )