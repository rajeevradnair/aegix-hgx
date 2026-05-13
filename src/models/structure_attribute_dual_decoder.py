import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

torch.manual_seed(42)


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

print("\nFeature matrix x shape:", x.shape)


source_ids = []
target_ids = []

for src, dst in G.edges():
    source_ids.append(node_to_id[src])
    target_ids.append(node_to_id[dst])

edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)

print("edge_index shape:", edge_index.shape)


adjacency_matrix = nx.to_numpy_array(G, nodelist=nodes)
a = torch.tensor(adjacency_matrix, dtype=torch.float)

print("Adjacency matrix shape:", a.shape)


data = Data(x=x, edge_index=edge_index)

print("\n================ DATA OBJECT ================")
print(data)


class StructureAttributeAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()

        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, embedding_dim)

        self.attribute_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    #Encoding node feature matrix through the two GCNConv layers
    def encode(self, x, edge_index):
        hidden = self.encoder_conv1(x, edge_index) #edge index is 2nd argument
        hidden = F.relu(hidden)

        z = self.encoder_conv2(hidden, edge_index) #edge index is 2nd argument

        return z #embedding

    def decode_attributes(self, z):
        #attribute decoder is essentially the forward pass in the reverse direction
        x_hat = self.attribute_decoder(z) 
        return x_hat

    def decode_structure(self, z):
        # *** Intuition is that if two nodes have compatible embeddings in the latent space, 
        # maybe there should be an edge between them. ***
        # The torch.sigmoid(z @ z.T) in GCN link prediction converts raw inner-product similarity scores 
        # (ranging from \(-\infty \) to \(+\infty \)) into a 0–1 probability space, effectively interpreting 
        # the output as the likelihood of an edge existing. 
        # While raw inner product isn't a perfect binary indicator, it acts as a differentiable approximation.
        a_hat = torch.sigmoid(z @ z.T)
        return a_hat

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        x_hat = self.decode_attributes(z)
        a_hat = self.decode_structure(z)

        # Return vector embedding, reconstructed node feature matrix, reconstructed adjacency matrix
        return z, x_hat, a_hat


input_dim = data.num_node_features
hidden_dim = 8
embedding_dim = 4

model = StructureAttributeAutoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ MODEL ================")
print(model)


print("\n================ TRAINING ================")

for epoch in range(5001):
    model.train()

    optimizer.zero_grad()

    z, x_hat, a_hat = model(data)

    attribute_loss = F.mse_loss(x_hat, x)
    structure_loss = F.mse_loss(a_hat, a)

    total_loss = attribute_loss + structure_loss

    total_loss.backward()
    optimizer.step()

    '''
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"total_loss={total_loss.item():.4f} | "
            f"attribute_loss={attribute_loss.item():.4f} | "
            f"structure_loss={structure_loss.item():.4f}"
        )
    '''

print("\n================ NODE ANOMALY SCORES ================")

model.eval()

with torch.no_grad():
    z, x_hat, a_hat = model(data)

    attribute_errors = torch.mean((x - x_hat) ** 2, dim=1)
    structure_errors = torch.mean((a - a_hat) ** 2, dim=1)

    alpha = 0.2
    anomaly_scores = alpha * structure_errors + (1 - alpha) * attribute_errors

# print(f"Anomaly score: {anomaly_scores}")

sorted_indices = torch.argsort(anomaly_scores, descending=True)

# print(f"Node indices descending order of anomaly score: {sorted_indices}")


for rank, idx in enumerate(sorted_indices):
    idx = idx.item()
    node_name = id_to_node[idx]

    print(
        f"rank={rank+1:2d} | "
        f"node={node_name:28s} | "
        f"anomaly score={anomaly_scores[idx].item():.6f} | "
        f"attr_error={attribute_errors[idx].item():.6f} | "
        f"struct_error={structure_errors[idx].item():.6f}"
    )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "node_embeddings": z,
        "node_to_id": node_to_id,
        "id_to_node": id_to_node,
        "anomaly_scores": anomaly_scores,
    },
    "structure_attribute_model.pt",
)

print("\nSaved model artifacts to structure_attribute_model.pt")
