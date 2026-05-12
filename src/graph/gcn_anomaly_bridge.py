import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Build my directed graph with node, node attributes, edge, edge attribute, node<>node relations
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

#Take a snapshot of the nodes in the graph
nodes = list(G.nodes())

# Create node_id <> node_name mappings
node_to_id = {node: idx for idx, node in enumerate(nodes)}
id_to_node = {idx: node for node, idx in node_to_id.items()}

print("\n================ NODE ORDERING ================")
for idx, node in id_to_node.items():
    print(f"{idx:2d}: {node}")

#Build node feature matrix from node attributes
#Only 2 features being considered 1) node type (one-hot encoded) and 2) risk score
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

#Build (2 x num_edges) dimension edge-index matrix from the graph edges
source_ids = []
target_ids = []

for src, dst in G.edges():
    source_ids.append(node_to_id[src])
    target_ids.append(node_to_id[dst])
# edge_index is a 2 x num_edges tensor where:
# - the first row contains source node IDs
# - the second row contains target node IDs
edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)

print("\n================ EDGE INDEX ================")
print("edge_index shape:", edge_index.shape)
print(edge_index)

#Determine labels (i.e. ground truth) for each node in nodes list based on whether they are anomalous or not
#Anomalous nodes are coded with 1
#Normal nodes are coded with 0

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


#Create PyG Data object from node feature matrix, edge index matrix and labels vector
data = Data(x=x, edge_index=edge_index, y=y)

print("\n================ PYG DATA OBJECT ================")
print(data)


#My simple GCN model with 3 layers - Conv -> ReLU -> Conv
#Layer 1: input dimension = number of features of each node, output dimension = hidden dimension
#Layer 2: input dimension = hidden dimension, output dimension = output dimension (2 for logits)
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

#Create the NN model, define the loss function and Adam optimizer
input_dim = data.num_node_features
hidden_dim = 8
output_dim = 2

model = SimpleGCN(input_dim, hidden_dim, output_dim)

#Since logits for each class are being returned from the model, we can use CrossEntropyLoss which combines LogSoftmax and NLLLoss in one single class. 
#Two logits returned where the fist logit corresponds to the normal class and the second logit corresponds to the anomalous class. 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ MODEL ================")
print(model)
print(model.conv1.lin.weight)


print("\n================ TRAINING ================")

#Train the model
for epoch in range(201):
    model.train()

    optimizer.zero_grad()

    logits = model(data) #pass the PyG Data object to the model's forward pass to get logits [normal_class, anomalous_class ] for each node

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

# Evaluate the model and print predictions, probabilities and hidden embeddings for each node
model.eval()

with torch.no_grad():
    logits, hidden_embedding = model(data, return_embeddings=True)
    #logits being converted into probabilities using softmax function where the first column corresponds to the normal class and the second column corresponds to the anomalous class
    probabilities = F.softmax(logits, dim=1)
    #The predicted class for each node is the one with the highest probability value (either normal or anomalous)
    predictions = probabilities.argmax(dim=1)

print(data.num_node_features, hidden_embedding.shape)

# Print node,node features, hidden embeddings for each node in the graph
print("\n================ NODE_FEATURES & NODE HIDDEN EMBEDDINGS ================")
for idx, node in enumerate(nodes):
    print("-" * 60)
    print(f"Node: {node}")
    print(f"Node feature:")
    print(f"  {data.x[idx]}")
    print(f"Hidden embedding:")
    print(f"  {hidden_embedding[idx]}")

# Print node,node features, hidden embeddings for each node in the graph
for idx in range(data.num_nodes):
    node_name = id_to_node[idx]
    true_label = data.y[idx].item()
    pred_label = predictions[idx].item()
    anomaly_probability = probabilities[idx, 1].item() # What is the probability of the node being anomalous according to the model?

    print(
        f"{idx:2d}: {node_name:28s} | "
        f"true={true_label} | pred={pred_label} | "
        f"anomaly_prob={anomaly_probability:.4f}"
    )


#Determine a manual contextual risk score for each node based on its own risk score and the average risk score of its outgoing neighbors (i.e. Successors).
print("\n================ MANUAL CONTEXTUAL RISK ================")

manual_contextual_risk = {}

for node in nodes:
    own_risk = G.nodes[node]["risk_score"]
    outgoing_neighbors = list(G.successors(node))

    if len(outgoing_neighbors) == 0:
        contextual_risk = own_risk #No outgoing neighbors, hence contextual risk is just the node's own risk score
    else:
        neighbor_risks = []

        for neighbor in outgoing_neighbors:
            neighbor_risk = G.nodes[neighbor]["risk_score"]
            neighbor_risks.append(neighbor_risk)

        avg_neighbor_risk = sum(neighbor_risks) / len(neighbor_risks) #Average risk score of the outgoing neighbors

        contextual_risk = 0.5 * own_risk + 0.5 * avg_neighbor_risk #Combine the node's own risk score and the average risk score of its outgoing neighbors to get a contextual risk score for the node

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
        f"own={own_risk:.2f} | "                    #print the node's own risk score
        f"context={contextual_risk:.3f} | "         #print the node's manual contextual risk score
        f"gcn_prob={anomaly_probability:.3f} | "    #print the node's anomaly probability according to the GCN model
        f"true={true_label} | pred={prediction}"    #print the node's true label and predicted label according to the GCN model
    )