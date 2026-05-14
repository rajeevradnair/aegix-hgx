import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

torch.manual_seed(42)

# Various node types and nodes
users = ["alice", "bob"]

hosts = [
    "laptop_01",
    "laptop_02",
    "server_01",
]

processes = [
    "chrome.exe",
    "powershell.exe",
    "unknown.exe",
]

external_ips = [
    "external_ip_8.8.8.8",
    "external_ip_185.10.10.10",
]

files = [
    "readme.txt",
    "payroll.csv",
]

domains = [
    "normal-domain.com",
    "malicious-domain.biz",
]

name_lists = {
    "User": users,
    "Host": hosts,
    "Process": processes,
    "ExternalIP": external_ips,
    "File": files,
    "Domain": domains,
}

# print(name_lists)

user_to_id = {name: idx for idx, name in enumerate(users)}
host_to_id = {name: idx for idx, name in enumerate(hosts)}
process_to_id = {name: idx for idx, name in enumerate(processes)}
external_ip_to_id = {name: idx for idx, name in enumerate(external_ips)}
file_to_id = {name: idx for idx, name in enumerate(files)}
domain_to_id = {name: idx for idx, name in enumerate(domains)}

id_to_user = {idx: name for idx, name in enumerate(users)}
id_to_host = {idx: name for idx, name in enumerate(hosts)}
id_to_process = {idx: name for idx, name in enumerate(processes)}
id_to_external_ip = {idx: name for idx, name in enumerate(external_ips)}
id_to_file = {idx: name for idx, name in enumerate(files)}
id_to_domain = {idx: name for idx, name in enumerate(domains)}

# PyG HeteroData
data = HeteroData()

# Feature matrixes for each node type
data["User"].x = torch.tensor(
    [
        [0.10, 0.0],  # alice: risk_score, is_privileged
        [0.35, 0.0],  # bob
    ],
    dtype=torch.float,
)

data["Host"].x = torch.tensor(
    [
        [0.20, 0.0],  # laptop_01: risk_score, is_server
        [0.45, 0.0],  # laptop_02
        [0.70, 1.0],  # server_01
    ],
    dtype=torch.float,
)

data["Process"].x = torch.tensor(
    [
        [0.10, 1.0, 0.0],  # chrome.exe: risk_score, is_signed, is_dual_use
        [0.55, 1.0, 1.0],  # powershell.exe
        [0.95, 0.0, 0.0],  # unknown.exe
    ],
    dtype=torch.float,
)

data["ExternalIP"].x = torch.tensor(
    [
        [0.15, 1.0],  # known benign IP
        [0.98, 0.0],  # suspicious IP
    ],
    dtype=torch.float,
)

data["File"].x = torch.tensor(
    [
        [0.10, 0.0],  # readme.txt: low risk, not sensitive
        [0.90, 1.0],  # payroll.csv: high risk, sensitive
    ],
    dtype=torch.float,
)

data["Domain"].x = torch.tensor(
    [
        [0.10, 0.0],  # normal-domain.com: low risk, not new
        [0.99, 1.0],  # malicious-domain.biz: high risk, new domain
    ],
    dtype=torch.float,
)


# Edge index for each canonical edge type
def build_edge_index(edge_pairs, src_lookup, dest_lookup):
    src_ids = []
    dest_ids = []
    for src_name, dest_name in edge_pairs:
        src_ids.append(src_lookup[src_name])
        dest_ids.append(dest_lookup[dest_name])
    return torch.tensor([src_ids, dest_ids], dtype=torch.long)


data["User", "logs_into", "Host"].edge_index = build_edge_index(
    [("alice", "laptop_01"), ("bob", "laptop_02")],
    user_to_id,
    host_to_id,
)

data["User", "accesses", "Host"].edge_index = build_edge_index(
    [("bob", "server_01")],
    user_to_id,
    host_to_id,
)

data["User", "launches", "Process"].edge_index = build_edge_index(
    [("alice", "powershell.exe")],
    user_to_id,
    process_to_id,
)

data["Host", "runs", "Process"].edge_index = build_edge_index(
    [("laptop_01", "chrome.exe"), ("laptop_02", "unknown.exe")],
    host_to_id,
    process_to_id,
)

data["Process", "connects_to", "ExternalIP"].edge_index = build_edge_index(
    [
        ("chrome.exe", "external_ip_8.8.8.8"),
        ("powershell.exe", "external_ip_185.10.10.10"),
        ("unknown.exe", "external_ip_185.10.10.10"),
    ],
    process_to_id,
    external_ip_to_id,
)

data["Process", "resolves", "Domain"].edge_index = build_edge_index(
    [
        ("chrome.exe", "normal-domain.com"),
        ("unknown.exe", "malicious-domain.biz"),
    ],
    process_to_id,
    domain_to_id,
)

data["Process", "reads_file", "File"].edge_index = build_edge_index(
    [
        ("chrome.exe", "readme.txt"),
        ("powershell.exe", "payroll.csv"),
    ],
    process_to_id,
    file_to_id,
)

data["Process", "writes_file", "File"].edge_index = build_edge_index(
    [
        ("unknown.exe", "payroll.csv"),
    ],
    process_to_id,
    file_to_id,
)

data["Process", "deletes_file", "File"].edge_index = build_edge_index(
    [
        ("unknown.exe", "payroll.csv"),
    ],
    process_to_id,
    file_to_id,
)

data["Process", "encrypts_file", "File"].edge_index = build_edge_index(
    [
        ("unknown.exe", "payroll.csv"),
    ],
    process_to_id,
    file_to_id,
)

data["Process", "touches", "Host"].edge_index = build_edge_index(
    [("unknown.exe", "server_01")],
    process_to_id,
    host_to_id,
)


# Define the ML model
class TypedRelationReconstructionDualDecoderModel(nn.Module):
    def __init__(self, node_type_num_dimensions, edge_types, hidden_dim, embedding_dim):
        super().__init__()

        self.node_types = list(node_type_num_dimensions.keys())
        self.edge_types = list(edge_types)
        self.embedding_dim = embedding_dim

        # common encoder
        self.encoders = nn.ModuleDict()

        # Decoder #1: Attribute decoders, one per node_type
        self.attribute_decoders = nn.ModuleDict()
        # Embedding dimension & Hidden dimensions is the same for all node types
        for node_type, input_dim in node_type_num_dimensions.items():
            self.encoders[node_type] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.attribute_decoders[node_type] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        # Decoder 2: Relations decoders, for each canonical edge type
        self.relation_matrices = nn.ParameterDict()
        # Weight matrix for each canonical edge type
        for edge_type in edge_types:
            key = self.edge_type_to_key(edge_type)  # canonical edge type - string repr
            self.relation_matrices[key] = nn.Parameter(
                torch.randn(embedding_dim, embedding_dim) * 0.1
            )

    def edge_type_to_key(self, edge_type):
        src_type, relation, dst_type = edge_type
        return f"{src_type}__{relation}__{dst_type}"

    # encode each node in each node_type to the embedding space
    def encode(self, data):
        z_dict = {}

        for node_type in self.node_types:
            z_dict[node_type] = self.encoders[node_type](data[node_type].x)

        return z_dict

    # decode node feature matrix for each node type
    def decode_attributes(self, z_dict):
        x_hat_dict = {}

        for node_type in self.node_types:
            x_hat_dict[node_type] = self.attribute_decoders[node_type](
                z_dict[node_type]
            )

        return x_hat_dict

    # decode relation probability of all edges within a given canonical edge type
    #       node embedding vectors of node_types &
    #       relational weights of canonical edge type
    #       sigmoid (z_src @ W_relation @ z_dest.T)
    def decode_relation(self, edge_type, z_src, z_dest):
        key = self.edge_type_to_key(edge_type)
        W_relation = self.relation_matrices[key]
        predicted_probability_scores = torch.sigmoid(z_src @ W_relation @ z_dest.T)

        return predicted_probability_scores

    # Forward pass
    def forward(self, data):
        z_dict = self.encode(data)
        x_hat_dict = self.decode_attributes(z_dict)
        return z_dict, x_hat_dict


def build_relation_target_matrix(data, edge_type):
    src_type, relation, dst_type = edge_type

    num_src_nodes = data[src_type].x.shape[0]
    num_dst_nodes = data[dst_type].x.shape[0]

    target = torch.zeros((num_src_nodes, num_dst_nodes), dtype=torch.float)

    edge_index = data[edge_type].edge_index

    for edge_position in range(edge_index.shape[1]):
        src_id = edge_index[0, edge_position].item()
        dst_id = edge_index[1, edge_position].item()

        target[src_id, dst_id] = 1.0

    return target


node_type_num_dimensions = {
    node_type: data[node_type].x.shape[1] for node_type in data.node_types
}

# Instantiate the model & Adam optimizer
model = TypedRelationReconstructionDualDecoderModel(
    node_type_num_dimensions=node_type_num_dimensions,
    edge_types=data.edge_types,
    hidden_dim=8,
    embedding_dim=4,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ HETERODATA OBJECT ================")
print(data)

print("\nMetadata:")
print(data.metadata())

print("\n================ MODEL ================")
print(model)


positive_edge_weight = 5.0

print("\n================ TRAINING TYPED RECONSTRUCTION ================")
for epoch in range(1001):
    model.train()
    optimizer.zero_grad()
    z_dict, x_hat_dict = model(data)

    # attribute loss
    attribute_loss = 0.0
    for node_type in data.node_types:
        original_x_node_type = data[node_type].x
        reconstructed_x_node_type = x_hat_dict[node_type]
        attribute_loss = attribute_loss + F.mse_loss(
            reconstructed_x_node_type, original_x_node_type
        )
        

    # structural loss
    structural_loss = 0.0
    for edge_type in data.edge_types:
        src_type, relation, dest_type = edge_type
        z_src = z_dict[src_type]
        z_dest = z_dict[dest_type]
        #Predicted probability score of relation z_src -> relation -> z_dest
        predicted_probability_scores = model.decode_relation(edge_type=edge_type, z_src=z_src, z_dest=z_dest)

        ground_truth_matrix = build_relation_target_matrix(data, edge_type)

        weight_matrix = torch.ones_like(ground_truth_matrix)

        weight_matrix[ground_truth_matrix == 1.0] = positive_edge_weight

        relation_loss = torch.mean(
            weight_matrix * (ground_truth_matrix - predicted_probability_scores) ** 2
        )
        structural_loss = structural_loss + relation_loss


    # total loss
    total_loss = attribute_loss + structural_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"total_loss={total_loss.item():.6f} | "
            f"attribute_loss={attribute_loss.item():.6f} | "
            f"typed_structure_loss={structural_loss.item():.6f}"
        )

print("\n================ NODE FEATURE RECONSTRUCTION ERROR FOR EACH NODE_TYPE (SORTED DESCENDING) ================")

model.eval()
with torch.no_grad():
    z_dict, x_hat_dict = model(data)

    for node_type in data.node_types:
        original_x = data[node_type].x
        reconstructed_x = x_hat_dict[node_type]

        per_node_errors = torch.mean((original_x - reconstructed_x) ** 2, dim=1)

        print(f"\nNode type: {node_type}")

        node_names = name_lists[node_type]
        sorted_indices = torch.argsort(per_node_errors, descending=True)

        for idx in sorted_indices:
            idx = idx.item()
            print(
                f"  {node_names[idx]:30s} | "
                f"attribute_error={per_node_errors[idx].item():.6f}"
            )

print("\n================ TYPED EDGE RECONSTRUCTION REPORT ACROSS ALL EDGES FROM ALL CANONICAL EDGE TYPE ================")

edge_prediction_errors = []

with torch.no_grad():

    #Iterate over every canonical edge type
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        #Predict (decode) the probability for the specific canonical edge type
        predicted_matrix = model.decode_relation(edge_type, z_src, z_dst)

        src_names = name_lists[src_type]
        dst_names = name_lists[dst_type]

        #Fetch the edge index matrix for the canonical edge type
        edge_index = data[edge_type].edge_index
        #For each canonical edge type, iterate over every edge in the edge index matrix
        for edge_position in range(edge_index.shape[1]):

            edge_index = data[edge_type].edge_index

            #Ground truth: A confirmed edge src_id -> dest_id exists in the real world
            src_id = edge_index[0, edge_position].item()
            dst_id = edge_index[1, edge_position].item()

            #Check the model predicted probability of the same edge: src_id -> dest_id)
            predicted_probability = predicted_matrix[src_id, dst_id].item()

            #Calculate the error of 
            edge_prediction_error = (1.0 - predicted_probability) ** 2

            src_name = src_names[src_id]
            dst_name = dst_names[dst_id]
            edge_prediction_errors.append({
                "edge_type": edge_type,
                "src": src_name,
                "relation": relation,
                "dst": dst_name,
                "predicted_probability": predicted_probability,
                "edge_error": edge_prediction_error,
            })

#Sort all edges (from all canonical edge type)
edge_prediction_errors = sorted(
    edge_prediction_errors,
    key=lambda row: row["edge_error"],
    reverse=True,
)

#Display in the order of decreasing edge prediction error
for rank, row in enumerate(edge_prediction_errors, start=1):
    print(
        f"rank={rank:2d} | "
        f"({row['edge_type'][0]}, {row['relation']}, {row['edge_type'][2]}) | "
        f"{row['src']:25s} -[{row['relation']:12s}]-> {row['dst']:25s} | "
        f"pred_prob={row['predicted_probability']:.6f} | "
        f"edge_error={row['edge_error']:.6f}"
    )


print("\n================ ANALYST-STYLE TYPED ACTION EXPLANATIONS ================")

top_k = 5

for row in edge_prediction_errors[:top_k]:
    print(
        f"The typed action '{row['src']} {row['relation']} {row['dst']}' "
        f"was risky because the model evaluated the full relation "
        f"({row['edge_type'][0]}, {row['relation']}, {row['edge_type'][2]}) "
        f"and assigned it low probability "
        f"({row['predicted_probability']:.4f}), producing edge error "
        f"{row['edge_error']:.6f}."
    )