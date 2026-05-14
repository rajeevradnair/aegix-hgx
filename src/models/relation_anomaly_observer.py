import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch_geometric.data import HeteroData


torch.manual_seed(42)


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


user_to_id = {name: idx for idx, name in enumerate(users)}
host_to_id = {name: idx for idx, name in enumerate(hosts)}
process_to_id = {name: idx for idx, name in enumerate(processes)}
external_ip_to_id = {name: idx for idx, name in enumerate(external_ips)}
file_to_id = {name: idx for idx, name in enumerate(files)}
domain_to_id = {name: idx for idx, name in enumerate(domains)}

name_lists = {
    "User": users,
    "Host": hosts,
    "Process": processes,
    "ExternalIP": external_ips,
    "File": files,
    "Domain": domains,
}


data = HeteroData()

data["User"].x = torch.tensor([
    [0.10, 0.0],
    [0.35, 0.0],
], dtype=torch.float)

data["Host"].x = torch.tensor([
    [0.20, 0.0],
    [0.45, 0.0],
    [0.70, 1.0],
], dtype=torch.float)

data["Process"].x = torch.tensor([
    [0.10, 1.0, 0.0],
    [0.55, 1.0, 1.0],
    [0.95, 0.0, 0.0],
], dtype=torch.float)

data["ExternalIP"].x = torch.tensor([
    [0.15, 1.0],
    [0.98, 0.0],
], dtype=torch.float)

data["File"].x = torch.tensor([
    [0.10, 0.0],
    [0.90, 1.0],
], dtype=torch.float)

data["Domain"].x = torch.tensor([
    [0.10, 0.0],
    [0.99, 1.0],
], dtype=torch.float)


def build_edge_index(edge_pairs, src_lookup, dst_lookup):
    source_ids = []
    target_ids = []

    for src_name, dst_name in edge_pairs:
        source_ids.append(src_lookup[src_name])
        target_ids.append(dst_lookup[dst_name])

    return torch.tensor([source_ids, target_ids], dtype=torch.long)


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


relation_severity = {
    "logs_into": 0.30,
    "accesses": 0.45,
    "launches": 0.50,
    "runs": 0.40,
    "connects_to": 0.65,
    "resolves": 0.55,
    "reads_file": 0.25,
    "writes_file": 0.70,
    "deletes_file": 0.90,
    "encrypts_file": 1.00,
    "touches": 0.75,
}


class TypedRelationObserver(nn.Module):
    def __init__(self, input_dims, edge_types, hidden_dim, embedding_dim):
        super().__init__()

        self.node_types = list(input_dims.keys())
        self.edge_types = list(edge_types)

        self.encoders = nn.ModuleDict()
        self.attribute_decoders = nn.ModuleDict()
        self.relation_matrices = nn.ParameterDict()

        for node_type, input_dim in input_dims.items():
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

        for edge_type in self.edge_types:
            key = self.edge_type_to_key(edge_type)
            self.relation_matrices[key] = nn.Parameter(
                torch.randn(embedding_dim, embedding_dim) * 0.1
            )

    def edge_type_to_key(self, edge_type):
        src_type, relation, dst_type = edge_type
        return f"{src_type}__{relation}__{dst_type}"

    def encode(self, data):
        z_dict = {}

        for node_type in self.node_types:
            z_dict[node_type] = self.encoders[node_type](data[node_type].x)

        return z_dict

    def decode_attributes(self, z_dict):
        x_hat_dict = {}

        for node_type in self.node_types:
            x_hat_dict[node_type] = self.attribute_decoders[node_type](
                z_dict[node_type]
            )

        return x_hat_dict

    def decode_relation(self, z_src, z_dst, edge_type):
        key = self.edge_type_to_key(edge_type)
        W_relation = self.relation_matrices[key]

        return torch.sigmoid(z_src @ W_relation @ z_dst.T)

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


input_dims = {
    node_type: data[node_type].x.shape[1]
    for node_type in data.node_types
}

model = TypedRelationObserver(
    input_dims=input_dims,
    edge_types=data.edge_types,
    hidden_dim=8,
    embedding_dim=4,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ GRAPH METADATA ================")
print(data.metadata())

print("\n================ MODEL ================")
print(model)


positive_edge_weight = 5.0

print("\n================ TRAINING OBSERVER ================")

for epoch in range(401):
    model.train()
    optimizer.zero_grad()

    z_dict, x_hat_dict = model(data)

    attribute_loss = 0.0

    for node_type in data.node_types:
        original_x = data[node_type].x
        reconstructed_x = x_hat_dict[node_type]

        attribute_loss = attribute_loss + F.mse_loss(
            reconstructed_x,
            original_x,
        )

    structure_loss = 0.0

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        predicted_matrix = model.decode_relation(z_src, z_dst, edge_type)
        target_matrix = build_relation_target_matrix(data, edge_type)

        weight_matrix = torch.ones_like(target_matrix)
        weight_matrix[target_matrix == 1.0] = positive_edge_weight

        relation_loss = torch.mean(
            weight_matrix * (target_matrix - predicted_matrix) ** 2
        )

        structure_loss = structure_loss + relation_loss

    total_loss = attribute_loss + structure_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"total_loss={total_loss.item():.6f} | "
            f"attribute_loss={attribute_loss.item():.6f} | "
            f"typed_structure_loss={structure_loss.item():.6f}"
        )


def get_node_risk(data, node_type, node_id):
    return data[node_type].x[node_id, 0].item()


print("\n================ TYPED ACTION RISK REPORT ================")

model.eval()

edge_rows = []

with torch.no_grad():
    z_dict, x_hat_dict = model(data)

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        predicted_matrix = model.decode_relation(z_src, z_dst, edge_type)
        edge_index = data[edge_type].edge_index

        src_names = name_lists[src_type]
        dst_names = name_lists[dst_type]

        for edge_position in range(edge_index.shape[1]):
            src_id = edge_index[0, edge_position].item()
            dst_id = edge_index[1, edge_position].item()

            src_name = src_names[src_id]
            dst_name = dst_names[dst_id]

            predicted_probability = predicted_matrix[src_id, dst_id].item()
            edge_error = (1.0 - predicted_probability) ** 2

            source_risk = get_node_risk(data, src_type, src_id)
            destination_risk = get_node_risk(data, dst_type, dst_id)
            severity = relation_severity.get(relation, 0.50)

            typed_action_risk = (
                0.50 * edge_error
                + 0.20 * source_risk
                + 0.20 * destination_risk
                + 0.10 * severity
            )

            edge_rows.append({
                "edge_type": edge_type,
                "src_type": src_type,
                "src": src_name,
                "relation": relation,
                "dst_type": dst_type,
                "dst": dst_name,
                "predicted_probability": predicted_probability,
                "edge_error": edge_error,
                "source_risk": source_risk,
                "destination_risk": destination_risk,
                "relation_severity": severity,
                "typed_action_risk": typed_action_risk,
            })

edge_rows = sorted(
    edge_rows,
    key=lambda row: row["typed_action_risk"],
    reverse=True,
)

for rank, row in enumerate(edge_rows, start=1):
    print(
        f"rank={rank:2d} | "
        f"{row['src']:25s} -[{row['relation']:13s}]-> {row['dst']:25s} | "
        f"pred_prob={row['predicted_probability']:.4f} | "
        f"edge_error={row['edge_error']:.4f} | "
        f"src_risk={row['source_risk']:.2f} | "
        f"dst_risk={row['destination_risk']:.2f} | "
        f"severity={row['relation_severity']:.2f} | "
        f"typed_action_risk={row['typed_action_risk']:.4f}"
    )


print("\n================ NODE RISK FROM TYPED ACTIONS ================")

node_action_risk = defaultdict(float)

for row in edge_rows:
    src_key = f"{row['src_type']}::{row['src']}"
    dst_key = f"{row['dst_type']}::{row['dst']}"

    node_action_risk[src_key] = max(
        node_action_risk[src_key],
        row["typed_action_risk"],
    )

    node_action_risk[dst_key] = max(
        node_action_risk[dst_key],
        row["typed_action_risk"],
    )

ranked_nodes = sorted(
    node_action_risk.items(),
    key=lambda item: item[1],
    reverse=True,
)

for rank, (node_key, risk_score) in enumerate(ranked_nodes, start=1):
    print(
        f"rank={rank:2d} | "
        f"{node_key:40s} | "
        f"max_typed_action_risk={risk_score:.4f}"
    )


print("\n================ ANALYST EXPLANATIONS ================")

top_k = 5

for row in edge_rows[:top_k]:
    print(
        f"- The action '{row['src']} {row['relation']} {row['dst']}' "
        f"ranked high because it combines model surprise "
        f"(edge_error={row['edge_error']:.4f}), source risk "
        f"({row['source_risk']:.2f}), destination risk "
        f"({row['destination_risk']:.2f}), and relation severity "
        f"({row['relation_severity']:.2f})."
    )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "z_dict": z_dict,
        "input_dims": input_dims,
        "edge_rows": edge_rows,
        "node_action_risk": dict(node_action_risk),
        "name_lists": name_lists,
    },
    "relation_anomaly_observer_model.pt",
)

print("\nSaved observer artifacts to relation_anomaly_observer_model.pt")