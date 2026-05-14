from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


torch.manual_seed(42)


@dataclass
class EdgeRecord:
    src_type: str
    src_name: str
    relation: str
    dst_type: str
    dst_name: str
    label: int
    description: str


TRAIN_NODE_FEATURES = {
    "User": {
        "alice": [0.10, 0.0],
        "bob": [0.35, 0.0],
    },
    "Host": {
        "laptop_01": [0.20, 0.0],
        "laptop_02": [0.45, 0.0],
        "server_01": [0.70, 1.0],
    },
    "Process": {
        "chrome.exe": [0.10, 1.0, 0.0],
        "powershell.exe": [0.55, 1.0, 1.0],
    },
    "ExternalIP": {
        "external_ip_8.8.8.8": [0.15, 1.0],
        "external_ip_20.50.50.50": [0.20, 1.0],
    },
    "File": {
        "readme.txt": [0.10, 0.0],
        "quarterly_report.xlsx": [0.35, 0.0],
    },
    "Domain": {
        "normal-domain.com": [0.10, 0.0],
        "update.microsoft.com": [0.20, 0.0],
    },
}


TEST_NODE_FEATURES = {
    "User": {
        "alice": [0.10, 0.0],
        "bob": [0.35, 0.0],
    },
    "Host": {
        "laptop_01": [0.20, 0.0],
        "laptop_02": [0.45, 0.0],
        "server_01": [0.70, 1.0],
    },
    "Process": {
        "chrome.exe": [0.10, 1.0, 0.0],
        "powershell.exe": [0.55, 1.0, 1.0],
        "unknown.exe": [0.95, 0.0, 0.0],
    },
    "ExternalIP": {
        "external_ip_8.8.8.8": [0.15, 1.0],
        "external_ip_20.50.50.50": [0.20, 1.0],
        "external_ip_185.10.10.10": [0.98, 0.0],
    },
    "File": {
        "readme.txt": [0.10, 0.0],
        "quarterly_report.xlsx": [0.35, 0.0],
        "payroll.csv": [0.90, 1.0],
    },
    "Domain": {
        "normal-domain.com": [0.10, 0.0],
        "update.microsoft.com": [0.20, 0.0],
        "malicious-domain.biz": [0.99, 1.0],
    },
}


SCHEMA_EDGE_TYPES = [
    ("User", "logs_into", "Host"),
    ("User", "failed_login_to", "Host"),
    ("User", "accesses", "Host"),
    ("User", "launches", "Process"),

    ("Host", "runs", "Process"),

    ("Process", "connects_to", "ExternalIP"),
    ("Process", "resolves", "Domain"),

    ("Process", "reads_file", "File"),
    ("Process", "writes_file", "File"),
    ("Process", "deletes_file", "File"),
    ("Process", "encrypts_file", "File"),

    ("Process", "touches", "Host"),
]


RELATION_SEVERITY = {
    "logs_into": 0.30,
    "failed_login_to": 0.80,
    "accesses": 0.45,
    "launches": 0.50,
    "runs": 0.45,
    "connects_to": 0.65,
    "resolves": 0.55,
    "reads_file": 0.25,
    "writes_file": 0.70,
    "deletes_file": 0.90,
    "encrypts_file": 1.00,
    "touches": 0.75,
}


TRAIN_EDGES = [
    EdgeRecord("User", "alice", "logs_into", "Host", "laptop_01", 0, "Alice normal login"),
    EdgeRecord("User", "bob", "logs_into", "Host", "laptop_02", 0, "Bob normal login"),
    EdgeRecord("User", "bob", "accesses", "Host", "server_01", 0, "Bob normal server access"),

    EdgeRecord("User", "alice", "launches", "Process", "chrome.exe", 0, "Alice launches Chrome"),
    EdgeRecord("User", "bob", "launches", "Process", "powershell.exe", 0, "Bob launches PowerShell for admin task"),

    EdgeRecord("Host", "laptop_01", "runs", "Process", "chrome.exe", 0, "Laptop runs Chrome"),
    EdgeRecord("Host", "laptop_02", "runs", "Process", "powershell.exe", 0, "Laptop runs PowerShell"),

    EdgeRecord("Process", "chrome.exe", "connects_to", "ExternalIP", "external_ip_8.8.8.8", 0, "Chrome connects to known IP"),
    EdgeRecord("Process", "powershell.exe", "connects_to", "ExternalIP", "external_ip_20.50.50.50", 0, "PowerShell connects to known update IP"),

    EdgeRecord("Process", "chrome.exe", "resolves", "Domain", "normal-domain.com", 0, "Chrome resolves normal domain"),
    EdgeRecord("Process", "powershell.exe", "resolves", "Domain", "update.microsoft.com", 0, "PowerShell resolves update domain"),

    EdgeRecord("Process", "chrome.exe", "reads_file", "File", "readme.txt", 0, "Chrome reads low-risk file"),
    EdgeRecord("Process", "powershell.exe", "reads_file", "File", "quarterly_report.xlsx", 0, "PowerShell reads business report"),
]


TEST_EDGES = [
    EdgeRecord("User", "alice", "logs_into", "Host", "laptop_01", 0, "Normal repeated login"),
    EdgeRecord("User", "bob", "logs_into", "Host", "laptop_02", 0, "Normal repeated login"),
    EdgeRecord("User", "bob", "accesses", "Host", "server_01", 0, "Normal server access"),

    EdgeRecord("User", "alice", "launches", "Process", "chrome.exe", 0, "Normal Chrome launch"),
    EdgeRecord("User", "bob", "launches", "Process", "powershell.exe", 0, "Normal admin PowerShell launch"),

    EdgeRecord("Host", "laptop_01", "runs", "Process", "chrome.exe", 0, "Normal process execution"),
    EdgeRecord("Host", "laptop_02", "runs", "Process", "powershell.exe", 0, "Normal process execution"),

    EdgeRecord("Process", "chrome.exe", "connects_to", "ExternalIP", "external_ip_8.8.8.8", 0, "Normal known IP connection"),
    EdgeRecord("Process", "powershell.exe", "connects_to", "ExternalIP", "external_ip_20.50.50.50", 0, "Normal update IP connection"),

    EdgeRecord("Process", "chrome.exe", "resolves", "Domain", "normal-domain.com", 0, "Normal DNS resolution"),
    EdgeRecord("Process", "powershell.exe", "resolves", "Domain", "update.microsoft.com", 0, "Normal update DNS resolution"),

    EdgeRecord("Process", "chrome.exe", "reads_file", "File", "readme.txt", 0, "Normal low-risk file read"),
    EdgeRecord("Process", "powershell.exe", "reads_file", "File", "quarterly_report.xlsx", 0, "Normal report read"),

    EdgeRecord("User", "bob", "failed_login_to", "Host", "server_01", 1, "Repeated failed login to critical server"),
    EdgeRecord("Host", "laptop_02", "runs", "Process", "unknown.exe", 1, "Host runs unsigned unknown process"),

    EdgeRecord("Process", "unknown.exe", "connects_to", "ExternalIP", "external_ip_185.10.10.10", 1, "Unknown process connects to suspicious IP"),
    EdgeRecord("Process", "unknown.exe", "resolves", "Domain", "malicious-domain.biz", 1, "Unknown process resolves malicious domain"),

    EdgeRecord("Process", "unknown.exe", "reads_file", "File", "payroll.csv", 1, "Unknown process reads sensitive payroll file"),
    EdgeRecord("Process", "unknown.exe", "writes_file", "File", "payroll.csv", 1, "Unknown process writes sensitive payroll file"),
    EdgeRecord("Process", "unknown.exe", "deletes_file", "File", "payroll.csv", 1, "Unknown process deletes sensitive payroll file"),
    EdgeRecord("Process", "unknown.exe", "encrypts_file", "File", "payroll.csv", 1, "Unknown process encrypts sensitive payroll file"),
    EdgeRecord("Process", "unknown.exe", "touches", "Host", "server_01", 1, "Unknown process touches critical server"),

    EdgeRecord("Process", "powershell.exe", "connects_to", "ExternalIP", "external_ip_185.10.10.10", 1, "PowerShell connects to suspicious external IP"),
]


def build_name_lists(node_features):
    return {
        node_type: list(nodes.keys())
        for node_type, nodes in node_features.items()
    }


def build_id_maps(name_lists):
    return {
        node_type: {name: idx for idx, name in enumerate(names)}
        for node_type, names in name_lists.items()
    }


def build_empty_edge_index():
    return torch.empty((2, 0), dtype=torch.long)


def build_hetero_data(node_features, edge_records, schema_edge_types):
    name_lists = build_name_lists(node_features)
    id_maps = build_id_maps(name_lists)

    data = HeteroData()

    for node_type, nodes in node_features.items():
        feature_rows = []

        for node_name in name_lists[node_type]:
            feature_rows.append(nodes[node_name])

        data[node_type].x = torch.tensor(feature_rows, dtype=torch.float)

    grouped_edges = defaultdict(list)

    for edge in edge_records:
        edge_type = (edge.src_type, edge.relation, edge.dst_type)
        grouped_edges[edge_type].append(edge)

    for edge_type in schema_edge_types:
        src_type, relation, dst_type = edge_type
        edges = grouped_edges.get(edge_type, [])

        if not edges:
            data[edge_type].edge_index = build_empty_edge_index()
            continue

        source_ids = []
        target_ids = []

        for edge in edges:
            source_ids.append(id_maps[src_type][edge.src_name])
            target_ids.append(id_maps[dst_type][edge.dst_name])

        data[edge_type].edge_index = torch.tensor(
            [source_ids, target_ids],
            dtype=torch.long,
        )

    return data, name_lists, id_maps


train_data, train_name_lists, train_id_maps = build_hetero_data(
    TRAIN_NODE_FEATURES,
    TRAIN_EDGES,
    SCHEMA_EDGE_TYPES,
)

test_data, test_name_lists, test_id_maps = build_hetero_data(
    TEST_NODE_FEATURES,
    TEST_EDGES,
    SCHEMA_EDGE_TYPES,
)

print("\n================ TRAIN GRAPH METADATA ================")
print(train_data.metadata())

print("\n================ TEST GRAPH METADATA ================")
print(test_data.metadata())


class StaticTypedObserver(nn.Module):
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
    node_type: train_data[node_type].x.shape[1]
    for node_type in train_data.node_types
}

model = StaticTypedObserver(
    input_dims=input_dims,
    edge_types=SCHEMA_EDGE_TYPES,
    hidden_dim=16,
    embedding_dim=8,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n================ MODEL ================")
print(model)


def count_positive_edges_by_relation(data, schema_edge_types):
    counts = {}

    for edge_type in schema_edge_types:
        counts[edge_type] = data[edge_type].edge_index.shape[1]

    return counts


train_relation_positive_counts = count_positive_edges_by_relation(
    train_data,
    SCHEMA_EDGE_TYPES,
)


positive_edge_weight = 5.0

print("\n================ TRAINING ON NORMAL GRAPH ================")

for epoch in range(601):
    model.train()
    optimizer.zero_grad()

    z_dict, x_hat_dict = model(train_data)

    attribute_loss = 0.0

    for node_type in train_data.node_types:
        original_x = train_data[node_type].x
        reconstructed_x = x_hat_dict[node_type]

        attribute_loss = attribute_loss + F.mse_loss(
            reconstructed_x,
            original_x,
        )

    structure_loss = 0.0

    for edge_type in SCHEMA_EDGE_TYPES:
        src_type, relation, dst_type = edge_type

        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        predicted_matrix = model.decode_relation(z_src, z_dst, edge_type)
        target_matrix = build_relation_target_matrix(train_data, edge_type)

        weight_matrix = torch.ones_like(target_matrix)
        weight_matrix[target_matrix == 1.0] = positive_edge_weight

        relation_loss = torch.mean(
            weight_matrix * (target_matrix - predicted_matrix) ** 2
        )

        structure_loss = structure_loss + relation_loss

    total_loss = attribute_loss + structure_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"total_loss={total_loss.item():.6f} | "
            f"attribute_loss={attribute_loss.item():.6f} | "
            f"typed_structure_loss={structure_loss.item():.6f}"
        )


def get_node_risk(data, node_type, node_id):
    return data[node_type].x[node_id, 0].item()


def get_relation_novelty(edge_type, train_counts):
    return 1.0 if train_counts.get(edge_type, 0) == 0 else 0.0


def compute_typed_action_risk(
    edge_error,
    source_risk,
    destination_risk,
    relation_severity,
    relation_novelty,
):
    return (
        0.50 * edge_error
        + 0.15 * source_risk
        + 0.15 * destination_risk
        + 0.15 * relation_severity
        + 0.05 * relation_novelty
    )


print("\n================ SCORING TEST GRAPH ================")

model.eval()

edge_rows = []

with torch.no_grad():
    test_z_dict, test_x_hat_dict = model(test_data)

    for edge in TEST_EDGES:
        edge_type = (edge.src_type, edge.relation, edge.dst_type)

        src_id = test_id_maps[edge.src_type][edge.src_name]
        dst_id = test_id_maps[edge.dst_type][edge.dst_name]

        z_src = test_z_dict[edge.src_type]
        z_dst = test_z_dict[edge.dst_type]

        predicted_matrix = model.decode_relation(z_src, z_dst, edge_type)

        predicted_probability = predicted_matrix[src_id, dst_id].item()
        edge_error = (1.0 - predicted_probability) ** 2

        source_risk = get_node_risk(test_data, edge.src_type, src_id)
        destination_risk = get_node_risk(test_data, edge.dst_type, dst_id)
        severity = RELATION_SEVERITY.get(edge.relation, 0.50)
        novelty = get_relation_novelty(edge_type, train_relation_positive_counts)

        typed_action_risk = compute_typed_action_risk(
            edge_error=edge_error,
            source_risk=source_risk,
            destination_risk=destination_risk,
            relation_severity=severity,
            relation_novelty=novelty,
        )

        edge_rows.append({
            "edge_type": edge_type,
            "src_type": edge.src_type,
            "src": edge.src_name,
            "relation": edge.relation,
            "dst_type": edge.dst_type,
            "dst": edge.dst_name,
            "label": edge.label,
            "description": edge.description,
            "predicted_probability": predicted_probability,
            "edge_error": edge_error,
            "source_risk": source_risk,
            "destination_risk": destination_risk,
            "relation_severity": severity,
            "relation_novelty": novelty,
            "typed_action_risk": typed_action_risk,
        })


edge_rows = sorted(
    edge_rows,
    key=lambda row: row["typed_action_risk"],
    reverse=True,
)

print("\n================ RANKED TYPED ACTION RISK REPORT ================")

for rank, row in enumerate(edge_rows, start=1):
    label_text = "SUSPICIOUS" if row["label"] == 1 else "NORMAL"

    print(
        f"rank={rank:2d} | "
        f"label={label_text:10s} | "
        f"{row['src']:25s} -[{row['relation']:15s}]-> {row['dst']:25s} | "
        f"pred_prob={row['predicted_probability']:.4f} | "
        f"edge_error={row['edge_error']:.4f} | "
        f"src_risk={row['source_risk']:.2f} | "
        f"dst_risk={row['destination_risk']:.2f} | "
        f"severity={row['relation_severity']:.2f} | "
        f"novelty={row['relation_novelty']:.1f} | "
        f"typed_action_risk={row['typed_action_risk']:.4f}"
    )


labels = [row["label"] for row in edge_rows]
scores = [row["typed_action_risk"] for row in edge_rows]
edge_error_scores = [row["edge_error"] for row in edge_rows]

print("\n================ EVALUATION METRICS ================")

if SKLEARN_AVAILABLE and len(set(labels)) == 2:
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    roc_auc_edge_only = roc_auc_score(labels, edge_error_scores)
    pr_auc_edge_only = average_precision_score(labels, edge_error_scores)

    print(f"Typed action risk ROC-AUC: {roc_auc:.4f}")
    print(f"Typed action risk PR-AUC:  {pr_auc:.4f}")
    print(f"Edge-error-only ROC-AUC:   {roc_auc_edge_only:.4f}")
    print(f"Edge-error-only PR-AUC:    {pr_auc_edge_only:.4f}")
else:
    print("scikit-learn is not installed or labels contain only one class.")
    print("To enable ROC-AUC and PR-AUC:")
    print("pip install scikit-learn")


print("\n================ NODE RISK FROM TEST ACTIONS ================")

node_action_risk = defaultdict(float)
node_supporting_edges = defaultdict(list)

for row in edge_rows:
    src_key = f"{row['src_type']}::{row['src']}"
    dst_key = f"{row['dst_type']}::{row['dst']}"

    for node_key in [src_key, dst_key]:
        if row["typed_action_risk"] > node_action_risk[node_key]:
            node_action_risk[node_key] = row["typed_action_risk"]

        node_supporting_edges[node_key].append(row)

ranked_nodes = sorted(
    node_action_risk.items(),
    key=lambda item: item[1],
    reverse=True,
)

for rank, (node_key, risk_score) in enumerate(ranked_nodes, start=1):
    print(
        f"rank={rank:2d} | "
        f"{node_key:45s} | "
        f"max_typed_action_risk={risk_score:.4f}"
    )


print("\n================ ANALYST EXPLANATIONS ================")

top_k = 8

for row in edge_rows[:top_k]:
    label_text = "injected suspicious action" if row["label"] == 1 else "normal action"

    print(
        f"- {row['src']} {row['relation']} {row['dst']} "
        f"ranked high as a {label_text}. "
        f"Reason: model surprise={row['edge_error']:.4f}, "
        f"source risk={row['source_risk']:.2f}, "
        f"destination risk={row['destination_risk']:.2f}, "
        f"relation severity={row['relation_severity']:.2f}, "
        f"relation novelty={row['relation_novelty']:.1f}. "
        f"Context: {row['description']}."
    )

'''
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "train_relation_positive_counts": train_relation_positive_counts,
        "edge_rows": edge_rows,
        "node_action_risk": dict(node_action_risk),
        "input_dims": input_dims,
        "schema_edge_types": SCHEMA_EDGE_TYPES,
    },
    "observer_static_detector_model.pt",
)

print("\nSaved observer artifacts to observer_static_detector_model.pt")
'''