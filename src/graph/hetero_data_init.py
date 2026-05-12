import torch
from torch_geometric.data import HeteroData


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
    "payroll.csv",
]

domains = [
    "malicious-domain.biz",
]


user_to_id = {name: idx for idx, name in enumerate(users)}
host_to_id = {name: idx for idx, name in enumerate(hosts)}
process_to_id = {name: idx for idx, name in enumerate(processes)}
external_ip_to_id = {name: idx for idx, name in enumerate(external_ips)}
file_to_id = {name: idx for idx, name in enumerate(files)}
domain_to_id={name:idx for idx, name in enumerate(domains)}

print("\n================ LOCAL NODE IDS ================")
print("Users:", user_to_id)
print("Hosts:", host_to_id)
print("Processes:", process_to_id)
print("External IPs:", external_ip_to_id)
print("Files:", file_to_id)
print("Domains:", domain_to_id)

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
    [0.90, 1.0],
], dtype=torch.float)

data["Domain"].x = torch.tensor([
    [0.99, 1.0]
], dtype=torch.float)

def build_edge_index(edge_pairs, src_lookup, dst_lookup):
    source_ids = []
    target_ids = []

    for src_name, dst_name in edge_pairs:
        source_ids.append(src_lookup[src_name])
        target_ids.append(dst_lookup[dst_name])

    return torch.tensor([source_ids, target_ids], dtype=torch.long)


login_edges = [
    ("alice", "laptop_01"),
    ("bob", "laptop_02"),
]

data["User", "logs_into", "Host"].edge_index = build_edge_index(
    login_edges,
    user_to_id,
    host_to_id,
)


access_edges = [
    ("bob", "server_01"),
]

data["User", "accesses", "Host"].edge_index = build_edge_index(
    access_edges,
    user_to_id,
    host_to_id,
)


launch_edges = [
    ("alice", "powershell.exe"),
]

data["User", "launches", "Process"].edge_index = build_edge_index(
    launch_edges,
    user_to_id,
    process_to_id,
)


runs_edges = [
    ("laptop_01", "chrome.exe"),
    ("laptop_02", "unknown.exe"),
]

data["Host", "runs", "Process"].edge_index = build_edge_index(
    runs_edges,
    host_to_id,
    process_to_id,
)


connects_edges = [
    ("chrome.exe", "external_ip_8.8.8.8"),
    ("powershell.exe", "external_ip_185.10.10.10"),
    ("unknown.exe", "external_ip_185.10.10.10"),
]

data["Process", "connects_to", "ExternalIP"].edge_index = build_edge_index(
    connects_edges,
    process_to_id,
    external_ip_to_id,
)


writes_file_edges = [
    ("unknown.exe", "payroll.csv"),
]

data["Process", "writes_file", "File"].edge_index = build_edge_index(
    writes_file_edges,
    process_to_id,
    file_to_id,
)


touches_edges = [
    ("unknown.exe", "server_01"),
]

data["Process", "touches", "Host"].edge_index = build_edge_index(
    touches_edges,
    process_to_id,
    host_to_id,
)

resolve_edges = [
    ("unknown.exe", "malicious-domain.biz"),
]

data["Process", "resolves", "Domain"].edge_index = build_edge_index(
    resolve_edges,
    process_to_id,
    domain_to_id,
)


data["Process"].y = torch.tensor([
    0,
    0,
    1,
], dtype=torch.long)


print("\n================ HETERODATA OBJECT ================")
print(data)

print("\nMetadata:")
print(data.metadata())


print("\n================ NODE FEATURE SHAPES ================")

for node_type in data.node_types:
    print(f"{node_type:12s} x shape: {data[node_type].x.shape}")


print("\n================ EDGE INDEX SHAPES ================")

for edge_type in data.edge_types:
    print(f"{str(edge_type):50s} edge_index shape: {data[edge_type].edge_index.shape}")


torch.save(data, "hetero_cyber_graph.pt")

print("\nSaved heterogeneous graph to hetero_cyber_graph.pt")