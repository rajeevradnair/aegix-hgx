"""
Goal:
- Build a mock cybersecurity graph.
- Convert it into:
    A = adjacency matrix
    X = node feature matrix
- Print and explain both matrices.
- Save them as .npy files for future GNN work.

Run:
    python adjacency_matrix.py
"""

import numpy as np
import networkx as nx

# ============================================================
# 1. Build the same cybersecurity graph from Day 3
# ============================================================

G = nx.DiGraph()

# -----------------------------
# Users
# -----------------------------
G.add_node("alice", node_type="User", risk_score=0.10, department="Engineering")
G.add_node("bob", node_type="User", risk_score=0.35, department="Finance")

# -----------------------------
# Hosts
# -----------------------------
G.add_node("laptop_01", node_type="Host", os="Windows", criticality="Medium")
G.add_node("laptop_02", node_type="Host", os="Windows", criticality="Medium")
G.add_node("server_01", node_type="Host", os="Linux", criticality="High")

# -----------------------------
# Processes
# -----------------------------
G.add_node("chrome.exe", node_type="Process", signed=True, reputation="Known")
G.add_node("powershell.exe", node_type="Process", signed=True, reputation="DualUse")
G.add_node("unknown.exe", node_type="Process", signed=False, reputation="Unknown")

# -----------------------------
# External IPs
# -----------------------------
G.add_node(
    "external_ip_8.8.8.8", node_type="ExternalIP", country="US", reputation="Known"
)
G.add_node(
    "external_ip_185.10.10.10",
    node_type="ExternalIP",
    country="Unknown",
    reputation="Suspicious",
)

# -----------------------------
# File
# -----------------------------
G.add_node("payroll.csv", node_type="File", sensitivity="High")


# Edges
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


# ============================================================
# 2. Fix node order
# ============================================================
# Matrix rows/columns depend on node order.
# We make the order explicit so interpretation is stable.

nodes = list(G.nodes())

print("\n================ NODE ORDER ================")
for idx, node in enumerate(nodes):
    print(f"{idx:2d}: {node}")


# ============================================================
# 3. Build adjacency matrix A
# ============================================================
# A[i][j] = 1 means nodes[i] -> nodes[j]

A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)

print("\n================ Build Adjacency Matrix A ================")
print(A)

print("\nA shape:", A.shape)
print("A[i][j] = 1 means node_i points to node_j.")


# ============================================================
# 4. Manually verify a few edges
# ============================================================


def explain_edge(src, dst):
    src_idx = nodes.index(src)
    dst_idx = nodes.index(dst)

    value = A[src_idx][dst_idx]

    print(f"A[{src_idx}][{dst_idx}] = {value:.0f} " f"means {src} -> {dst}")


print("\n================ Verify a few edges ================")
explain_edge("bob", "laptop_02")
explain_edge("laptop_02", "unknown.exe")
explain_edge("unknown.exe", "payroll.csv")
explain_edge("powershell.exe", "external_ip_185.10.10.10")


# ============================================================
# 5. Build node feature matrix X
# ============================================================
# We need numeric features.
# For now, we use simple hand-coded features:
#
# [is_user,
#  is_host,
#  is_process,
#  is_external_ip,
#  is_file,
#  risk_score,
#  is_signed,
#  is_sensitive]

feature_names = [
    "is_user",
    "is_host",
    "is_process",
    "is_external_ip",
    "is_file",
    "risk_score",
    "is_signed",
    "is_sensitive",
]


def get_features_for_node(node, attrs):
    node_type = attrs.get("node_type")

    is_user = 1.0 if node_type == "User" else 0.0
    is_host = 1.0 if node_type == "Host" else 0.0
    is_process = 1.0 if node_type == "Process" else 0.0
    is_external_ip = 1.0 if node_type == "ExternalIP" else 0.0
    is_file = 1.0 if node_type == "File" else 0.0

    # Risk score defaults by type and attributes.
    risk_score = 0.0

    if node_type == "User":
        risk_score = attrs.get("risk_score", 0.0)

    elif node_type == "Host":
        criticality = attrs.get("criticality", "Low")
        if criticality == "High":
            risk_score = 0.70
        elif criticality == "Medium":
            risk_score = 0.40
        else:
            risk_score = 0.20

    elif node_type == "Process":
        reputation = attrs.get("reputation", "Known")
        signed = attrs.get("signed", True)

        if reputation == "Unknown":
            risk_score = 0.90
        elif reputation == "DualUse":
            risk_score = 0.60
        else:
            risk_score = 0.20

        if signed is False:
            risk_score += 0.10

        risk_score = min(risk_score, 1.0)

    elif node_type == "ExternalIP":
        reputation = attrs.get("reputation", "Known")
        if reputation == "Suspicious":
            risk_score = 0.95
        else:
            risk_score = 0.20

    elif node_type == "File":
        sensitivity = attrs.get("sensitivity", "Low")
        if sensitivity == "High":
            risk_score = 0.85
        else:
            risk_score = 0.20

    is_signed = 1.0 if attrs.get("signed", False) is True else 0.0
    is_sensitive = 1.0 if attrs.get("sensitivity", "Low") == "High" else 0.0

    return [
        is_user,
        is_host,
        is_process,
        is_external_ip,
        is_file,
        risk_score,
        is_signed,
        is_sensitive,
    ]


print("\n================ Revisiting the Graph ================")
print(G.nodes())
print(G.edges())

print("\n================ Build the Node feature matrix X ================")
#print("Features for the nodes:")
#print(f"{' ':25s}", feature_names)
X_rows = []
for node in nodes:
    X_rows.append(get_features_for_node(node, G.nodes[node]))
    #print(f"{node:25s}", X_rows[-1])

X=np.array(X_rows, dtype=np.float32)
print(X.shape, "\n", X)

print("\n================ Readable node features ================")

for node_idx, node in enumerate(nodes):
    print(f"\nNode {node_idx}: {node}")
    for feature_idx, feature_name in enumerate(feature_names):
        print(f"  {feature_name:15s}: {X[node_idx][feature_idx]:.2f}")


# ============================================================
# 7. Compute simple graph statistics
# ============================================================

print("\n================ Simple graph statistics ================")

for node in nodes:
    print(
        f"{node:28s} | "
        f"in_degree={G.in_degree(node):2d} | "
        f"out_degree={G.out_degree(node):2d} | "
        f"total_degree={G.degree(node):2d}"
    )

# ============================================================
# 8. Identify risky nodes using simple rules
# ============================================================

print("\n================ Simple risk check ================")

risk_col = feature_names.index("risk_score")
for node_idx, node in enumerate(nodes):
    risk = X[node_idx][risk_col]
    if risk >= 0.80:
        print(f"High-risk node: {node:28s} | risk_score={risk:.2f}")

# ============================================================
# 9. Save A and X for future GNN work
# ============================================================
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

np.save(os.path.join(script_dir, "out_adjacency_matrix_A.npy"), A)
np.save(os.path.join(script_dir, "out_node_feature_matrix_X.npy"), X)


file_path=os.path.join(script_dir, "out_node_order.txt")
with open(file_path, "w") as f:
    for idx, node in enumerate(nodes):
        f.write(f"{idx}: {node}\n")

file_path=os.path.join(script_dir, "out_feature_names.txt")
with open(file_path, "w") as f:
    for idx, feature in enumerate(feature_names):
        f.write(f"{idx}: {feature}\n")

print("\nSaved files:")
print("out_adjacency_matrix_A.npy")
print("out_node_feature_matrix_X.npy")
print("out_node_order.txt")
print("out_feature_names.txt")
