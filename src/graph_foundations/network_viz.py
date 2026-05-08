"""
Aegis-HGX Day 3: Cybersecurity Graph Basics with NetworkX

Goal:
- Represent users, hosts, processes, external IPs, and files as graph nodes.
- Represent security actions as directed edges.
- Visualize the graph.
- Print nodes, edges, paths, degrees, and adjacency matrix.

Run:
    python network_viz.py
"""

import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. Create a directed graph
# ============================================================
# A directed graph means edges have direction.
# Example:
#     alice -> laptop_01
# means Alice logged into laptop_01.
# The reverse is not automatically true.

G = nx.DiGraph()


# ============================================================
# 2. Add nodes with cybersecurity types and attributes
# ============================================================

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
G.add_node("server_backup_01", node_type="Host", os="Linux", criticality="High")

# -----------------------------
# Processes
# -----------------------------
G.add_node("chrome.exe", node_type="Process", signed=True, reputation="Known")
G.add_node("powershell.exe", node_type="Process", signed=True, reputation="DualUse")
G.add_node("unknown.exe", node_type="Process", signed=False, reputation="Unknown")

# -----------------------------
# External IPs
# -----------------------------
G.add_node("external_ip_8.8.8.8", node_type="ExternalIP", country="US", reputation="Known")
G.add_node("external_ip_185.10.10.10", node_type="ExternalIP", country="Unknown", reputation="Suspicious")

# -----------------------------
# Sensitive file
# -----------------------------
G.add_node("payroll.csv", node_type="File", sensitivity="High")


# ============================================================
# 3. Add directed edges with relationship types
# ============================================================

# Normal-ish activity
G.add_edge("alice", "laptop_01", relation="logs_into", timestamp="09:00")
G.add_edge("laptop_01", "chrome.exe", relation="runs", timestamp="09:05")
G.add_edge("chrome.exe", "external_ip_8.8.8.8", relation="connects_to", timestamp="09:06")

# More suspicious activity
G.add_edge("alice", "powershell.exe", relation="launches", timestamp="10:10")
G.add_edge("powershell.exe", "external_ip_185.10.10.10", relation="connects_to", timestamp="10:11")

G.add_edge("bob", "laptop_02", relation="logs_into", timestamp="11:00")
G.add_edge("laptop_02", "unknown.exe", relation="runs", timestamp="11:03")
G.add_edge("unknown.exe", "external_ip_185.10.10.10", relation="connects_to", timestamp="11:04")
G.add_edge("unknown.exe", "payroll.csv", relation="writes_file", timestamp="11:06")

G.add_edge("unknown.exe", "server_backup_01", relation="touches", timestamp="11:13")


# Server access
G.add_edge("bob", "server_01", relation="accesses", timestamp="11:10")
G.add_edge("unknown.exe", "server_01", relation="touches", timestamp="11:12")


# ============================================================
# 4. Print basic graph summary
# ============================================================

print("\n================ GRAPH SUMMARY ================")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")


# ============================================================
# 5. Print nodes and node attributes
# ============================================================

print("\n================ NODES ================")

for node, attrs in G.nodes(data=True):
    print(f"{node:28s} | {attrs}")


# ============================================================
# 6. Print edges and edge attributes
# ============================================================

print("\n================ EDGES ================")

for src, dst, attrs in G.edges(data=True):
    print(f"{src:28s} -> {dst:28s} | {attrs}")


# ============================================================
# 7. Ask graph investigation questions
# ============================================================

print("\n================ INVESTIGATION QUESTIONS ================")

print("\nWhat does laptop_02 point to?")
print(list(G.successors("laptop_02")))

print("\nWhat points into server_01?")
print(list(G.predecessors("server_01")))

print("\nWhat did unknown.exe touch or connect to?")
print(list(G.successors("unknown.exe")))

print("\nWho or what points into external_ip_185.10.10.10?")
print(list(G.predecessors("external_ip_185.10.10.10")))


# ============================================================
# 8. Find suspicious paths
# ============================================================

print("\n================ SUSPICIOUS PATHS ================")

# Path from Bob to suspicious external IP
path_bob_to_ip = nx.shortest_path(
    G,
    source="bob",
    target="external_ip_185.10.10.10"
)

print("\nPath from bob to suspicious external IP:")
print(" -> ".join(path_bob_to_ip))

# Path from Bob to sensitive file
path_bob_to_file = nx.shortest_path(
    G,
    source="bob",
    target="payroll.csv"
)

print("\nPath from bob to sensitive file:")
print(" -> ".join(path_bob_to_file))

# Path from Alice to suspicious external IP
path_alice_to_ip = nx.shortest_path(
    G,
    source="alice",
    target="external_ip_185.10.10.10"
)

print("\nPath from alice to suspicious external IP:")
print(" -> ".join(path_alice_to_ip))


# ============================================================
# 9. Degree analysis
# ============================================================
# Degree is the number of connections a node has.
# In directed graphs:
# - in_degree = number of incoming edges
# - out_degree = number of outgoing edges

print("\n================ DEGREE ANALYSIS ================")

for node in G.nodes():
    print(
        f"{node:28s} | "
        f"in_degree={G.in_degree(node):2d} | "
        f"out_degree={G.out_degree(node):2d}"
    )


# ============================================================
# 10. Build adjacency matrix
# ============================================================

print("\n================ ADJACENCY MATRIX ================")

nodes = list(G.nodes())
adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

print("\nNode order:")
for idx, node in enumerate(nodes):
    print(f"{idx:2d}: {node}")

print("\nAdjacency matrix:")
print(adj_matrix)

print("\nInterpretation:")
print("If adjacency_matrix[i][j] = 1, then node_i points to node_j.")


# ============================================================
# 11. Visualize the graph
# ============================================================

# Layout controls where nodes appear visually.
pos = nx.spring_layout(G, seed=42)

# Assign colors by node type.
node_colors = []

for node, attrs in G.nodes(data=True):
    node_type = attrs["node_type"]

    if node_type == "User":
        node_colors.append("lightblue")
    elif node_type == "Host":
        node_colors.append("lightgreen")
    elif node_type == "Process":
        node_colors.append("orange")
    elif node_type == "ExternalIP":
        node_colors.append("salmon")
    elif node_type == "File":
        node_colors.append("violet")
    else:
        node_colors.append("gray")

plt.figure(figsize=(14, 9))

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=2200,
    edgecolors="black",
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle="->",
    arrowsize=22,
    width=2,
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold",
)

edge_labels = nx.get_edge_attributes(G, "relation")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
)

plt.title("Mock Cybersecurity Graph", fontsize=16)
plt.axis("off")
plt.tight_layout()

# Save the image so it can be committed to GitHub.
plt.savefig("/Users/rnair/projects/aegis-hgx/artifacts/visualizations/aegis_hgx_graph_foundations.png", dpi=200)

# Show the image locally.
plt.show()


print("\nGraph visualization saved as:")
print("aegis_hgx_graph_foundations.png")