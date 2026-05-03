import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# -----------------------------
# Add users
# -----------------------------
G.add_node("alice", node_type="User", risk_score=0.1)
G.add_node("bob", node_type="User", risk_score=0.2)

# -----------------------------
# Add hosts
# -----------------------------
G.add_node("laptop_01", node_type="Host", os="Windows")
G.add_node("laptop_02", node_type="Host", os="Windows")
G.add_node("server_01", node_type="Host", os="Linux")

# -----------------------------
# Add processes
# -----------------------------
G.add_node("chrome.exe", node_type="Process", signed=True)
G.add_node("powershell.exe", node_type="Process", signed=True)
G.add_node("unknown.exe", node_type="Process", signed=False)

# -----------------------------
# Add external IPs
# -----------------------------
G.add_node("external_ip_8.8.8.8", node_type="ExternalIP", country="US")
G.add_node("external_ip_185.10.10.10", node_type="ExternalIP", country="Unknown")

# -----------------------------
# Add files
# -----------------------------
G.add_node("sensitive_file_payroll.csv", node_type="file")

# -----------------------------
# Add relationships
# -----------------------------
G.add_edge("alice", "laptop_01", relation="logs_into")
G.add_edge("bob", "laptop_02", relation="logs_into")

G.add_edge("laptop_01", "chrome.exe", relation="runs")
G.add_edge("laptop_01", "powershell.exe", relation="runs")
G.add_edge("laptop_02", "unknown.exe", relation="runs")

G.add_edge("powershell.exe", "server_01", relation="connects_to")
G.add_edge("chrome.exe", "external_ip_8.8.8.8", relation="connects_to")
G.add_edge("unknown.exe", "external_ip_185.10.10.10", relation="connects_to")

G.add_edge("bob", "server_01", relation="accesses")
G.add_edge("unknown.exe", "server_01", relation="touches")

G.add_edge("uknown.exe", "sensitive_file_payroll.csv", relation="writes_file")
G.add_edge("powershell.exe", "external_ip_185.10.10.10", relation="connects_to")

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

print("\nNodes:")
for node, attrs in G.nodes(data=True):
    print(f"{node:30s} {attrs}")

print("\nEdges:")
for src, dst, attrs in G.edges(data=True):
    print(f"{src:30} -> {dst:30s} {attrs}")

'''
# -----------------------------
# Visualize graph
# -----------------------------

# Layout decides where nodes appear on the plot.
pos = nx.spring_layout(G, seed=38)

# Assign simple colors based on node type.
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
    else:
        node_colors.append("gray")

plt.figure(figsize=(12, 8))

NODE_SIZE = 1800 

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=NODE_SIZE,
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
    node_size=NODE_SIZE,
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
)

# Draw edge labels such as logs_into, runs, connects_to.
edge_labels = nx.get_edge_attributes(G, "relation")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
)

plt.title("Aegis-HGX Mini Cybersecurity Graph")
plt.axis("off")
plt.tight_layout()
plt.show()
'''

print("\nPredecessors & Successors:")

#print(list(G.successors("laptop_01")))
#print(list(G.predecessors("server_01")))

noi=[n for n, attr in G.nodes(data=True) if attr.get("node_type") in ("Host", "ExternalIP") ]
for n in noi:
    ps=list(G.predecessors(n))
    for p in ps:
        print(f"{p} points to {n}")

noi=[n for n, attr in G.nodes(data=True) if attr.get("node_type")=="User" ]
for n in noi:
    ss=list(G.successors(n))
    for s in ss:
        print(f"{n} points to {s}")

print("\nPath from bob to external suspicious IP:")

path = nx.shortest_path(
    G,
    source="bob",
    target="external_ip_185.10.10.10"
)
print(path)

print(nx.shortest_path(G, "alice", "external_ip_185.10.10.10"))

print("\nAdjacency Matrix:")
nodes=list(G.nodes())
adj_matrix=nx.to_numpy_array(G, nodelist=nodes)

print("Node order:")
for i, node in enumerate(nodes):
    print(i, node)

print(adj_matrix)

print("\nNeighbors of a node:")
print(list(G.neighbors("laptop_01")))