from dataclasses import dataclass
from collections import defaultdict

ALLOWED_NODE_TYPES = {
    "User",
    "Host",
    "Process",
    "ExternalIP",
    "File",
    "Domain",
}

ALLOWED_EDGE_TYPES = {
    ("User", "logs_into", "Host"),
    ("User", "accesses", "Host"),
    ("User", "launches", "Process"),
    ("Host", "runs", "Process"),
    ("Process", "connects_to", "ExternalIP"),
    ("Process", "writes_file", "File"),
    ("Process", "touches", "Host"),
    ("Process", "resolves", "Domain"),
}

nodes = {
    "alice": {"node_type": "User", "risk_score": 0.10},
    "bob": {"node_type": "User", "risk_score": 0.35},

    "laptop_01": {"node_type": "Host", "risk_score": 0.20},
    "laptop_02": {"node_type": "Host", "risk_score": 0.45},
    "server_01": {"node_type": "Host", "risk_score": 0.70},

    "chrome.exe": {"node_type": "Process", "risk_score": 0.10},
    "powershell.exe": {"node_type": "Process", "risk_score": 0.55},
    "unknown.exe": {"node_type": "Process", "risk_score": 0.95},

    "external_ip_8.8.8.8": {"node_type": "ExternalIP", "risk_score": 0.15},
    "external_ip_185.10.10.10": {"node_type": "ExternalIP", "risk_score": 0.98},

    "payroll.csv": {"node_type": "File", "risk_score": 0.90},

    "malicious-domain.biz": {"node_type": "Domain", "risk_score": 0.99},
}

@dataclass
class EdgeEvent:
    src: str
    relation: str
    dst: str
    timestamp: str


candidate_edges = [
    # Valid edges
    EdgeEvent("alice", "logs_into", "laptop_01", "09:00"),
    EdgeEvent("bob", "logs_into", "laptop_02", "11:00"),
    EdgeEvent("laptop_01", "runs", "chrome.exe", "09:05"),
    EdgeEvent("laptop_02", "runs", "unknown.exe", "11:03"),
    EdgeEvent("unknown.exe", "connects_to", "external_ip_185.10.10.10", "11:04"),
    EdgeEvent("unknown.exe", "writes_file", "payroll.csv", "11:06"),
    EdgeEvent("unknown.exe", "touches", "server_01", "11:12"),
    EdgeEvent("unknown.exe", "resolves", "malicious-domain.biz", "11:05"),

    # Invalid edges
    EdgeEvent("payroll.csv", "logs_into", "alice", "12:00"),
    EdgeEvent("external_ip_185.10.10.10", "runs", "unknown.exe", "12:01"),
    EdgeEvent("server_01", "writes_file", "bob", "12:02"),
]

def validate_nodes(node_dict):
    errors = []

    for node_name, attrs in node_dict.items():
        node_type = attrs.get("node_type")

        if node_type not in ALLOWED_NODE_TYPES:
            errors.append(
                f"Node {node_name} has invalid node_type={node_type}"
            )

    return errors


def validate_edge(edge, node_dict):
    if edge.src not in node_dict:
        return False, f"Source node does not exist: {edge.src}", None

    if edge.dst not in node_dict:
        return False, f"Destination node does not exist: {edge.dst}", None

    src_type = node_dict[edge.src]["node_type"]
    dst_type = node_dict[edge.dst]["node_type"]

    canonical_edge_type = (
        src_type,
        edge.relation,
        dst_type,
    )

    if canonical_edge_type not in ALLOWED_EDGE_TYPES:
        return (
            False,
            f"Invalid edge type {canonical_edge_type} for edge "
            f"{edge.src} -[{edge.relation}]-> {edge.dst}",
            canonical_edge_type,
        )

    return True, "Valid", canonical_edge_type

node_errors = validate_nodes(nodes)

if node_errors:
    print("\nNode validation errors:")
    for error in node_errors:
        print("  -", error)
else:
    print("\nAll node types are valid.")


accepted_edges_by_type = defaultdict(list)
rejected_edges = []

for edge in candidate_edges:
    is_valid, message, canonical_edge_type = validate_edge(edge, nodes)

    if is_valid:
        accepted_edges_by_type[canonical_edge_type].append(edge)
    else:
        rejected_edges.append((edge, message))

print("\n================ ACCEPTED EDGES ================")

for edge_type, edges in accepted_edges_by_type.items():
    print(f"\n{edge_type}:")
    for edge in edges:
        print(
            f"  {edge.src} -[{edge.relation}]-> {edge.dst} "
            f"at {edge.timestamp}"
        )

print("\n================ REJECTED EDGES ================")

for edge, reason in rejected_edges:
    print(
        f"{edge.src} -[{edge.relation}]-> {edge.dst} "
        f"at {edge.timestamp}"
    )
    print(f"  Reason: {reason}")


print("\n================ VALIDATION SUMMARY ================")
print(f"Allowed node types: {len(ALLOWED_NODE_TYPES)}")
print(f"Acceptable edge types: {len(accepted_edges_by_type)}")
print(f"Candidate edges: {len(candidate_edges)}")
print(
    "Accepted edges:",
    sum(len(edges) for edges in accepted_edges_by_type.values())
)
print(f"Rejected edges: {len(rejected_edges)}")
