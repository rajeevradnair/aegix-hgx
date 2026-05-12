import torch
from torch_geometric.data import HeteroData

users = {
    "alice": {"risk_score": 0.10, "privilege_level": 0.30},
    "bob": {"risk_score": 0.35, "privilege_level": 0.60},
}

hosts = {
    "laptop_01": {"risk_score": 0.20, "criticality": 0.40},
    "laptop_02": {"risk_score": 0.45, "criticality": 0.40},
    "server_01": {"risk_score": 0.70, "criticality": 0.90},
}

processes = {
    "chrome.exe": {"risk_score": 0.10, "signed_flag": 1.0},
    "powershell.exe": {"risk_score": 0.55, "signed_flag": 1.0},
    "unknown.exe": {"risk_score": 0.95, "signed_flag": 0.0},
}

external_ips = {
    "external_ip_8.8.8.8": {"risk_score": 0.15, "known_country_flag": 1.0},
    "external_ip_185.10.10.10": {"risk_score": 0.98, "known_country_flag": 0.0},
}

files = {
    "payroll.csv": {"risk_score": 0.90, "sensitivity": 1.0},
}

user_to_id = {name: idx for idx, name in enumerate(users.keys())}
host_to_id = {name: idx for idx, name in enumerate(hosts.keys())}
process_to_id = {name: idx for idx, name in enumerate(processes.keys())}
external_ip_to_id = {name: idx for idx, name in enumerate(external_ips.keys())}
file_to_id = {name: idx for idx, name in enumerate(files.keys())}

id_maps = {
    "User": user_to_id,
    "Host": host_to_id,
    "Process": process_to_id,
    "ExternalIP": external_ip_to_id,
    "File": file_to_id,
}

print(id_maps)
print("\n================ MAP EACH NODE TO IDS IN THEIR OWN LOCAL SPACE ================")
for node_type, mapping in id_maps.items():
    print(f"\n{node_type}:")
    for name, idx in mapping.items():
        print(f"  {idx}: {name}")