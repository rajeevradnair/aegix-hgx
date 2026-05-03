import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Aegis-HGX Day 2
# Autoencoder Reconstruction Error for Cybersecurity Anomalies
# ============================================================

# We make results more reproducible.
torch.manual_seed(42)


# ------------------------------------------------------------
# 1. Define cybersecurity feature names
# ------------------------------------------------------------
# Each row represents behavior for one user/host in a time window.
# Example: one machine's activity over 10 minutes.

feature_names = [
    "login_count",
    "failed_login_count",
    "bytes_sent_mb",
    "bytes_received_mb",
    "process_count",
    "unique_destinations",
    "dns_queries",
    "privileged_commands",
    "file_writes",
    "off_hours_activity",
]

input_dim = len(feature_names)


# ------------------------------------------------------------
# 2. Create "normal" cybersecurity behavior data
# ------------------------------------------------------------
# These rows are intentionally boring.
# They represent normal workstation/user behavior.

normal_data = torch.tensor([
    [5, 1, 120, 300, 80, 10, 25, 0, 40, 0],
    [6, 0, 100, 280, 75, 12, 22, 0, 35, 0],
    [4, 1, 90, 250, 85, 9, 20, 0, 38, 0],
    [7, 2, 140, 310, 78, 11, 30, 1, 42, 0],
    [5, 0, 110, 290, 82, 10, 26, 0, 39, 0],
    [6, 1, 130, 305, 79, 13, 28, 0, 41, 0],
    [4, 0, 95, 260, 83, 8, 21, 0, 36, 0],
    [5, 1, 115, 295, 81, 12, 27, 0, 40, 0],
    [6, 1, 125, 300, 77, 11, 24, 0, 37, 0],
    [5, 0, 105, 275, 84, 9, 23, 0, 39, 0],
], dtype=torch.float32)


# ------------------------------------------------------------
# 3. Create realistic anomaly examples
# ------------------------------------------------------------
# These are not random vectors.
# Each one represents a cyber-shaped suspicious pattern.

anomaly_data = torch.tensor([
    # Brute-force login pattern:
    # login count is low, but failed logins are extremely high.
    [2, 35, 150, 280, 90, 12, 26, 0, 39, 0],

    # Low-and-slow / exfiltration-like pattern:
    # huge outbound data and many unique destinations.
    [5, 1, 5000, 300, 85, 200, 300, 0, 45, 0],

    # Malware/process explosion pattern:
    # many processes, privileged commands, file writes, off-hours activity.
    [4, 0, 100, 260, 400, 80, 150, 20, 500, 1],
], dtype=torch.float32)


# ------------------------------------------------------------
# 4. Scale the data
# ------------------------------------------------------------
# Important:
# We calculate mean and std using ONLY normal data.
# This simulates the real-world setup:
# "Learn what normal looks like, then detect what deviates."

mean = normal_data.mean(dim=0)
std = normal_data.std(dim=0)

# Some features may have zero standard deviation.
# Example: off_hours_activity may be 0 for all normal rows.
# Dividing by zero would break the program.
# So if std is 0, replace it with 1.
std = torch.where(std == 0, torch.ones_like(std), std)

normal_scaled = (normal_data - mean) / std
anomaly_scaled = (anomaly_data - mean) / std


# ------------------------------------------------------------
# 5. Define the autoencoder
# ------------------------------------------------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAE, self).__init__()

        # Encoder:
        # 10 input features -> 5 hidden values -> 2 bottleneck values
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Decoder:
        # 2 bottleneck values -> 5 hidden values -> 10 reconstructed features
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


model = SimpleAE(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ------------------------------------------------------------
# 6. Train on normal behavior only
# ------------------------------------------------------------
print("\nStarting model training on normal cybersecurity behavior...")

epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()

    reconstructed = model(normal_scaled)

    loss = criterion(reconstructed, normal_scaled)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Training Reconstruction Loss: {loss.item():.6f}")


# ------------------------------------------------------------
# 7. Establish baseline reconstruction threshold
# ------------------------------------------------------------
print("\nCalculating normal reconstruction error baseline...")

model.eval()
baseline_losses = []

with torch.no_grad():
    for x in normal_scaled:
        reconstruction = model(x)

        # MSE for one row
        loss = criterion(reconstruction, x)

        baseline_losses.append(loss.item())

baseline_tensor = torch.tensor(baseline_losses)

mu = torch.mean(baseline_tensor)
sigma = torch.std(baseline_tensor)

threshold = mu + (3 * sigma)

print(f"Baseline mean reconstruction error: {mu:.6f}")
print(f"Baseline std reconstruction error:  {sigma:.6f}")
print(f"Anomaly threshold:                 {threshold:.6f}")


# ------------------------------------------------------------
# 8. Test normal + anomalous behavior
# ------------------------------------------------------------
print("\nRunning inference on normal and anomalous behavior...")

all_data_scaled = torch.cat([normal_scaled, anomaly_scaled], dim=0)
all_data_raw = torch.cat([normal_data, anomaly_data], dim=0)

labels = (
    ["NORMAL"] * len(normal_data)
    + ["BRUTE_FORCE"]
    + ["EXFILTRATION"]
    + ["MALWARE_LIKE"]
)

with torch.no_grad():
    reconstructed_all_data_scaled = model(all_data_scaled)

    # Per-row reconstruction error:
    # mean over the 10 features
    row_errors = torch.mean((all_data_scaled - reconstructed_all_data_scaled) ** 2, dim=1)


print("\nSummary Results:")
print("-" * 80)

for i, error in enumerate(row_errors):
    status = "ALERT" if error > threshold else "OK"

    print(
        f"Row {i:02d} | type={labels[i]:13s} | "
        f"status={status:5s} | reconstruction_error={error.item():.6f}"
    )


# ------------------------------------------------------------
# 9. Explain each anomaly with per-feature reconstruction error
# ------------------------------------------------------------
print("\nDetailed Feature-Level Explanation for Anomalies:")
print("=" * 80)

for idx in range(len(normal_data), len(all_data_scaled)):
    x = all_data_scaled[idx]
    x_hat = reconstructed_all_data_scaled[idx]

    # per-feature errors are being calculated on scaled data only 
    # for the purposes of sorting the indices to find the dominant features
    per_feature_error = (x - x_hat) ** 2
    #print("***", x)
    #print("***", x_hat)
    #print("***", per_feature_error)

    print(f"\nExample {idx}: {labels[idx]}")
    print(f"Overall reconstruction error: {row_errors[idx].item():.6f}")
    print("Top feature reconstruction errors:")

    # Sort feature errors from largest to smallest
    sorted_indices = torch.argsort(per_feature_error, descending=True)

    for feature_idx in sorted_indices:
        feature_idx = feature_idx.item()

        print(
            f"  {feature_names[feature_idx]:25s} "
            f"raw_value={all_data_raw[idx][feature_idx].item():8.2f} "
            f"feature_error={per_feature_error[feature_idx].item():.6f}"
        )


# ------------------------------------------------------------
# 10. Final interpretation
# ------------------------------------------------------------
print("\nInterpretation:")
print("-" * 80)
print("The autoencoder was trained only on normal behavior.")
print("Normal examples should reconstruct with low error.")
print("Cyber-shaped anomalous examples should reconstruct with high error.")
print("The feature-level errors tell us WHY the model raised an alert.")