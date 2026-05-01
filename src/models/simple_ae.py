import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the 3-Layer Autoencoder
class SimpleAE(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAE, self).__init__()
        # Encoder: input_dim -> 5 -> 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 2) # Bottleneck
        )
        # Decoder: 2 -> 5 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, input_dim),
            nn.Sigmoid() # Keeps output between 0 and 1
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# 2. Setup model
input_dim = 10
model = SimpleAE(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# "Normal" behavior: mostly 1s
normal_data = torch.ones((100, input_dim)) 
#normal_data = torch.rand((100, input_dim)) 
print(f'Normal behavor is mostly : {normal_data}')

# 3. Training Loop
print("\nStarting model training on normal behavior...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(normal_data)
    loss = criterion(output, normal_data)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. Establish the baseline for reconstruction error
print("\nStarting model eval on normal behavior to calculate reconstruction error thresholds...")
model.eval() # Turn off gradients/dropout
baseline_losses = []

with torch.no_grad():
    for x in normal_data:
        reconstruction = model(x)
        loss = criterion(reconstruction, x)
        baseline_losses.append(loss.item())

baseline_tensor = torch.tensor(baseline_losses)
mu = torch.mean(baseline_tensor)
std = torch.std(baseline_tensor)

# Set the 0-trust threshold
threshold = mu + (3 * std) 
print(f"Baseline mean of Reconstruction Error: {mu:.6f}")
print(f"Baseline std dev of Reconstruction Error: {std:.6f}")
print(f"Threshold for anomalous detection set to: {threshold:.6f}")

# 5. The Detection Test
print("\nStarting Inferences ... Testing for anomaly detection")
ground_truth_anomaly = torch.rand((1, input_dim)) # Random "messy" data

with torch.no_grad():
    reconstructed_anomaly = model(ground_truth_anomaly)    
    error_anomaly = criterion(reconstructed_anomaly, ground_truth_anomaly)

print(f"Anomalous Data Reconstruction Error: {error_anomaly.item():.6f}")

if error_anomaly > threshold:
    print(f"STATUS: Anomaly Successfully Detected: {ground_truth_anomaly}")