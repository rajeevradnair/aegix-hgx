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

# 3. Training Loop
print("Starting Training on Normal Behavior...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(normal_data)
    loss = criterion(output, normal_data)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print(type(loss), loss.mean())

# 4. The Detection Test
print("\n--- Testing Anomaly Detection ---")
ground_truth_normal = torch.ones((1, input_dim))
ground_truth_anomaly = torch.rand((1, input_dim)) # Random "messy" data

with torch.no_grad():
    reconstructed_normal = model(test_normal)
    reconstructed_anomaly = model(test_anomaly)
    
    error_normal = criterion(reconstructed_normal, ground_truth_normal)
    error_anomaly = criterion(reconstructed_anomaly, ground_truth_anomaly)

print(f"Normal Data Reconstruction Error: {error_normal.item():.6f}")
print(f"Anomalous Data Reconstruction Error: {error_anomaly.item():.6f}")

if error_anomaly > error_normal * 10:
    print("STATUS: Anomaly Successfully Detected!")