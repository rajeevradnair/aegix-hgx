# Edge-Level Structure Explanations

## Problem

Node-level anomaly scores tell us which entity is suspicious, but not which action caused the suspicion.

## Solution

Use the reconstructed adjacency matrix to compute edge-level reconstruction error.

For each actual edge:

edge_error = (A[i][j] - A_hat[i][j])²

If the edge exists but the model predicts a low probability, the edge is suspicious.

## Example

Actual edge:

unknown.exe -> payroll.csv

If A = 1 and A_hat = 0.10:

edge_error = 0.81

Interpretation:

The model did not expect unknown.exe to write to payroll.csv.

## Why This Matters

SOC analysts need actionable explanations.

Bad:

unknown.exe anomaly score = 0.92

Better:

unknown.exe was flagged because:
- unknown.exe writes_file payroll.csv
- unknown.exe connects_to suspicious external IP
- unknown.exe touches server_01

## Aegis-HGX Relevance

This moves the project from anomaly detection toward analyst-ready incident explanation.