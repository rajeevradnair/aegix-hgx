# Aegis-HGX Release 1: The Observer

The Observer is a static heterogeneous graph anomaly detector for cybersecurity snapshots.

It trains on normal typed cybersecurity behavior and scores a separate test snapshot containing normal actions plus injected suspicious actions.

## What It Detects

The release ranks risky typed actions such as:

- unknown.exe encrypts_file payroll.csv
- unknown.exe resolves malicious-domain.biz
- bob failed_login_to server_01
- powershell.exe connects_to suspicious external IP

## Model Design

The model uses:

- separate node encoders by node type
- separate attribute decoders by node type
- relation-specific bilinear decoders for typed edges

A typed edge is represented as:

```text
source_type + relation + destination_type