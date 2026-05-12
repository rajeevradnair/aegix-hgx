# Heterogeneous Edge Validation

## Core Idea

A cybersecurity graph needs typed relationships.

A valid edge is not just:

source -> destination

It is:

source_type -> relation -> destination_type

## Example Valid Edge

User logs_into Host

alice logs_into laptop_01

## Example Invalid Edge

File logs_into User

payroll.csv logs_into alice

## Why Validation Matters

Graph ML models learn from the graph structure we provide. If invalid relationships enter the graph, the model can learn meaningless or harmful patterns.
Garbage In -> Garbage Out

## Valid But Suspicious vs Invalid

Invalid edge:

File logs_into User

This should be rejected.

Valid but suspicious edge:

Process writes_file File

This should be accepted, then scored later.

## Link to Aegis-HGX

This validation layer protects the graph ingestion pipeline before we build HeteroData and heterogeneous GNN models.