# GCN Anomaly Detection Bridge

## Core Idea

A node can be suspicious because of:

1. Its own features.
2. Its direct neighbors.
3. Its multi-hop graph context.

## Why GCNs Matter

A tabular model sees each event or entity mostly in isolation.

A graph model sees relationships:

- User logs into Host
- Host runs Process
- Process connects to External IP
- Process writes File

This allows the model to reason about attack paths.

## Feature Anomaly

A node looks suspicious by itself.

Example:

- unknown.exe has high risk score.
- external_ip_185.10.10.10 has suspicious reputation.

## Structural Anomaly

A node has suspicious relationships.

Example:

- unknown.exe writes payroll.csv.
- unknown.exe touches server_01.
- unknown.exe connects to a suspicious external IP.

## Contextual Anomaly

A node becomes suspicious because of its neighbors.

Example:

- bob may look normal alone.
- But bob connects to laptop_02, which runs unknown.exe.
- unknown.exe touches sensitive assets.

## Important Lesson

GCNs are useful because they allow suspicion to flow through relationships.

In Aegis-HGX, this is the bridge from feature-based anomaly detection to graph-based behavioral intelligence.