# Static Typed-Edge Anomaly Detector

## Core Correction

The detector should not train and score on the same graph.

Correct anomaly detection flow:

1. Train on normal graph.
2. Score separate test graph.
3. Rank suspicious injected actions.

## Train Graph

The training graph contains only normal behavior.

Examples:

- alice logs_into laptop_01
- bob logs_into laptop_02
- laptop_01 runs chrome.exe
- chrome.exe reads_file readme.txt
- chrome.exe resolves normal-domain.com

## Test Graph

The test graph contains normal behavior plus suspicious injected actions.

Examples:

- bob failed_login_to server_01
- laptop_02 runs unknown.exe
- unknown.exe connects_to suspicious IP
- unknown.exe resolves malicious-domain.biz
- unknown.exe writes_file payroll.csv
- unknown.exe deletes_file payroll.csv
- unknown.exe encrypts_file payroll.csv

## Model

The model uses:

- one encoder per node type
- one attribute decoder per node type
- one bilinear relation decoder per canonical edge type

Relation score:

score = sigmoid(z_src @ W_relation @ z_dst.T)

## Typed Action Risk

typed_action_risk combines:

- model surprise
- source risk
- destination risk
- relation severity
- relation novelty

## Why This Matters

This is much closer to a real static SOC observer.

Instead of saying:

unknown.exe is anomalous

we can say:

unknown.exe encrypts_file payroll.csv ranked high because it was surprising, the source process was risky, the destination file was sensitive, the relation was severe, and the relation pattern was novel in normal training.

## Release 1 Readiness

This script prepares Release 1 by producing:

- edge scores
- edge labels
- ROC-AUC / PR-AUC
- node risk aggregation
- analyst explanations