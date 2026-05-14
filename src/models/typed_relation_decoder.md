# Typed Relation Decoder

## Problem

A simple source-destination decoder is not enough when two node types can have multiple relationship types.

Example:

Process reads_file File
Process writes_file File
Process deletes_file File
Process encrypts_file File

These relationships have different cybersecurity meanings.

## Solution

Use a relation-specific bilinear decoder.

Instead of:

score = sigmoid(z_src @ z_dst.T)

Use:

score = sigmoid(z_src @ W_relation @ z_dst.T)

Each canonical edge type gets its own W_relation.

## Why This Matters

unknown.exe reads_file payroll.csv

is different from:

unknown.exe encrypts_file payroll.csv

The model must score the full typed action:

source_type + relation + destination_type

## Aegis-HGX Relevance

This moves Aegis-HGX from generic graph anomaly detection toward typed cybersecurity behavior intelligence.