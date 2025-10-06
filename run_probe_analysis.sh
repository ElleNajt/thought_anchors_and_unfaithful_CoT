#!/bin/bash
# Convenience script to run linear probe analysis

echo "Installing dependencies..."
pip install -q matplotlib seaborn scikit-learn 2>/dev/null

echo ""
echo "========================================================================"
echo "Running Linear Probe Analysis"
echo "========================================================================"
echo ""

python linear_probe_analysis.py

echo ""
echo "Analysis complete! Check the generated plots:"
echo "  - linear_probe_heatmap.png"
echo "  - linear_probe_layers.png"
echo "  - linear_probe_sentences.png"

