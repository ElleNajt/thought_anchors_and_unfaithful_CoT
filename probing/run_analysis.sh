#!/bin/bash
# Convenience script to run linear probe analyses

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "Linear Probe Analysis Runner"
echo "=============================================================================="
echo ""
echo "Choose an analysis to run:"
echo "  1) Transplant Analysis (RECOMMENDED - tests thought anchors)"
echo "  2) Original Analysis (hint-influenced CoT)"
echo "  3) Both analyses"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Running TRANSPLANT analysis..."
        echo "This tests if hint is decodable from transplanted CoT"
        echo ""
        python "$SCRIPT_DIR/scripts/linear_probe_transplant.py"
        ;;
    2)
        echo ""
        echo "Running ORIGINAL analysis..."
        echo "This tests hint encoding in hint-influenced CoT"
        echo ""
        python "$SCRIPT_DIR/scripts/linear_probe_analysis.py"
        ;;
    3)
        echo ""
        echo "Running BOTH analyses..."
        echo ""
        echo "================================"
        echo "1/2: Original Analysis"
        echo "================================"
        python "$SCRIPT_DIR/scripts/linear_probe_analysis.py"
        
        echo ""
        echo "================================"
        echo "2/2: Transplant Analysis"
        echo "================================"
        python "$SCRIPT_DIR/scripts/linear_probe_transplant.py"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "Analysis complete! Check results in:"
echo "  - results/data/     (CSV files)"
echo "  - results/figures/  (PNG plots)"
echo "=============================================================================="

