#!/bin/bash
# Example script to run a small splice experiment

echo "=================================================="
echo "SPLICE EXPERIMENT - EXAMPLE RUN"
echo "=================================================="
echo ""
echo "This will run a small test experiment on 3 problems"
echo "with high delta_cue_p sentences."
echo ""

# Configuration
COT_DATA="../data_problems/20251006_142931_deepseek_r1/cot_responses.json"
DELTA_THRESHOLD=0.7
NUM_SAMPLES=2  # Lower for local inference
PROBLEMS="3,288,59"
CONCURRENCY=1  # Sequential for local model (manage GPU memory)

echo "Configuration:"
echo "  Model: DeepSeek-R1-Distill-Qwen-14B (local)"
echo "  CoT Data: $COT_DATA"
echo "  Delta Threshold: $DELTA_THRESHOLD"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Problems: $PROBLEMS"
echo "  Concurrency: $CONCURRENCY"
echo ""

# Check if CoT data exists
if [ ! -f "$COT_DATA" ]; then
    echo "ERROR: CoT data file not found: $COT_DATA"
    echo ""
    echo "Please update the COT_DATA variable in this script to point to your data."
    echo "Available data files:"
    ls -la ../data/*/cot_responses.json 2>/dev/null || echo "  No data files found in ../data/"
    exit 1
fi

echo "Starting experiment..."
echo ""

# Run the experiment
python run_splice_experiment.py \
  --cot-data "$COT_DATA" \
  --delta-threshold $DELTA_THRESHOLD \
  --num-samples $NUM_SAMPLES \
  --concurrency $CONCURRENCY \
  --problems "$PROBLEMS"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Experiment completed successfully!"
    echo "=================================================="
    echo ""
    echo "Results are in: results/YYYYMMDD_HHMMSS/"
    echo ""
    echo "To analyze the results, run:"
    echo "  python analyze_results.py results/YYYYMMDD_HHMMSS/splice_results.json --cot-data $COT_DATA"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "✗ Experiment failed"
    echo "=================================================="
    echo ""
    echo "Check the error messages above."
    exit 1
fi

