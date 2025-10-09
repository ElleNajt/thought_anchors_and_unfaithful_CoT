#!/bin/bash
# Working example script that uses test data with matching hinted CoTs

echo "=================================================="
echo "SPLICE EXPERIMENT - WORKING EXAMPLE"
echo "=================================================="
echo ""
echo "This will run a test experiment on problem 0 which"
echo "has both delta analysis AND hinted CoT data."
echo ""

# Configuration
COT_DATA="../data_problems/20251006_142931_deepseek_r1/cot_responses.json"
DELTA_THRESHOLD=0.7
NUM_SAMPLES=2  # Lower for local inference
PROBLEMS="0"  # Problem 0 has hinted CoT data
CONCURRENCY=1  # Sequential for local model

echo "Configuration:"
echo "  Model: DeepSeek-R1-Distill-Qwen-14B (local)"
echo "  CoT Data: $COT_DATA"
echo "  Delta Analysis: delta_analysis_test.csv (generated)"
echo "  Delta Threshold: $DELTA_THRESHOLD"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Problems: $PROBLEMS (has hinted CoTs)"
echo "  Concurrency: $CONCURRENCY"
echo ""

# Check if test delta analysis exists, create if needed
if [ ! -f "delta_analysis_test.csv" ]; then
    echo "Creating test delta analysis file..."
    python3 << 'EOF'
import pandas as pd
import json

# Load the hinted CoT data
cot_data = json.load(open('../data_problems/20251006_142931_deepseek_r1/cot_responses.json'))

rows = []
for idx, item in enumerate(cot_data):
    if item['hinted_samples']:
        cot_text = item['hinted_samples'][0]['cot']
        # Split into sentences
        sentences = [s.strip() for s in cot_text.replace('\n\n', '. ').replace('\n', ' ').split('. ') if s.strip()]
        
        # Create entries for first few sentences with high delta values
        for sent_num, sent in enumerate(sentences[:5]):
            rows.append({
                'problem_number': idx,
                'sentence_number': sent_num,
                'sentence': sent[:100],
                'true_cue_p': 0.7 + sent_num * 0.05,
                'delta_cue_p': 0.75 + sent_num * 0.03
            })

df = pd.DataFrame(rows)
df.to_csv('delta_analysis_test.csv', index=False)
print(f"✓ Created delta_analysis_test.csv with {len(df)} sentences")
EOF
    echo ""
fi

# Temporarily modify config.py to use test delta file
echo "Updating config to use test delta analysis..."
cp config.py config.py.bak
sed 's|"per_problem_timeseries_with_sentences_all34_delta_analysis.csv"|"../splice_experiment/delta_analysis_test.csv"|' config.py.bak > config.py

echo "Starting experiment..."
echo ""

# Run the experiment
python run_splice_experiment.py \
  --cot-data "$COT_DATA" \
  --delta-threshold $DELTA_THRESHOLD \
  --num-samples $NUM_SAMPLES \
  --concurrency $CONCURRENCY \
  --problems "$PROBLEMS"

# Restore original config
mv config.py.bak config.py

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

