#!/bin/bash
# Simple test script that uses matching delta analysis and hinted CoT data

echo "=================================================="
echo "SPLICE EXPERIMENT - WORKING TEST"
echo "=================================================="
echo ""

# Step 1: Create test delta analysis if needed
if [ ! -f "delta_analysis_test.csv" ]; then
    echo "Creating test delta analysis for problems with hinted CoTs..."
    python3 << 'EOF'
import pandas as pd
import json

cot_data = json.load(open('../data_problems/20251006_142931_deepseek_r1/cot_responses.json'))
rows = []

for idx, item in enumerate(cot_data):
    if item['hinted_samples']:
        cot_text = item['hinted_samples'][0]['cot']
        sentences = [s.strip() for s in cot_text.replace('\n\n', '. ').replace('\n', ' ').split('. ') if s.strip()]
        
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
print(f"✓ Created delta_analysis_test.csv with {len(df)} sentences for problems {sorted(df['problem_number'].unique())}")
EOF
    echo ""
fi

# Step 2: Temporarily swap config
echo "Using test configuration..."
mv config.py config_prod.py.bak 2>/dev/null || true
cp config_test.py config.py

# Step 3: Run experiment
echo ""
echo "Running experiment on problem 0 (has hinted CoT)..."
echo ""

python run_splice_experiment.py \
  --cot-data "../data_problems/20251006_142931_deepseek_r1/cot_responses.json" \
  --delta-threshold 0.7 \
  --num-samples 2 \
  --concurrency 1 \
  --problems "0"

EXIT_CODE=$?

# Step 4: Restore config
mv config_prod.py.bak config.py 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Test completed successfully!"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "✗ Test failed"
    echo "=================================================="
    exit 1
fi

