# Quickstart Guide

## Setup

1. Ensure you're in the parent project's virtual environment:
```bash
cd /Users/reillyhaskins/Desktop/MATS/mini_projects/thought_anchors_and_unfaithful_CoT
source venv/bin/activate
```

2. Install any missing dependencies:
```bash
cd splice_experiment
pip install -r requirements.txt
```

3. Ensure your `.env` file in the parent directory has:
```
OPENROUTER_API_KEY=your_key_here
MODEL_NAME=deepseek/deepseek-r1
```

## Running Your First Experiment

### Step 1: Find your CoT data

You need the original CoT data that was used to generate the delta analysis. For example:
```bash
# List available CoT data
ls -la ../data/*/cot_responses.json
```

Let's say you want to use: `../data/20251006_142931_deepseek_r1/cot_responses.json`

### Step 2: Run the experiment (small test first)

Start with a high threshold and specific problems to test the setup:

```bash
python run_splice_experiment.py \
  --cot-data ../data/20251006_142931_deepseek_r1/cot_responses.json \
  --delta-threshold 0.7 \
  --num-samples 3 \
  --problems 3,288,59
```

This will:
- Find sentences with delta_cue_p > 0.7
- Sample 3 unhinted continuations per splice point
- Only process problems 3, 288, and 59 (which have very high delta sentences)

**Expected output:**
```
Loaded 591 sentence entries from delta analysis
Loaded 1802 questions
Loaded 50 hinted CoTs

Found 8 sentences with delta_cue_p >= 0.7

Top 10 high delta sentences:
  Problem 3, Sentence 0: delta=0.820
  Problem 288, Sentence 11: delta=0.820
  ...

=== Processing Problem 3, Sentence 0 ===
Delta cue_p: 0.820
Splice sentence: Okay, I'm trying to figure out what Ashford's article is addressing...
  Prefix length: 0 chars
  ⚠ Empty prefix for sentence 0

=== Processing Problem 288, Sentence 11 ===
...
```

### Step 3: Analyze results

Once the experiment finishes, you'll see:
```
✓ Experiment complete! Results saved to results/20251009_HHMMSS
```

Analyze the results:
```bash
python analyze_results.py \
  results/20251009_HHMMSS/splice_results.json \
  --cot-data ../data/20251006_142931_deepseek_r1/cot_responses.json
```

This will:
- Compute statistics comparing splice to original distributions
- Generate visualizations in `results/20251009_HHMMSS/visualizations/`
- Save a `statistics.csv` file

### Step 4: Check results

```bash
# View summary
cat results/20251009_HHMMSS/summary.json

# View statistics
cat results/20251009_HHMMSS/visualizations/statistics.csv

# View plots
open results/20251009_HHMMSS/visualizations/
```

## Running a Full Experiment

Once you've tested the setup, run on more data:

```bash
# All sentences with delta > 0.5, 5 samples each
python run_splice_experiment.py \
  --cot-data ../data/20251006_142931_deepseek_r1/cot_responses.json \
  --delta-threshold 0.5 \
  --num-samples 5 \
  --concurrency 5
```

This will take longer (API calls for each splice point × num_samples).

## Interpreting Results

### Key Metrics

1. **drop_from_hinted**: How much does P(hinted answer) decrease after splicing?
   - Large positive values = anchor was broken
   - Near zero or negative = anchor held strong

2. **closer_to**: Is the spliced distribution closer to unhinted or hinted?
   - "unhinted" = splice successfully shifted toward unhinted reasoning
   - "hinted" = anchor persisted despite splice

3. **anchor_broken**: Boolean flag if drop_from_hinted > 0.1 (10%)

### What to Look For

**If high delta sentences are true anchors:**
- High `drop_from_hinted` values
- Most splices should be `closer_to: "unhinted"`
- Strong correlation between `delta_cue_p` and `drop_from_hinted`

**If they're not anchors (null hypothesis):**
- Small or no drop
- Splices remain close to hinted distribution
- No clear pattern

## Troubleshooting

### "Empty prefix for sentence 0"
This happens when trying to splice at the first sentence. The script will skip these automatically.
**Solution**: This is expected behavior - you can't splice before the first sentence.

### "No hinted CoT found for problem X"
The problem number in the delta analysis doesn't match the CoT data.
**Solution**: Make sure you're using the CoT data that corresponds to the delta analysis.

### API rate limits
If you see many timeouts or errors:
**Solution**: Reduce `--concurrency` (try 3 or 2)

### Sentence splitting mismatch
The script uses a simple sentence splitter which might not match exactly what was used in the original analysis.
**Solution**: If you notice mismatches, you may need to adjust the `split_cot_into_sentences()` method.

## Example Output Structure

```
splice_experiment/
├── results/
│   └── 20251009_143022/
│       ├── splice_results.json          # Raw data
│       ├── metadata.json                # Experiment config
│       ├── summary.json                 # High-level summary
│       └── visualizations/
│           ├── statistics.csv           # Detailed statistics
│           ├── delta_vs_drop_scatter.png
│           ├── probability_comparison_bars.png
│           ├── splice_similarity.png
│           └── problem_*_distribution.png  (one per splice point)
```

## Next Steps

Once you have results:

1. Look at the `statistics.csv` to identify which splice points showed the strongest effects
2. Examine individual distribution plots for interesting cases
3. Check if problems with higher `delta_cue_p` show stronger anchor breaking
4. Compare to the original cue_p analysis to see if predictions hold

## Advanced: Custom Analysis

You can load the results JSON directly for custom analysis:

```python
import json
import pandas as pd

# Load results
with open('results/20251009_HHMMSS/splice_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.json_normalize(results)

# Custom analysis
# ...
```

