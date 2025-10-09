# File Guide: Splice Experiment

## Quick Start Files

### 📋 README.md
**Purpose**: Main documentation for the experiment
**Read this**: To understand the experiment design, methodology, and expected outputs

### 🚀 QUICKSTART.md
**Purpose**: Step-by-step guide to run your first experiment
**Read this**: When you're ready to actually run the code

### 📊 EXPERIMENT_OVERVIEW.md
**Purpose**: Detailed scientific overview with hypotheses, metrics, and interpretation
**Read this**: For deeper understanding of the research question and expected results

## Executable Scripts

### 🐍 run_splice_experiment.py
**Purpose**: Main experiment script
**Usage**:
```bash
python run_splice_experiment.py \
  --cot-data ../data/YOUR_DATA/cot_responses.json \
  --delta-threshold 0.5 \
  --num-samples 5 \
  --concurrency 5
```
**What it does**:
1. Loads high-delta sentences from delta analysis CSV
2. For each high-delta sentence:
   - Extracts the hinted CoT prefix
   - Samples unhinted continuations from the model
   - Saves results
3. Outputs to `results/YYYYMMDD_HHMMSS/`

### 📈 analyze_results.py
**Purpose**: Analysis script for experiment results
**Usage**:
```bash
python analyze_results.py \
  results/YYYYMMDD_HHMMSS/splice_results.json \
  --cot-data ../data/YOUR_DATA/cot_responses.json
```
**What it does**:
1. Loads splice results and original distributions
2. Computes statistics (drop_from_hinted, closer_to, etc.)
3. Generates visualizations
4. Saves to `results/YYYYMMDD_HHMMSS/visualizations/`

### 🧪 test_setup.py
**Purpose**: Verify setup before running experiment
**Usage**:
```bash
python test_setup.py
```
**What it does**:
1. Checks that all imports work
2. Verifies data files exist
3. Checks environment variables
4. Tests loading the delta analysis data
5. Shows top high-delta sentences

### 🔧 run_example.sh
**Purpose**: Shell script wrapper for a quick test run
**Usage**:
```bash
./run_example.sh
```
**What it does**:
- Runs experiment on 3 example problems with high deltas
- Uses sensible defaults for quick testing
- Prints helpful output messages

## Configuration Files

### ⚙️ config.py
**Purpose**: Central configuration for all scripts
**Contains**:
- Project paths (data directories, etc.)
- API configuration (model name, API key)
- Experiment parameters (defaults for thresholds, sample sizes)
- Output paths

**To modify**: Edit this file to change defaults

### 📦 requirements.txt
**Purpose**: Python dependencies
**Usage**:
```bash
pip install -r requirements.txt
```
**Contains**: openai, pandas, matplotlib, seaborn, python-dotenv

### 🚫 .gitignore
**Purpose**: Git ignore rules
**Excludes**: results/, __pycache__/, IDE files, OS files

## Input Data (from parent project)

These files are read by the scripts but live in the parent directory:

### 📊 ../cue_p_analysis/per_problem_timeseries_with_sentences_all34_delta_analysis.csv
**Purpose**: Source of high-delta sentences
**Format**: CSV with columns: problem_number, sentence_number, sentence, true_cue_p, delta_cue_p
**Used by**: run_splice_experiment.py

### 📝 ../data/professor_hinted_mmlu.json
**Purpose**: Original MMLU questions with hints
**Format**: JSON array of questions with base_prompt and hinted_prompt
**Used by**: run_splice_experiment.py

### 💭 ../data/YYYYMMDD_HHMMSS_model/cot_responses.json
**Purpose**: Original CoT responses (hinted and unhinted)
**Format**: JSON array with hinted_samples, no_hint_samples, distributions
**Used by**: run_splice_experiment.py (for hinted CoTs), analyze_results.py (for comparison)

## Output Files

After running the experiment, you'll get:

### 📁 results/YYYYMMDD_HHMMSS/

#### splice_results.json
Full experimental data with all splice samples
```json
[
  {
    "problem_number": 3,
    "sentence_number": 0,
    "delta_cue_p": 0.82,
    "splice_samples": [...],
    "answer_distribution": {...},
    ...
  }
]
```

#### metadata.json
Experiment configuration and parameters
```json
{
  "timestamp": "...",
  "model": "deepseek/deepseek-r1",
  "delta_threshold": 0.5,
  "num_samples": 5,
  ...
}
```

#### summary.json
High-level summary statistics
```json
{
  "total_splice_points": 20,
  "average_hinted_answer_probability": 0.45,
  ...
}
```

#### visualizations/

- `statistics.csv` - Detailed per-splice-point statistics
- `delta_vs_drop_scatter.png` - Scatter plot: delta vs drop
- `probability_comparison_bars.png` - Bar chart comparing probabilities
- `splice_similarity.png` - Which distribution splice is closer to
- `problem_X_sent_Y_distribution.png` - Individual distribution plots

## Typical Workflow

```
1. Read README.md and EXPERIMENT_OVERVIEW.md
   └─> Understand what the experiment does

2. Run test_setup.py
   └─> Verify everything is configured correctly

3. Edit run_example.sh (optional)
   └─> Update COT_DATA path if needed

4. Run run_example.sh OR run_splice_experiment.py directly
   └─> Execute the experiment

5. Run analyze_results.py
   └─> Generate statistics and visualizations

6. Review results/YYYYMMDD_HHMMSS/visualizations/
   └─> Interpret the findings
```

## Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| Import errors | Run `pip install -r requirements.txt` |
| "OPENROUTER_API_KEY not set" | Add to parent `.env` file |
| "Delta analysis CSV not found" | Check that `cue_p_analysis/` directory exists |
| "Empty prefix for sentence 0" | Expected - can't splice before first sentence |
| "No hinted CoT found for problem X" | Ensure CoT data matches delta analysis |
| API rate limits | Reduce `--concurrency` parameter |

## Advanced Usage

### Custom Analysis

```python
import json
import pandas as pd

# Load results
with open('results/TIMESTAMP/splice_results.json') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.json_normalize(results)

# Your custom analysis here
# ...
```

### Batch Processing

```bash
# Process multiple CoT datasets
for COT_DATA in ../data/*/cot_responses.json; do
    python run_splice_experiment.py \
        --cot-data "$COT_DATA" \
        --delta-threshold 0.5 \
        --num-samples 5
done
```

### Filtering by Problem

```bash
# Only process specific problems
python run_splice_experiment.py \
    --cot-data ../data/.../cot_responses.json \
    --problems 3,59,68,288,371 \
    --num-samples 10
```

## Questions?

- Check QUICKSTART.md for setup issues
- Check EXPERIMENT_OVERVIEW.md for methodology questions
- Review the inline comments in the Python files
- Look at the example output in results/ directories

