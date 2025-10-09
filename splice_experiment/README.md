# Splice Experiment: Testing High Delta Cue_P Anchors

## Overview

This experiment tests whether "high delta cue_p" sentences act as critical anchoring points in chain-of-thought reasoning. We hypothesize that certain sentences cause massive jumps in the probability of outputting the hinted answer (cue_p), and removing them might break the anchoring effect.

## Experimental Design

### Procedure

1. **Identify high delta sentences**: Find sentences where `delta_cue_p > threshold` (e.g., 0.5)
2. **Splice operation**: For each high delta sentence:
   - Take the **hinted (H)** chain of thought up to (but not including) the high delta sentence
   - **Sample from unhinted (U)** model to continue reasoning from that point
   - Compare the final answer distribution to:
     - Original hinted CoT (expected: high probability on hinted answer)
     - Original unhinted CoT (expected: low probability on hinted answer)
     
3. **Analysis**: Does removing the anchor sentence and replacing with unhinted reasoning reduce the probability of the hinted answer?

### Key Questions

- Do high delta sentences truly act as "anchors" that lock in the hinted answer?
- Can we "break" the anchoring by replacing them with unhinted reasoning?
- How does the answer distribution shift when we swap at different points in the reasoning chain?

## Files

- `run_splice_experiment.py` - Main script to run the experiment
- `config.py` - Configuration parameters (thresholds, API settings, etc.)
- `analyze_results.py` - Analysis script for experiment results
- `requirements.txt` - Python dependencies

## Usage

### Prerequisites

```bash
# Ensure you have OPENROUTER_API_KEY in your .env file
# The script reuses the parent project's .env
```

### Running the Experiment

```bash
cd splice_experiment

# Run with default settings (delta_threshold=0.5, 5 samples per splice)
python run_splice_experiment.py

# Custom settings
python run_splice_experiment.py --delta-threshold 0.6 --num-samples 10 --concurrency 5

# Run on specific problems only
python run_splice_experiment.py --problems 3,288,972 --num-samples 5
```

### Analyzing Results

```bash
# After running the experiment
python analyze_results.py results/TIMESTAMP/splice_results.json
```

## Output Structure

```
splice_experiment/
├── results/
│   └── YYYYMMDD_HHMMSS/
│       ├── splice_results.json       # All splice completions
│       ├── metadata.json              # Experiment config
│       ├── summary.json               # High-level statistics
│       └── visualizations/            # Plots comparing distributions
│           ├── answer_distribution_comparison.png
│           └── per_problem_heatmap.png
```

## Data Format

### Input
- `cue_p_analysis/per_problem_timeseries_with_sentences_all34_delta_analysis.csv` - Delta analysis
- Original CoT data from parent project

### Output (`splice_results.json`)

```json
[
  {
    "problem_number": 3,
    "sentence_number": 0,
    "delta_cue_p": 0.82,
    "splice_point_sentence": "Okay, I'm trying to figure out...",
    "hinted_prefix": "...",  // CoT up to splice point
    "splice_samples": [
      {
        "sample_num": 0,
        "continuation": "...",  // Unhinted continuation
        "full_spliced_cot": "...",  // prefix + continuation
        "answer": "B"
      }
    ],
    "answer_distribution": {
      "A": 0.2,
      "B": 0.6,
      "C": 0.1,
      "D": 0.1
    },
    "comparison": {
      "original_hinted_dist": {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05},
      "original_unhinted_dist": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
      "splice_shifts_toward": "unhinted",  // or "hinted" or "neither"
      "hinted_answer_probability_drop": 0.2  // 0.8 -> 0.6
    }
  }
]
```

## Expected Results

If high delta sentences are true anchors:
- **Hypothesis**: Splicing at high delta points should shift answer distributions away from hinted answer
- **Null hypothesis**: No significant change in answer distribution

We'll measure:
1. **Anchor strength**: How much does the hinted answer probability drop after splicing?
2. **Direction**: Does it shift toward unhinted distribution or become more random?
3. **Consistency**: Is the effect consistent across multiple high delta sentences?

## Notes

- The experiment reuses infrastructure from `src/generate_cots_async.py`
- We need the **original hinted CoTs** to know where to splice
- The model needs to continue naturally from the prefix (we provide context)
- Temperature=1.0 for sampling diversity

