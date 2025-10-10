# Sentence Classification Analysis

This directory contains classification results for high-influence sentences from chain-of-thought reasoning.

## Files

- `high_diff_sentence_classifications.json` - Classification results for sentences with high cue_p diff
- `faith_counterfactual_qwen-14b_demo.csv` - Per-sentence cue_p data
- `Professor_itc_failure_threshold0.*.json` - Original question data with different thresholds
- `classification_plots/` - Visualization outputs

## Quick Commands

### Generate Classifications

```bash
python src/test_high_diff_sentence_alone.py \
  --csv-path sentence_classifications/faith_counterfactual_qwen-14b_demo.csv \
  --json-path sentence_classifications/Professor_itc_failure_threshold0.2_correct_base_no_mention.json \
  --min-diff 0.3
```

### Visualize Results

```bash
python src/plot_classification_analysis.py \
  --json-path sentence_classifications/high_diff_sentence_classifications.json
```

### Inspect Individual Cases

List all mechanisms:
```bash
python src/inspect_classifications.py \
  --json-path sentence_classifications/high_diff_sentence_classifications.json \
  --list-mechanisms
```

View specific mechanism:
```bash
python src/inspect_classifications.py \
  --json-path sentence_classifications/high_diff_sentence_classifications.json \
  --mechanism "False Framing"
```

Export to markdown:
```bash
python src/inspect_classifications.py \
  --json-path sentence_classifications/high_diff_sentence_classifications.json \
  --mechanism "Definitional Stretch" \
  --output definitional_stretch_cases.md
```

Search for mechanism anywhere (not just primary):
```bash
python src/inspect_classifications.py \
  --json-path sentence_classifications/high_diff_sentence_classifications.json \
  --mechanism "Certainty After Uncertainty" \
  --position any
```

## Taxonomy Overview

### Tier 1: Context-Independent Errors
1. False Framing
2. False Categorization
3. Definitional Stretch
4. Separation Fallacy
5. Misreading
6. Vague Principle Invocation
7. False Memory/Confabulation
8. Recognition Signal

### Tier 2: Context-Dependent Amplifiers

**2A: Backtracking From Correct Answer**
9. Artificial Ambiguity
10. Correct-Then-Abandon
11. Progressive Elimination

**2B: Building Unjustified Certainty**
12. Certainty After Uncertainty
13. Decision-Point Hedge Resolution

**2C: Semantic/Lexical Priming**
14. Keyword Repetition
15. Concept Activation
16. Lexical Matching Activation

