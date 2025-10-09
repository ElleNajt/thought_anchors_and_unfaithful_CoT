# Cue_p Dynamics Analysis

This directory contains analysis of how `cue_p` (probability of outputting the hinted answer) evolves as sentences from hint-influenced Chain-of-Thought (CoT) reasoning are transplanted into non-hint runs.

## Data

- **34 problems** from Professor-hinted MMLU dataset (threshold 0.2)
- **589 total sentences** across all problems
- Each sentence has an associated `true_cue_p` value measured by transplanting it into non-hint runs

## Key Findings

### Delta cue_p Statistics

- **Mean delta**: 0.0538 (slight positive trend overall)
- **Median delta**: 0.0000 (most sentences don't change cue_p)
- **Range**: -0.50 to +0.82
- **Positive deltas**: 47.7% of sentences (281/589)
- **Negative deltas**: 24.3% of sentences (143/589)
- **Zero change**: 28.0% of sentences

### High-Impact "Anchor Points"

- **43 sentences** show large increases (Δ ≥ 0.3)
- **14 sentences** show large decreases (Δ ≤ -0.2)
- Large jumps occur at various positions, not just early sentences

### Top Delta Increases

The highest single-sentence jumps in cue_p are:

1. **+0.82**: Problem 3, Sentence 0 - Opening reasoning about Ashford's article
2. **+0.82**: Problems 288 & 972, Sentence 11 - Discussion of presidential negotiation powers
3. **+0.76**: Problem 59, Sentence 4 - Assertion about accordion in Tiny Tim's song
4. **+0.72**: Problem 715, Sentence 21 - Discussion of land meaning in common language

### Largest Decreases

1. **-0.50**: Problem 804, Sentence 4 - Factual statement about Voting Rights Act
2. **-0.30**: Problem 768, Sentence 5 - Question about accounting procedures
3. **-0.28**: Problems 26, 877, 295 - Various factual clarifications

## Plots Generated

### 1. `01_delta_distribution.png`
- Histogram and boxplot of delta_cue_p values
- Shows distribution is centered near zero but with significant positive tail
- Many sentences have small or zero effect, but some have large impacts

### 2. `02_delta_by_sentence_position.png`
- Scatter plot: individual sentence deltas vs position
- Line plot: average delta by position
- **Finding**: Early sentences (0-5) have higher variance, but high deltas occur throughout

### 3. `03_cue_p_trajectories.png`
- Spaghetti plot of all 34 problem trajectories
- Average trajectory with confidence bands
- Heatmap of cue_p evolution per problem
- Distribution of final cue_p values
- **Finding**: Problems show diverse trajectories; some converge quickly, others fluctuate

### 4. `04_anchor_points_analysis.png`
- Position distribution of large increases/decreases
- Average absolute delta by position
- Cumulative delta trajectories
- **Finding**: High-impact sentences occur throughout, not just at beginning

### 5. `05_convergence_analysis.png`
- When problems "lock in" to high cue_p (>0.8)
- Initial vs final cue_p scatter
- **Finding**: Strong initial cue_p doesn't guarantee maintaining it; some problems start low and converge high

## Files

- `per_problem_timeseries_with_sentences_all34.csv` - Raw timeseries data
- `per_problem_timeseries_with_sentences_all34_delta_analysis.csv` - Delta analysis sorted by highest delta
- `analyze_cue_p_dynamics.py` - Analysis script that generates all plots
- `figures/` - Directory containing all generated plots

## Usage

To regenerate the analysis:

```bash
cd cue_p_analysis
python3 analyze_cue_p_dynamics.py
```

## Interpretation

### What This Tells Us

1. **Hint influence is concentrated**: Only ~43 sentences (7.3%) show large jumps in cue_p
2. **Context matters**: The same hint-influenced reasoning can have vastly different effects depending on position
3. **Non-monotonic**: cue_p doesn't always increase - some sentences actually decrease it
4. **Initial sentences are powerful**: Many (but not all) high deltas occur in first 5 sentences
5. **Diverse trajectories**: No single pattern describes all problems

### Implications for Thought Anchors

These results suggest:
- **Thought anchors** (sentences that strongly shift probability toward hinted answer) exist and are identifiable
- They occur at various points in reasoning, not just early
- Some sentences "lock in" a direction, while others allow flexibility
- The hint's influence propagates non-uniformly through the CoT

## Questions for Further Investigation

1. What makes certain sentences "anchor points" with high delta?
2. Are specific types of reasoning (factual claims, logical steps, questions) more likely to be anchors?
3. Can we predict which sentences will have high delta based on linguistic features?
4. Do anchors correspond to specific token positions or attention patterns in the model?
5. Can we identify "unlock" sentences that reduce hint influence?

