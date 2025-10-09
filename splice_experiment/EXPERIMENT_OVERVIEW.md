# Splice Experiment: Detailed Overview

## Research Question

**Do high-delta cue_p sentences act as critical "anchoring points" that lock in the hinted answer during chain-of-thought reasoning?**

## Background

From the delta analysis (`per_problem_timeseries_with_sentences_all34_delta_analysis.csv`), we've identified sentences that cause massive jumps in `cue_p` (probability of outputting the hinted answer). For example:

- Problem 3, Sentence 0: `delta_cue_p = 0.82` (jumps from 0 to 0.82)
- Problem 288, Sentence 11: `delta_cue_p = 0.82` (jumps from 0.14 to 0.96)
- Problem 59, Sentence 4: `delta_cue_p = 0.76` (jumps from 0.24 to 1.0)

These sentences appear to be critical moments where the model "commits" to the hinted answer.

## Hypothesis

**H1 (Anchor Hypothesis):** High-delta sentences are causal anchors. If we replace them and subsequent reasoning with unhinted continuations, the model will be "freed" from the anchor and shift toward the unhinted answer distribution.

**H0 (Null Hypothesis):** High-delta sentences are merely correlated with the final answer but not causal. Replacing them won't significantly change the answer distribution.

## Experimental Design

### Intervention: The "Splice"

For each high-delta sentence:

1. **Extract prefix**: Take the hinted CoT up to (but not including) the high-delta sentence
2. **Sample unhinted continuation**: Prompt the unhinted model to continue from that prefix
3. **Measure outcome**: Compare the answer distribution to:
   - Original hinted distribution (baseline high on hinted answer)
   - Original unhinted distribution (baseline low on hinted answer)

### Visual Representation

```
Original Hinted CoT:
[Sentence 0] [Sentence 1] ... [HIGH DELTA SENTENCE] [Sentence N+1] ... [Answer: H]
                                       ^
                                       |
                              Splice point
                                       
Spliced CoT:
[Sentence 0] [Sentence 1] ... [UNHINTED CONTINUATION ..................] [Answer: ?]
      (from hinted model)              (from unhinted model)
```

### Predictions

If **H1** (anchors are causal):
- `P(hinted answer | splice)` << `P(hinted answer | original hinted)`
- `P(hinted answer | splice)` ≈ `P(hinted answer | original unhinted)`
- Strong correlation between `delta_cue_p` and `drop_from_hinted`

If **H0** (anchors not causal):
- `P(hinted answer | splice)` ≈ `P(hinted answer | original hinted)`
- No significant shift in distribution
- Random relationship between `delta_cue_p` and outcome

## Metrics

### Primary Metrics

1. **Drop from Hinted (`drop_from_hinted`)**
   - Definition: `P(hinted | original hinted) - P(hinted | splice)`
   - Interpretation: How much does the hinted answer probability decrease?
   - Strong evidence for H1: `drop_from_hinted > 0.2` (20% drop)

2. **Shift from Unhinted (`shift_from_unhinted`)**
   - Definition: `P(hinted | splice) - P(hinted | original unhinted)`
   - Interpretation: Does splice move toward unhinted baseline?
   - Strong evidence for H1: `shift_from_unhinted ≈ 0`

3. **Distribution Similarity (`closer_to`)**
   - Definition: L1 distance to hinted vs unhinted distributions
   - Interpretation: Which original condition is splice more similar to?
   - Strong evidence for H1: Most splices closer to "unhinted"

### Secondary Metrics

4. **Anchor Broken (binary)**
   - Definition: `drop_from_hinted > 0.1` (10% threshold)
   - Interpretation: Did the splice significantly reduce hinted probability?

5. **Correlation: Delta vs Drop**
   - Definition: Pearson correlation between `delta_cue_p` and `drop_from_hinted`
   - Interpretation: Do bigger deltas lead to bigger drops?
   - Strong evidence for H1: `r > 0.5`

## Implementation Details

### Key Technical Decisions

1. **Continuation Prompt**: We provide the prefix as "what you've thought so far" and ask the model to continue. This allows natural continuation without explicitly showing it was from a hinted context.

2. **Sentence Splitting**: We use a simple regex-based sentence splitter. This may not perfectly match the original analysis, so we verify by checking if the splice sentence appears in the CoT.

3. **Sample Size**: Default 5 samples per splice point for statistical power while managing API costs.

4. **Temperature**: 1.0 to match the original sampling distribution.

### Limitations

1. **Prefix Contamination**: The prefix might contain subtle hints that bias the unhinted continuation. This would reduce our measured effect size but still test the anchor hypothesis.

2. **Model Consistency**: The model's internal state at the splice point may differ from a fresh start. This is inherent to the intervention design.

3. **Sentence Boundaries**: Our sentence splitting may not perfectly match the original analysis. We mitigate this by checking for exact sentence matches.

## Expected Timeline

For a typical run with 20 splice points and 5 samples each:
- API calls: 20 × 5 = 100 calls
- Time per call: ~5-10 seconds
- Total time: ~10-20 minutes
- Cost: ~$0.50-1.00 (depending on model and rates)

## Data Flow

```
Input:
├── per_problem_timeseries_with_sentences_all34_delta_analysis.csv
│   └── High delta sentences identified
├── cot_responses.json (from original experiment)
│   └── Original hinted CoTs to splice
└── professor_hinted_mmlu.json
    └── Questions and base prompts

↓ run_splice_experiment.py ↓

Output:
├── splice_results.json
│   └── All splice samples with answers
├── metadata.json
│   └── Experiment configuration
└── summary.json
    └── Aggregate statistics

↓ analyze_results.py ↓

Analysis:
├── statistics.csv
│   └── Computed metrics per splice point
└── visualizations/
    ├── delta_vs_drop_scatter.png
    ├── probability_comparison_bars.png
    ├── splice_similarity.png
    └── problem_*_distribution.png
```

## Interpretation Guide

### Strong Evidence for H1 (Anchors are Causal)

- Most splice points show `drop_from_hinted > 0.2`
- Average `drop_from_hinted` > 0.15
- >70% of splices closer to "unhinted"
- Strong positive correlation (`r > 0.5`) between `delta_cue_p` and `drop_from_hinted`
- Visualizations show clear shift toward unhinted distributions

### Weak/No Evidence for H1

- Average `drop_from_hinted` < 0.05
- <50% of splices closer to "unhinted"
- No correlation between `delta_cue_p` and `drop_from_hinted`
- Spliced distributions remain similar to hinted originals

### Partial Evidence / Nuanced Results

- Moderate drops (0.05-0.15) suggest partial anchor effects
- Variation across problems suggests context-dependent anchoring
- Some high-delta sentences break anchors while others don't
  - Could indicate different types of reasoning steps
  - Or differences in how early/late they occur

## Follow-Up Experiments

If H1 is supported:

1. **Dose-response**: Test different delta thresholds to find the minimum delta for anchoring
2. **Position effects**: Does early vs late position in the CoT matter?
3. **Multiple splices**: What if we splice at multiple points in the same CoT?
4. **Reverse splice**: Insert hinted reasoning into unhinted CoTs

If H0 is supported:

1. **Look elsewhere**: Maybe anchoring happens differently (e.g., at the final answer sentence)
2. **Stronger intervention**: Try more disruptive interventions
3. **Different models**: Test if the pattern is model-specific

## Related Work

This experiment is inspired by:
- **Thought anchors** literature on how early reasoning steps constrain later ones
- **Unfaithful CoT** research showing CoT doesn't always reflect true reasoning
- **Causal tracing** methods in mechanistic interpretability

## Contact

For questions or to discuss results, please contact the project maintainer.

