# Linear Probe Comparison: Original vs Transplanted

This folder contains two different linear probe analyses that test different hypotheses about how hints influence model reasoning.

## Two Scripts, Two Questions

### 1. `linear_probe_analysis.py` - Original Hint-Influenced CoT

**What it does:**
- Takes examples where model saw the hint
- Analyzes activations during hint-influenced generation
- Trains probes to predict: "Which answer was hinted?"

**Question:** When the model processes hint-influenced text, where is the hint encoded in activations?

**Example:**
```
Input: "Stanford Professor answered D. [question]"
Model generates CoT with hint present
Probe: Can we decode "D" from activations?
```

**Measures:** Representation during hint-influenced processing

---

### 2. `linear_probe_transplant.py` - Transplanted CoT (NEW!)

**What it does:**
- Takes hint-influenced CoT but removes the hint
- Transplants CoT after no-hint question
- Analyzes activations during transplanted processing
- Trains probes to predict: "Which answer was used to generate this CoT?"

**Question:** When hint-influenced CoT is transplanted (hint removed), is the hint still decodable from activations?

**Example:**
```
Input: "[no-hint question] + [CoT generated after seeing hint D]"
Note: Hint "D" was never shown to the model!
Probe: Can we still decode "D" from activations?
```

**Measures:** "Thought anchors" - whether reasoning carries latent hint representations

---

## Why Both Matter

| Experiment | Tests | Interpretation if Hint is Decodable |
|------------|-------|-------------------------------------|
| **Original** | Hint processing | Expected - model knows about hint |
| **Transplant** | Thought anchors | Surprising - reasoning "remembers" hint without seeing it! |

## The Thought Anchor Hypothesis

If `linear_probe_transplant.py` shows **high accuracy**, it means:

1. ✅ CoT reasoning carries latent representations of the hint
2. ✅ These representations persist even when hint is removed
3. ✅ "Thought anchors" exist - reasoning is anchored to the hint used to generate it
4. ✅ This explains why transplanted CoT biases model behavior

If it shows **low accuracy** (near chance):
- ❌ Hint not strongly encoded in transplanted activations
- ❌ Behavioral effects may come from other mechanisms

## How to Run

```bash
# Original analysis (hint-influenced)
python linear_probe_analysis.py

# Transplant analysis (hint removed but CoT transplanted)
python linear_probe_transplant.py
```

## Output Files

### Original Analysis:
- `linear_probe_results.csv`
- `linear_probe_heatmap.png`
- `linear_probe_layers.png`
- `linear_probe_sentences.png`

### Transplant Analysis:
- `linear_probe_transplant_results.csv`
- `linear_probe_transplant_heatmap.png`
- `linear_probe_transplant_layers.png`
- `linear_probe_transplant_sentences.png`

## Expected Results

**Prediction:** Transplant probes should show:
- Lower accuracy than original (hint signal weaker without direct exposure)
- But still above chance (25% for 4 classes)
- Accuracy increases through layers (deeper layers integrate reasoning)
- Accuracy increases through sentences (hint signal accumulates)

If these patterns hold, it provides strong evidence for thought anchors!

