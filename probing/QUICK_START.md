# 🚀 Quick Start Guide

## Run Analysis

```bash
# From the probing directory
python scripts/linear_probe_transplant.py
```

## File Locations

### Scripts
- `scripts/linear_probe_analysis.py` - Original analysis (hint-influenced CoT)
- `scripts/linear_probe_transplant.py` - Transplant analysis (main experiment)

### Results (Auto-generated)
- `results/data/*.csv` - CSV data files
- `results/figures/*.png` - Visualizations

### Documentation
- `docs/LINEAR_PROBE_README.md` - Detailed usage guide
- `docs/PROBE_COMPARISON.md` - Comparison of analyses
- `README.md` - Overview

## Interactive Menu

```bash
bash run_analysis.sh
```

Choose:
1. Transplant analysis (recommended)
2. Original analysis
3. Both

## Expected Runtime

- ~2.5 minutes: Collecting activations (30 problems × 48 layers)
- ~20 minutes: Training probes
- Total: ~22-25 minutes

## Outputs

Results automatically save to:
- `results/data/linear_probe_transplant_results.csv`
- `results/figures/linear_probe_transplant_heatmap.png`
- `results/figures/linear_probe_transplant_layers.png`
- `results/figures/linear_probe_transplant_sentences.png`

## Configuration

Edit in the script:
```python
MAX_PROBLEMS = 30      # Number of problems (None = all 41)
START_SENTENCE = 3     # Transplant start position
NUM_SENTENCES = 5      # Number of sentences to transplant
```

---

💡 **Tip**: Check `results/figures/` for visualizations after running!

