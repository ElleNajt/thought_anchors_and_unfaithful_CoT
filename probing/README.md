# Linear Probe Analysis

This folder contains tools for analyzing thought anchors through linear probing of model activations.

## 📁 Folder Structure

```
probing/
├── scripts/                      # Analysis scripts
│   ├── linear_probe_analysis.py      # Probe hint-influenced CoT
│   └── linear_probe_transplant.py    # Probe transplanted CoT (main experiment)
├── results/                      # Output directory
│   ├── data/                         # CSV results
│   └── figures/                      # Visualizations (heatmaps, trends)
├── docs/                         # Documentation
│   ├── LINEAR_PROBE_README.md        # Detailed usage guide
│   └── PROBE_COMPARISON.md           # Comparison of the two analyses
└── README.md                     # This file
```

## 🎯 Quick Start

### Run Transplant Analysis (Recommended)
```bash
cd probing
python scripts/linear_probe_transplant.py
```

This tests the **thought anchor hypothesis**: Can we decode the hint from transplanted CoT activations even when the hint was never shown?

### Run Original Analysis
```bash
python scripts/linear_probe_analysis.py
```

This analyzes hint-influenced CoT where the model saw the hint directly.

## 📊 What Gets Generated

### Data Files (results/data/)
- `linear_probe_transplant_results.csv` - Accuracy per (layer, sentence)
- `linear_probe_results.csv` - Original analysis results

### Figures (results/figures/)
- `*_heatmap.png` - Layer × Sentence accuracy grid
- `*_layers.png` - Average accuracy by layer
- `*_sentences.png` - Average accuracy by sentence position

## 🔬 Understanding the Results

### Transplant Analysis (Key Experiment)

**High accuracy (>60%)** means:
✅ Hint is decodable from transplanted CoT
✅ "Thought anchors" exist - reasoning carries latent hint representations
✅ Explains why transplanted CoT biases model behavior

**Low accuracy (~25% chance)** means:
❌ Hint not strongly encoded in transplanted activations
❌ Behavioral effects may come from other mechanisms

### Interpreting the Heatmap

- **Rows (Layers)**: Early layers (0-10) vs deep layers (40-48)
- **Columns (Sentences)**: Position in transplanted CoT
- **Color**: 
  - 🟢 Green (>60%): Hint is decodable
  - 🟡 Yellow (~50%): Weak signal
  - 🔴 Red (<35%): Near chance

### Key Patterns to Look For

1. **Layer trends**: Does accuracy increase in deeper layers?
2. **Sentence trends**: Does hint signal accumulate through CoT?
3. **Hot spots**: Which (layer, sentence) combinations show strongest hint encoding?

## ⚙️ Configuration

Edit in `scripts/linear_probe_transplant.py`:

```python
MAX_PROBLEMS = 30      # Number of problems (None = all 41)
START_SENTENCE = 3     # Which sentence to start transplanting from
NUM_SENTENCES = 5      # How many sentences to transplant
```

## 📖 Further Reading

- **`docs/LINEAR_PROBE_README.md`** - Detailed usage and interpretation
- **`docs/PROBE_COMPARISON.md`** - Comparison of transplant vs original analysis
- **`../cot_transplant.py`** - Behavioral transplant experiment

## 🐛 Troubleshooting

**"No probes trained" error:**
- Increase MAX_PROBLEMS
- Check data folder has JSON files

**Out of memory:**
- Reduce MAX_PROBLEMS
- Use smaller model

**Slow training:**
- Expected: ~25s per layer
- Total time: ~20 minutes for 30 problems

## 📝 Citation

If you use this code, please cite the thought anchors research and mention the linear probing methodology.

