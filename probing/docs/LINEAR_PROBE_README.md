# Linear Probe Analysis for CoT Hint Detection

This script trains linear probes to detect when the hint answer becomes linearly decodable from model activations.

## What It Does

The script:
1. **Extracts activations** from every layer of the model at each sentence position in the CoT
2. **Trains linear probes** (logistic regression) to predict the hint answer (A/B/C/D) from these activations
3. **Evaluates on test set** to measure generalization
4. **Creates visualizations** showing where and when the hint becomes decodable

## Key Question

**At which layer and sentence position does the hint answer become linearly decodable?**

This helps identify:
- When the model "commits" to the hinted answer
- Which layers encode hint information most strongly
- Whether early or late CoT sentences carry more hint signal

## Installation

```bash
# Install dependencies
bash install_probe_deps.sh

# Or manually:
pip install matplotlib seaborn scikit-learn pandas numpy
```

## Usage

```bash
cd /root/thought_anchors_and_unfaithful_CoT
python linear_probe_analysis.py
```

The script will:
- Load the DeepSeek model (same as transplant experiments)
- Process problems and extract activations
- Train probes for each (layer, sentence) combination
- Generate plots and save results

## Output Files

1. **`linear_probe_results.csv`** - Raw results with accuracy per (layer, sentence)
2. **`linear_probe_heatmap.png`** - Heatmap showing accuracy across layers × sentences
3. **`linear_probe_layers.png`** - Average accuracy per layer
4. **`linear_probe_sentences.png`** - Average accuracy per sentence position

## Configuration

Edit these variables in `main()`:

```python
MAX_PROBLEMS = 30  # Number of problems to analyze (None = all)
```

## Interpreting Results

- **Chance accuracy**: 25% (4 classes: A, B, C, D)
- **Good accuracy**: >60% suggests hint is decodable
- **High accuracy in early sentences**: Hint encoded early in reasoning
- **High accuracy in later layers**: Hint information processed in deeper layers

## Example Insights

The analysis will show:
- Which layer first encodes the hint (e.g., "Layer 15")
- Which sentence position first reveals the hint (e.g., "Sentence 3")
- Whether certain layers are "specialists" at encoding the hint
- How hint information evolves through the CoT

## Comparison to Transplant

- **Transplant**: Shows behavioral effect (does transplanted CoT bias the answer?)
- **Linear Probe**: Shows representational effect (when is hint encoded in activations?)

Together, these reveal both *what the model does* and *what it knows*.

