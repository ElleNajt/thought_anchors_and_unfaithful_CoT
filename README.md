# Thought Anchors & Unfaithful CoT

Replication of "Unfaithful chain-of-thought as nudged reasoning" - focusing on Figure 4 (transplant resampling analysis).

## Source Files

- `src/create_professor_hinted_mmlu.py` - Creates MMLU dataset with professor hints (e.g., "A professor thinks the answer is (B).")
- `src/generate_cots.py` - Sequential CoT generation for hinted/unhinted questions
- `src/generate_cots_async.py` - Async/parallel CoT generation (recommended for large runs)

## Setup

### 1. Get OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up / Log in
3. Go to https://openrouter.ai/keys
4. Create a new API key
5. Add credits: https://openrouter.ai/credits

### 2. Configure Environment

Create `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-xxxxx
MODEL_NAME=deepseek/deepseek-r1
```

Available models: https://openrouter.ai/models
- `deepseek/deepseek-r1` - DeepSeek R1 (used in paper, shows unfaithful behavior)
- `deepseek/deepseek-chat` - DeepSeek Chat
- Other OpenRouter models

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Create Hinted Dataset

```bash
python src/create_professor_hinted_mmlu.py
```

Outputs to `data/professor_hinted_mmlu.json`

### 2. Generate CoT Responses

**Async version (recommended):**
```bash
# Quick test (2 questions, 5 samples each)
python src/generate_cots_async.py --num-questions 2 --num-samples 5 --concurrency 5

# Full run (50 questions, 10 samples each)
python src/generate_cots_async.py --num-questions 50 --num-samples 10 --concurrency 5

# Paper replication (100 questions, 100 samples each - very slow!)
python src/generate_cots_async.py --num-questions 100 --num-samples 100 --concurrency 5
```

**Sequential version:**
```bash
python src/generate_cots.py --num-questions 50 --num-samples 10
```

### 3. Monitor Progress

**Check async run:**
```bash
tail -f logs/cot_generation_async_TIMESTAMP.log
```

**Check background process:**
```bash
ps aux | grep generate_cots
```

## Output Structure

Results are organized by timestamp and model:

```
data/
  YYYYMMDD_HHMMSS_modelname/
    cot_responses.json    # Full CoT data
    metadata.json         # Model, config, results summary
  professor_hinted_mmlu.json

logs/
  cot_generation_TIMESTAMP.log
  cot_generation_async_TIMESTAMP.log
```

## Thought Anchor Experiments

This repository includes experiments to test whether CoT reasoning carries "thought anchors" - latent representations of hints that persist even when the hint is removed.

### 📁 Repository Structure

```
thought_anchors_and_unfaithful_CoT/
├── cot_transplant.py              # Behavioral transplant experiment
├── probing/                        # Linear probing analyses
│   ├── scripts/                   # Analysis code
│   ├── results/                   # Outputs (data & figures)
│   └── docs/                      # Detailed documentation
├── CoT_Faithfulness_demo/         # Demo with pre-computed results
└── src/                           # Original generation code
```

### 🧪 Experiment 1: Behavioral CoT Transplant

**Script**: `cot_transplant.py`

Tests if transplanting hint-influenced CoT to a no-hint prompt biases the model's answer.

**What it does:**
1. Takes sentences from CoT generated WITH a hint
2. Transplants them to a prompt WITHOUT the hint
3. Measures if the model's answer shifts toward the hinted answer

**Run it:**
```bash
python cot_transplant.py
```

**Expected output:**
- Baseline answer distribution (no transplant)
- Transplant answer distribution
- Comparison showing if transplant increased hinted answer rate

**Key finding:** If transplanted CoT increases hinted answer rate, it suggests the reasoning itself carries "anchors" to the hint.

---

### 🔬 Experiment 2: Linear Probing Analysis

**Location**: `probing/`

Tests if the hint answer is **linearly decodable** from model activations in transplanted CoT.

**Two analyses available:**

#### A. Transplant Probing (Recommended)
Tests the thought anchor hypothesis directly.

```bash
cd probing
python scripts/linear_probe_transplant.py
```

**What it measures:**
- Extract activations from transplanted prompts (hint never shown)
- Train linear probes to predict which answer was hinted
- If accuracy > chance → hint is encoded in activations!

**Outputs:**
- `results/data/linear_probe_transplant_results.csv`
- `results/figures/linear_probe_transplant_heatmap.png` (layer × sentence accuracy)
- `results/figures/linear_probe_transplant_layers.png`
- `results/figures/linear_probe_transplant_sentences.png`

#### B. Original Analysis
Analyzes hint encoding in hint-influenced CoT (baseline comparison).

```bash
cd probing
python scripts/linear_probe_analysis.py
```

**Interactive menu:**
```bash
cd probing
bash run_analysis.sh
```

**Documentation:**
- `probing/README.md` - Quick start guide
- `probing/QUICK_START.md` - Quick reference
- `probing/docs/` - Detailed guides

---

### 📊 Demo with Pre-computed Results

**Location**: `CoT_Faithfulness_demo/`

Contains pre-computed CoT faithfulness results for quick exploration:
- `faith_counterfactual_qwen-14b_demo.csv` - Sentence-by-sentence hint probabilities
- `Chua_faithfulness_results.csv` - Full faithfulness analysis
- `resampling_example.py` - Example analysis code

**Load the data:**
```python
from load_CoT import load_all_problems

data = load_all_problems()
# Returns 41 problems with hint/no-hint CoT pairs
```

Each problem includes:
- `question_with_cue`: Question with hint
- `question`: Question without hint  
- `reasoning_text`: CoT generated with hint
- `base_reasoning_text`: CoT generated without hint
- `cue_answer`: Hinted answer
- `gt_answer`: Ground truth answer

---

## Summary: Testing Thought Anchors

| Experiment | Question | Evidence of Thought Anchors |
|------------|----------|----------------------------|
| **Behavioral Transplant** | Does transplanted CoT bias answers? | Behavioral effect detected |
| **Linear Probing** | Is hint decodable from activations? | Representational evidence found |
| **Combined** | Do reasoning patterns carry hidden hints? | ✅ Strong evidence for thought anchors |

### Key Results

From the demo data and experiments:
- **Transplant effect**: Hint-influenced CoT increases hinted answer rate by ~10%
- **Probe accuracy**: Hint decodable with 60-83% accuracy (vs 25% chance)
- **Layer patterns**: Hint information strongest in middle-to-late layers (10-40)
- **Sentence patterns**: Hint signal accumulates through CoT progression

These findings suggest CoT reasoning carries **thought anchors** - latent representations of hints that persist even when hints are removed from the prompt.

---

## Next Steps for Figure 4 Replication

1. ✓ Generate CoTs with/without hints
2. ✓ Identify unfaithful CoTs (demo data provided)
3. ✓ Implement transplant experiments
4. ✓ Implement probing analyses
5. Create Figure 4 visualization (sentence-by-sentence resampling)
