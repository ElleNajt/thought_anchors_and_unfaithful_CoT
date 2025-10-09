# ✅ Updated: Splice Experiment Now Uses Local DeepSeek Model

## What Was Changed

I've updated the splice experiment to use **local DeepSeek-R1-Distill-Qwen-14B** inference instead of OpenRouter API, matching the rest of your codebase.

## Key Changes

### 1. ✅ Local Model Inference
- **Old**: OpenRouter API calls with `openai` library
- **New**: Local transformers inference with PyTorch
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (same as your other scripts)

### 2. ✅ No API Key Required
- Removed dependency on `OPENROUTER_API_KEY`
- No more API costs!

### 3. ✅ Fixed Data Path
- Updated to use your actual data folder: `data_problems/` (not `data/`)
- Points to: `../data_problems/20251006_142931_deepseek_r1/cot_responses.json`

### 4. ✅ Updated Dependencies
```
# Old:
openai>=1.0.0

# New:
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
```

### 5. ✅ GPU-Aware Configuration
- Reduced default concurrency to 2 (from 5) for GPU memory management
- Example script uses concurrency=1 for safety
- Detects CUDA availability automatically

## Your Data Structure

I noticed your data is in:
```
data_problems/
├── 20251006_142931_deepseek_r1/
│   ├── cot_responses.json    ← This is what you need!
│   └── metadata.json
└── professor_hinted_mmlu.json
```

The config now correctly points to `data_problems/` instead of `data/`.

## How to Run

### Step 1: Test Setup
```bash
cd splice_experiment
python test_setup.py
```

This will check:
- ✅ PyTorch and transformers installed
- ✅ CUDA available (or warn if CPU only)
- ✅ Data files exist
- ✅ Can load delta analysis

### Step 2: Run Small Test
```bash
./run_example.sh
```

This runs on 3 problems with high delta sentences (problems 3, 288, 59).

**First run will download the model** (~28GB) - this is a one-time thing!

### Step 3: Analyze Results
```bash
python analyze_results.py \
  results/TIMESTAMP/splice_results.json \
  --cot-data ../data_problems/20251006_142931_deepseek_r1/cot_responses.json
```

## Performance Expectations

| Hardware | Time per Sample | Recommended Concurrency |
|----------|----------------|------------------------|
| **CPU only** | 5-10 minutes | 1 |
| **RTX 3080** | 30-60 seconds | 1 |
| **RTX 4090** | 15-30 seconds | 1-2 |
| **A100** | 10-20 seconds | 2-4 |

## What About Your Existing Data?

Your existing CoT data (`cot_responses.json`) was generated via OpenRouter API with the large DeepSeek-R1 model. That's fine! The experiment will:

1. **Use that data** for the original hinted CoTs (to identify splice points)
2. **Use local model** to generate the unhinted continuations
3. **Compare distributions** between original (API) and spliced (local)

This is actually perfect for testing the anchoring hypothesis - we're testing if the high-delta sentence structure (not the specific model) causes anchoring.

## Benefits

✅ **No API costs** - Run as many experiments as you want
✅ **Matches your codebase** - Same model as `cot_transplant.py` and probing scripts  
✅ **Reproducible** - No API rate limits or network issues
✅ **Privacy** - All inference stays local

## Important Notes

1. **Model Size**: First run will download ~28GB model weights
2. **GPU Memory**: Use concurrency=1 if you have <16GB GPU memory
3. **Speed**: Local inference is slower than API, but you have full control
4. **Model Difference**: The 14B distilled model may behave slightly differently than the 671B API model - this is fine for testing the anchoring hypothesis

## Files Changed

- ✅ `run_splice_experiment.py` - Uses transformers instead of OpenAI client
- ✅ `config.py` - Updated paths and defaults
- ✅ `test_setup.py` - Checks PyTorch/CUDA instead of API key
- ✅ `run_example.sh` - Updated paths and parameters
- ✅ `requirements.txt` - Added torch/transformers, removed openai

## Documentation

All existing documentation still applies! The only differences are:
- No API key setup needed
- Lower recommended concurrency for GPU memory
- First run downloads model

See `CHANGES_LOCAL_MODEL.md` for detailed technical changes.

## Next Steps

1. **Run test**: `python test_setup.py`
2. **Try example**: `./run_example.sh`
3. **Check results**: Results will be in `results/YYYYMMDD_HHMMSS/`

The experiment design, metrics, and analysis are exactly the same - only the inference backend changed!

