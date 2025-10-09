# Changes: Switched from API to Local Model

## Summary

The splice experiment has been updated to use **local DeepSeek-R1-Distill-Qwen-14B** inference instead of OpenRouter API calls. This matches the rest of your codebase (e.g., `cot_transplant.py`, probing scripts).

## What Changed

### 1. Model Inference (`run_splice_experiment.py`)

**Before**: Used OpenAI client to call OpenRouter API
```python
from openai import AsyncOpenAI
self.client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
)
```

**After**: Loads local DeepSeek model with transformers
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 2. Generation Method

**Before**: Async API call
```python
response = await self.client.chat.completions.create(
    model=config.MODEL_NAME,
    max_tokens=4096,
    ...
)
```

**After**: Local generation with async wrapper
```python
def generate_continuation_sync(self, prefix: str, base_prompt: str):
    inputs = self.tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=config.TEMPERATURE,
            ...
        )
    continuation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 3. Configuration (`config.py`)

**Before**: 
- Required `OPENROUTER_API_KEY`
- Model: `"deepseek/deepseek-r1"` (API path)
- Concurrency: 5 (API calls)

**After**:
- No API key needed
- Model: `"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"` (HuggingFace model)
- Concurrency: 2 (GPU memory management)

### 4. Dependencies (`requirements.txt`)

**Removed**:
- `openai>=1.0.0`

**Added**:
- `torch>=2.0.0`
- `transformers>=4.36.0`
- `accelerate>=0.25.0`

### 5. Setup Testing (`test_setup.py`)

**Before**: Checked for `OPENROUTER_API_KEY` environment variable

**After**: Checks for PyTorch, transformers, and CUDA availability

### 6. Example Script (`run_example.sh`)

**Before**: 3 samples per problem, concurrency 5

**After**: 2 samples per problem, concurrency 1 (for GPU memory)

## Benefits of Local Model

✅ **No API costs** - Run unlimited experiments
✅ **Consistency** - Matches your existing codebase setup
✅ **Privacy** - All data stays local
✅ **Reproducibility** - No API rate limits or version changes
✅ **Control** - Full control over model parameters

## Trade-offs

⚠️ **Speed**: Local generation is slower than API (but you can batch if you have GPU memory)
⚠️ **Setup**: Requires ~28GB disk space for model weights
⚠️ **Hardware**: Needs GPU for reasonable speed (CPU works but is very slow)

## Usage

Everything remains the same except:

1. **No need for API key** - Remove `OPENROUTER_API_KEY` from `.env`
2. **Model loads on first run** - Will download ~28GB model first time
3. **Lower concurrency** - Use 1-2 instead of 5-10 to avoid GPU OOM

### Example Commands

```bash
# Test setup (will check for PyTorch, CUDA)
python test_setup.py

# Run example (2 samples, 1 concurrent)
./run_example.sh

# Full experiment (adjust num-samples and concurrency based on your GPU)
python run_splice_experiment.py \
  --cot-data ../data/20251006_142931_deepseek_r1/cot_responses.json \
  --delta-threshold 0.5 \
  --num-samples 3 \
  --concurrency 1
```

## Hardware Recommendations

| GPU | Concurrency | Samples | Notes |
|-----|-------------|---------|-------|
| **No GPU (CPU)** | 1 | 1-2 | Very slow (~5-10 min/sample) |
| **RTX 3080 (10GB)** | 1 | 2-3 | ~30-60s/sample |
| **RTX 4090 (24GB)** | 2 | 5 | ~15-30s/sample |
| **A100 (40GB+)** | 2-4 | 10 | ~10-20s/sample |

## Metadata Changes

Results now include:
```json
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
  "inference_type": "local",
  "device": "cuda",  // or "cpu"
  ...
}
```

## Backward Compatibility

✅ Analysis scripts unchanged - they just read the JSON results
✅ Output format unchanged - same JSON structure
✅ Can still compare results from API runs vs local runs

## Notes

- The DeepSeek-R1-Distill-Qwen-14B (14B model) is different from the larger DeepSeek-R1 (671B model used via API)
- Results may differ slightly due to model differences
- Your existing CoT data was generated via API, but the splice experiment will use local model for new continuations
- This is fine - we're testing the anchoring effect, not exact model replication

