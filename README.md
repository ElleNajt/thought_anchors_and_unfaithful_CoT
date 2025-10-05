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

## Next Steps for Figure 4 Replication

1. ✓ Generate CoTs with/without hints
2. Filter questions with hint_effect > 10%
3. Manually identify unfaithful CoTs (answer changed, hint not mentioned)
4. Implement transplant resampling (truncate CoT sentence-by-sentence, resample)
5. Create Figure 4 visualization
