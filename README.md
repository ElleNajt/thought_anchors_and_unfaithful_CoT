# Thought Anchors & Unfaithful CoT

## Source Files

- `src/create_professor_hinted_mmlu.py` - Creates MMLU dataset with professor hints (e.g., "A professor thinks the answer is (B).")
- `src/generate_cots.py` - Generates chain-of-thought responses for hinted/unhinted questions, calculates hint effectiveness

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
- `deepseek/deepseek-r1` - DeepSeek R1 (used in paper)
- `deepseek/deepseek-chat` - DeepSeek Chat
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Create hinted dataset (outputs to data/professor_hinted_mmlu.json)
python src/create_professor_hinted_mmlu.py

# 2. Generate CoT responses (outputs to data/cot_responses.json, logs to logs/)
python src/generate_cots.py
```
