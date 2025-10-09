"""
Token-by-token transplant resampling from CSV data.

Loads problem data from faith_counterfactual CSV and runs token-level resampling.

Usage:
    python transplant_token_level_from_csv.py --csv-path CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv --json-path CoT_Faithfulness_demo/in_text/Professor_itc_failure_threshold0.15_correct_base_no_mention.json --problem-id 1219 --num-samples 10 --stride 5
"""

import json
import argparse
import asyncio
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Rate limiting
MAX_CONCURRENT = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


def split_into_tokens(text):
    """Split CoT into tokens (simple whitespace split)."""
    return text.split()


def extract_answer(response_text):
    """Extract letter answer from response."""
    text_upper = response_text.strip().upper()

    # Look for explicit answer patterns at the end
    lines = response_text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line_clean = line.strip()
        if len(line_clean) == 1 and line_clean in ['A', 'B', 'C', 'D']:
            return line_clean
        for letter in ['A', 'B', 'C', 'D']:
            if f"answer: {letter.lower()}" in line_clean.lower():
                return letter
            if f"answer is {letter.lower()}" in line_clean.lower():
                return letter

    # Fallback: return last A/B/C/D found
    for char in reversed(text_upper):
        if char in ['A', 'B', 'C', 'D']:
            return char

    return None


async def resample_with_truncated_cot(base_prompt, truncated_cot, temperature=1.0):
    """
    Resample with transplanted CoT prefix.

    Prompt format: {base_question}\n<think>\n{truncated_cot}
    """
    transplant_prompt = f"{base_prompt}\n<think>\n{truncated_cot}"

    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": transplant_prompt}]
                ),
                timeout=180.0
            )
            response_text = response.choices[0].message.content
            answer = extract_answer(response_text)

            # Log extraction failures
            if answer is None:
                print(f"        ⚠ Failed to extract answer. Response (first 200 chars): {response_text[:200]}...", flush=True)

            return answer
        except asyncio.TimeoutError:
            print(f"        ⚠ Timeout after 180s", flush=True)
            return None
        except Exception as e:
            print(f"        ⚠ Error: {type(e).__name__}: {str(e)}", flush=True)
            return None


async def transplant_token_level(base_prompt, cot_text, hinted_answer, num_samples=10, stride=5):
    """
    Perform token-by-token transplant resampling.

    Args:
        base_prompt: The unhinted question prompt
        cot_text: The full CoT text
        hinted_answer: The hinted answer letter
        num_samples: Number of resamples per position
        stride: Sample every N tokens (to reduce API calls)

    Returns:
        List of dicts with {position, num_tokens, hinted_pct, answers}
    """
    tokens = split_into_tokens(cot_text)

    if len(tokens) == 0:
        print(f"      ⚠ No tokens found in CoT")
        return []

    print(f"      CoT has {len(tokens)} tokens")
    print(f"      Sampling every {stride} tokens with {num_samples} samples each")

    results = []

    # Position 0: No CoT (baseline)
    print(f"      Position 0 (baseline, no CoT)...", flush=True)
    tasks = [resample_with_truncated_cot(base_prompt, "", temperature=1.0)
             for _ in range(num_samples)]
    answers = await asyncio.gather(*tasks)
    answers = [a for a in answers if a is not None]
    hinted_pct = answers.count(hinted_answer) / len(answers) if answers else 0.0
    results.append({
        "position": 0,
        "num_tokens": 0,
        "hinted_pct": hinted_pct,
        "num_samples": len(answers),
        "answers": answers
    })
    print(f"        {hinted_pct:.2%} hinted answer")

    # Sample at intervals
    positions = list(range(stride, len(tokens) + 1, stride))
    # Always include the final position
    if positions[-1] != len(tokens):
        positions.append(len(tokens))

    for i in positions:
        truncated_cot = " ".join(tokens[:i])
        print(f"      Position {i} ({i}/{len(tokens)} tokens)...", flush=True)

        tasks = [resample_with_truncated_cot(base_prompt, truncated_cot, temperature=1.0)
                 for _ in range(num_samples)]
        answers = await asyncio.gather(*tasks)
        answers = [a for a in answers if a is not None]
        hinted_pct = answers.count(hinted_answer) / len(answers) if answers else 0.0

        results.append({
            "position": i,
            "num_tokens": i,
            "hinted_pct": hinted_pct,
            "num_samples": len(answers),
            "answers": answers
        })
        print(f"        {hinted_pct:.2%} hinted answer")

    return results


async def main():
    parser = argparse.ArgumentParser(description='Token-level transplant from CSV data')
    parser.add_argument('--csv-path', type=str, required=True,
                       help='Path to faith_counterfactual CSV')
    parser.add_argument('--json-path', type=str, required=True,
                       help='Path to JSON with question prompts')
    parser.add_argument('--problem-id', type=int, required=True,
                       help='Problem ID (pn) to analyze')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of resamples per position (default: 10)')
    parser.add_argument('--stride', type=int, default=5,
                       help='Sample every N tokens (default: 5)')
    args = parser.parse_args()

    # Load CSV data
    print(f"Loading CSV from {args.csv_path}", flush=True)
    df = pd.read_csv(args.csv_path)

    # Get data for this problem
    problem_df = df[df['pn'] == args.problem_id]
    if len(problem_df) == 0:
        print(f"Error: Problem ID {args.problem_id} not found in CSV")
        return

    # Reconstruct CoT from sentences
    problem_df = problem_df.sort_values('sentence_num')
    cot_text = ' '.join(problem_df['sentence'].tolist())
    gt_answer = problem_df.iloc[0]['gt_answer']
    cue_answer = problem_df.iloc[0]['cue_answer']

    print(f"\nProblem ID: {args.problem_id}")
    print(f"Ground truth: {gt_answer}, Cue answer: {cue_answer}")
    print(f"Number of sentences: {len(problem_df)}")

    # Load JSON to get question prompt
    print(f"\nLoading JSON from {args.json_path}", flush=True)
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    # Find question
    question_item = None
    for item in json_data:
        if item['pn'] == args.problem_id:
            question_item = item
            break

    if not question_item:
        print(f"Error: Could not find question for problem {args.problem_id}")
        return

    base_prompt = question_item['question']
    print(f"\nQuestion: {base_prompt[:200]}...")

    print(f"\nRunning token-level transplant resampling...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Max concurrent: {MAX_CONCURRENT}\n", flush=True)

    # Run analysis
    curve = await transplant_token_level(
        base_prompt,
        cot_text,
        cue_answer,
        num_samples=args.num_samples,
        stride=args.stride
    )

    # Save results
    output_dir = Path(args.csv_path).parent
    output_file = output_dir / f"token_level_pn{args.problem_id}.json"

    result = {
        'problem_id': args.problem_id,
        'gt_answer': gt_answer,
        'cue_answer': cue_answer,
        'cot': cot_text,
        'curve': curve,
        'num_tokens': len(split_into_tokens(cot_text)),
        'stride': args.stride,
        'num_samples_per_position': args.num_samples
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {output_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
