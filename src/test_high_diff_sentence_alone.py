"""
Test if high cue_p diff sentences alone (without CoT context) can shift answers.

For each problem:
1. Find the sentence with highest diff (cue_p - cue_p_prev)
2. Test just that sentence appended to the question
3. Compare to baseline (no sentence)

Usage:
    python test_high_diff_sentence_alone.py --csv-path CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv --json-path CoT_Faithfulness_demo/in_text/Professor_itc_failure_threshold0.15_correct_base_no_mention.json --num-samples 20
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


def extract_answer(response_text):
    """Extract letter answer from response."""
    text_upper = response_text.strip().upper()

    # Look for explicit answer patterns
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

    # Fallback
    for char in reversed(text_upper):
        if char in ['A', 'B', 'C', 'D']:
            return char

    return None


async def resample_with_sentence(base_prompt, sentence, temperature=1.0):
    """Resample with just one sentence appended."""
    if sentence:
        prompt = f"{base_prompt}\n<think>\n{sentence}\n</think>"
    else:
        # Baseline: no sentence
        prompt = base_prompt

    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                ),
                timeout=180.0
            )
            response_text = response.choices[0].message.content
            answer = extract_answer(response_text)

            if answer is None:
                print(f"        ⚠ Failed to extract answer", flush=True)

            return answer
        except asyncio.TimeoutError:
            print(f"        ⚠ Timeout", flush=True)
            return None
        except Exception as e:
            print(f"        ⚠ Error: {type(e).__name__}: {str(e)}", flush=True)
            return None


async def test_high_diff_sentence(problem_id, base_prompt, high_diff_sentence, cue_answer, diff, num_samples=20):
    """
    Test high diff sentence alone vs baseline.

    Returns dict with baseline and sentence results.
    """
    print(f"\n  Testing problem {problem_id}")
    print(f"  High diff sentence (diff={diff:+.4f}):")
    print(f"    '{high_diff_sentence}'")

    # Baseline: no sentence
    print(f"  Baseline (no sentence)...", flush=True)
    tasks = [resample_with_sentence(base_prompt, "", temperature=1.0)
             for _ in range(num_samples)]
    baseline_answers = await asyncio.gather(*tasks)
    baseline_answers = [a for a in baseline_answers if a is not None]
    baseline_cue_pct = baseline_answers.count(cue_answer) / len(baseline_answers) if baseline_answers else 0.0
    print(f"    {baseline_cue_pct:.2%} cue answer ({len(baseline_answers)} samples)")

    # With high diff sentence
    print(f"  With high diff sentence...", flush=True)
    tasks = [resample_with_sentence(base_prompt, high_diff_sentence, temperature=1.0)
             for _ in range(num_samples)]
    sentence_answers = await asyncio.gather(*tasks)
    sentence_answers = [a for a in sentence_answers if a is not None]
    sentence_cue_pct = sentence_answers.count(cue_answer) / len(sentence_answers) if sentence_answers else 0.0
    print(f"    {sentence_cue_pct:.2%} cue answer ({len(sentence_answers)} samples)")

    delta = sentence_cue_pct - baseline_cue_pct
    print(f"  Delta: {delta:+.2%}")

    return {
        'problem_id': problem_id,
        'high_diff_sentence': high_diff_sentence,
        'diff': diff,
        'cue_answer': cue_answer,
        'baseline_cue_pct': baseline_cue_pct,
        'baseline_answers': baseline_answers,
        'sentence_cue_pct': sentence_cue_pct,
        'sentence_answers': sentence_answers,
        'delta': delta,
        'num_samples': num_samples
    }


async def main():
    parser = argparse.ArgumentParser(description='Test high diff sentences alone')
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--json-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--min-diff', type=float, default=0.35,
                       help='Minimum diff to consider (default: 0.35)')
    args = parser.parse_args()

    # Load CSV
    print(f"Loading CSV from {args.csv_path}", flush=True)
    df = pd.read_csv(args.csv_path)

    # Calculate diff
    df['diff'] = df['cue_p'] - df['cue_p_prev']

    # Load JSON
    print(f"Loading JSON from {args.json_path}", flush=True)
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    # Create lookup dict for questions
    question_lookup = {item['pn']: item['question'] for item in json_data}

    # Find high diff sentences for each problem
    print(f"\nFinding high diff sentences (min diff: {args.min_diff})...", flush=True)

    high_diff_problems = []
    for pn in df['pn'].unique():
        problem_df = df[df['pn'] == pn].sort_values('sentence_num')

        # Find max diff sentence
        max_diff_idx = problem_df['diff'].idxmax()
        max_diff_row = problem_df.loc[max_diff_idx]

        if max_diff_row['diff'] >= args.min_diff:
            high_diff_problems.append({
                'pn': pn,
                'sentence': max_diff_row['sentence'],
                'diff': max_diff_row['diff'],
                'cue_answer': max_diff_row['cue_answer'],
                'gt_answer': max_diff_row['gt_answer']
            })

    print(f"Found {len(high_diff_problems)} problems with diff >= {args.min_diff}")

    # Run experiments
    print(f"\nRunning experiments...")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples per condition: {args.num_samples}\n")

    results = []
    for i, prob in enumerate(high_diff_problems, 1):
        print(f"\n[{i}/{len(high_diff_problems)}] Problem {prob['pn']}", flush=True)

        base_prompt = question_lookup.get(prob['pn'])
        if not base_prompt:
            print(f"  ⚠ Question not found for problem {prob['pn']}, skipping")
            continue

        result = await test_high_diff_sentence(
            prob['pn'],
            base_prompt,
            prob['sentence'],
            prob['cue_answer'],
            prob['diff'],
            args.num_samples
        )
        result['gt_answer'] = prob['gt_answer']
        results.append(result)

    # Save results
    output_dir = Path(args.csv_path).parent
    output_file = output_dir / "high_diff_sentence_alone_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems tested: {len(results)}")

    # Calculate statistics
    positive_delta = sum(1 for r in results if r['delta'] > 0)
    negative_delta = sum(1 for r in results if r['delta'] < 0)
    zero_delta = sum(1 for r in results if r['delta'] == 0)

    mean_delta = sum(r['delta'] for r in results) / len(results) if results else 0
    mean_baseline = sum(r['baseline_cue_pct'] for r in results) / len(results) if results else 0
    mean_sentence = sum(r['sentence_cue_pct'] for r in results) / len(results) if results else 0

    print(f"\nMean baseline cue%: {mean_baseline:.2%}")
    print(f"Mean sentence cue%: {mean_sentence:.2%}")
    print(f"Mean delta: {mean_delta:+.2%}")
    print(f"\nDelta distribution:")
    print(f"  Positive (sentence > baseline): {positive_delta} ({positive_delta/len(results):.1%})")
    print(f"  Negative (sentence < baseline): {negative_delta} ({negative_delta/len(results):.1%})")
    print(f"  Zero: {zero_delta} ({zero_delta/len(results):.1%})")

    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
