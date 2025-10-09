"""
Token-by-token resampling within a specific sentence.

Usage:
    python transplant_sentence_tokens.py --csv-path CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv --json-path CoT_Faithfulness_demo/in_text/Professor_itc_failure_threshold0.15_correct_base_no_mention.json --problem-id 288 --sentence-num 11 --num-samples 10
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
    """Split text into tokens (whitespace split)."""
    return text.split()


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


async def resample_with_cot(base_prompt, cot_prefix, temperature=1.0):
    """Resample with CoT prefix."""
    transplant_prompt = f"{base_prompt}\n<think>\n{cot_prefix}"

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

            if answer is None:
                print(f"        ⚠ Failed to extract answer. Response: {response_text[:200]}...", flush=True)

            return answer
        except asyncio.TimeoutError:
            print(f"        ⚠ Timeout", flush=True)
            return None
        except Exception as e:
            print(f"        ⚠ Error: {type(e).__name__}: {str(e)}", flush=True)
            return None


async def token_level_within_sentence(base_prompt, cot_before_sentence, target_sentence, cue_answer, num_samples=10):
    """
    Run token-by-token resampling within a specific sentence.

    Args:
        base_prompt: Question prompt
        cot_before_sentence: CoT text before the target sentence
        target_sentence: The sentence to analyze token-by-token
        cue_answer: Cue answer letter
        num_samples: Number of resamples per position

    Returns:
        List of {position, num_tokens, text, hinted_pct, answers}
    """
    tokens = split_into_tokens(target_sentence)
    print(f"      Target sentence has {len(tokens)} tokens")
    print(f"      Sentence: {target_sentence}")

    results = []

    # Position 0: CoT up to (but not including) this sentence
    print(f"\n      Position 0 (before sentence)...", flush=True)
    tasks = [resample_with_cot(base_prompt, cot_before_sentence, temperature=1.0)
             for _ in range(num_samples)]
    answers = await asyncio.gather(*tasks)
    answers = [a for a in answers if a is not None]
    hinted_pct = answers.count(cue_answer) / len(answers) if answers else 0.0
    results.append({
        "position": 0,
        "num_tokens": 0,
        "text": "",
        "hinted_pct": hinted_pct,
        "num_samples": len(answers),
        "answers": answers
    })
    print(f"        {hinted_pct:.2%} hinted ({len(answers)} samples)")

    # Token by token within the sentence
    for i in range(1, len(tokens) + 1):
        partial_sentence = " ".join(tokens[:i])
        cot_with_partial = f"{cot_before_sentence} {partial_sentence}".strip()

        print(f"      Position {i}/{len(tokens)}: '{tokens[i-1]}'...", flush=True)

        tasks = [resample_with_cot(base_prompt, cot_with_partial, temperature=1.0)
                 for _ in range(num_samples)]
        answers = await asyncio.gather(*tasks)
        answers = [a for a in answers if a is not None]
        hinted_pct = answers.count(cue_answer) / len(answers) if answers else 0.0

        results.append({
            "position": i,
            "num_tokens": i,
            "text": partial_sentence,
            "hinted_pct": hinted_pct,
            "num_samples": len(answers),
            "answers": answers
        })
        print(f"        {hinted_pct:.2%} hinted ({len(answers)} samples)")

    return results


async def main():
    parser = argparse.ArgumentParser(description='Token-level resampling within a sentence')
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--json-path', type=str, required=True)
    parser.add_argument('--problem-id', type=int, required=True)
    parser.add_argument('--sentence-num', type=int, required=True,
                       help='Sentence number to analyze (0-indexed)')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    # Load CSV
    print(f"Loading CSV from {args.csv_path}", flush=True)
    df = pd.read_csv(args.csv_path)

    problem_df = df[df['pn'] == args.problem_id].sort_values('sentence_num')
    if len(problem_df) == 0:
        print(f"Error: Problem {args.problem_id} not found")
        return

    # Get target sentence
    target_row = problem_df[problem_df['sentence_num'] == args.sentence_num]
    if len(target_row) == 0:
        print(f"Error: Sentence {args.sentence_num} not found")
        return

    target_sentence = target_row.iloc[0]['sentence']
    cue_answer = target_row.iloc[0]['cue_answer']
    cue_p = target_row.iloc[0]['cue_p']
    cue_p_prev = target_row.iloc[0]['cue_p_prev']

    # Build CoT before this sentence
    sentences_before = problem_df[problem_df['sentence_num'] < args.sentence_num]['sentence'].tolist()
    cot_before = ' '.join(sentences_before)

    print(f"\nProblem {args.problem_id}, Sentence {args.sentence_num}")
    print(f"Cue answer: {cue_answer}")
    print(f"cue_p before: {cue_p_prev:.2%}, after: {cue_p:.2%}, increase: {cue_p - cue_p_prev:+.2%}")

    # Load question
    print(f"\nLoading question from {args.json_path}", flush=True)
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)

    question_item = next((item for item in json_data if item['pn'] == args.problem_id), None)
    if not question_item:
        print(f"Error: Question not found for problem {args.problem_id}")
        return

    base_prompt = question_item['question']
    print(f"Question: {base_prompt[:150]}...")

    print(f"\nRunning token-level analysis...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Samples per position: {args.num_samples}\n", flush=True)

    # Run analysis
    curve = await token_level_within_sentence(
        base_prompt,
        cot_before,
        target_sentence,
        cue_answer,
        args.num_samples
    )

    # Save results
    output_dir = Path(args.csv_path).parent
    output_file = output_dir / f"sentence_tokens_pn{args.problem_id}_s{args.sentence_num}.json"

    result = {
        'problem_id': args.problem_id,
        'sentence_num': args.sentence_num,
        'target_sentence': target_sentence,
        'cue_answer': cue_answer,
        'cue_p_before': cue_p_prev,
        'cue_p_after': cue_p,
        'cue_p_increase': cue_p - cue_p_prev,
        'num_tokens': len(split_into_tokens(target_sentence)),
        'curve': curve,
        'num_samples_per_position': args.num_samples
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {output_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
