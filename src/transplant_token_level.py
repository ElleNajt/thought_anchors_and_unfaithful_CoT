"""
Token-by-token transplant resampling.

Instead of sentence-level, measure how the hinted answer probability
changes as we add each token of the CoT.

Usage:
    python transplant_token_level.py --input data/TIMESTAMP/filtered_unfaithful_cots.json --question-idx 0 --sample-idx 0 --num-samples 5
"""

import json
import argparse
import asyncio
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
    """
    Split CoT into tokens (simple whitespace split for now).

    For more accurate tokenization, could use tiktoken, but
    whitespace is sufficient for this analysis.
    """
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

    Prompt format: {base_question}\n\n<think>\n{truncated_cot}
    """
    transplant_prompt = f"{base_prompt}\n\n<think>\n{truncated_cot}"

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
    parser = argparse.ArgumentParser(description='Token-level transplant resampling')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered_unfaithful_cots.json')
    parser.add_argument('--question-idx', type=int, default=0,
                       help='Question index (default: 0)')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index (default: 0)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of resamples per position (default: 10)')
    parser.add_argument('--stride', type=int, default=5,
                       help='Sample every N tokens (default: 5)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    # Load data
    print(f"Loading from {input_path}", flush=True)
    with open(input_path, 'r') as f:
        unfaithful_data = json.load(f)

    # Get question and sample
    question = unfaithful_data[args.question_idx]
    sample = question['unfaithful_samples'][args.sample_idx]

    print(f"\nAnalyzing Question {args.question_idx}, Sample {args.sample_idx}", flush=True)
    print(f"Subject: {question['subject']}", flush=True)
    print(f"Correct: {question['correct_answer']}, Hinted: {question['hinted_answer']}", flush=True)

    # Load dataset to get base prompt
    dataset_path = Path(__file__).parent.parent / "data" / "professor_hinted_mmlu.json"
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    base_prompt = None
    for item in dataset:
        if item['id'] == question['question_id']:
            base_prompt = item['base_prompt']
            break

    if not base_prompt:
        print(f"Error: Could not find base prompt for question {question['question_id']}")
        return

    print(f"\nRunning token-level transplant resampling...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Max concurrent: {MAX_CONCURRENT}\n", flush=True)

    # Run analysis
    curve = await transplant_token_level(
        base_prompt,
        sample['cot'],
        question['hinted_answer'],
        num_samples=args.num_samples,
        stride=args.stride
    )

    # Save results
    result = {
        'question_id': question['question_id'],
        'question_idx': args.question_idx,
        'sample_idx': args.sample_idx,
        'sample_num': sample['sample_num'],
        'subject': question['subject'],
        'correct_answer': question['correct_answer'],
        'hinted_answer': question['hinted_answer'],
        'cot': sample['cot'],
        'curve': curve,
        'num_tokens': len(split_into_tokens(sample['cot'])),
        'stride': args.stride,
        'num_samples_per_position': args.num_samples
    }

    output_file = output_dir / f"token_level_q{args.question_idx}_s{args.sample_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {output_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
