"""
Transplant resampling: Measure cumulative hint effect sentence-by-sentence.

For each unfaithful CoT:
1. Split into sentences
2. For each position i, truncate to sentences 0..i
3. Transplant to base prompt (no hint): {question} <think> {sentences 0..i}
4. Resample N times and measure % giving hinted answer
5. Plot cumulative effect curve

Usage:
    python transplant_resampling.py --input data/TIMESTAMP/filtered_unfaithful_cots.json --num-samples 20

Output:
    transplant_results.json with cumulative effect curves
"""

import json
import argparse
import asyncio
import re
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


def split_into_sentences(text):
    """
    Split CoT into sentences.

    Simple heuristic: split on . ! ? followed by space/newline.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


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
    # Build transplant prompt
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
            return answer
        except asyncio.TimeoutError:
            print(f"      ⚠ Timeout")
            return None
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None


async def transplant_resample_sample(base_prompt, cot_text, hinted_answer, num_samples=20):
    """
    Perform transplant resampling for a single unfaithful CoT.

    Returns:
        List of dicts with {position, num_sentences, hinted_pct, answers}
    """
    sentences = split_into_sentences(cot_text)

    if len(sentences) == 0:
        print(f"      ⚠ No sentences found in CoT")
        return []

    print(f"      CoT has {len(sentences)} sentences")

    results = []

    # Position 0: No CoT (baseline)
    print(f"      Position 0 (baseline, no CoT)...")
    tasks = [resample_with_truncated_cot(base_prompt, "", temperature=1.0)
             for _ in range(num_samples)]
    answers = await asyncio.gather(*tasks)
    answers = [a for a in answers if a is not None]
    hinted_pct = answers.count(hinted_answer) / len(answers) if answers else 0.0
    results.append({
        "position": 0,
        "num_sentences": 0,
        "hinted_pct": hinted_pct,
        "num_samples": len(answers),
        "answers": answers
    })
    print(f"        {hinted_pct:.2%} hinted answer")

    # Positions 1..n: Truncated CoT
    for i in range(1, len(sentences) + 1):
        truncated_cot = " ".join(sentences[:i])
        print(f"      Position {i} ({i}/{len(sentences)} sentences)...")

        tasks = [resample_with_truncated_cot(base_prompt, truncated_cot, temperature=1.0)
                 for _ in range(num_samples)]
        answers = await asyncio.gather(*tasks)
        answers = [a for a in answers if a is not None]
        hinted_pct = answers.count(hinted_answer) / len(answers) if answers else 0.0

        results.append({
            "position": i,
            "num_sentences": i,
            "hinted_pct": hinted_pct,
            "num_samples": len(answers),
            "answers": answers
        })
        print(f"        {hinted_pct:.2%} hinted answer")

    return results


async def process_question(question_data, base_prompt, num_samples):
    """Process all unfaithful samples for a question."""
    print(f"\n=== {question_data['subject']}: {question_data['question'][:60]}... ===", flush=True)
    print(f"Correct: {question_data['correct_answer']}, Hinted: {question_data['hinted_answer']}", flush=True)
    print(f"{question_data['num_unfaithful']} unfaithful samples", flush=True)

    transplant_results = []

    for sample_idx, sample in enumerate(question_data['unfaithful_samples']):
        print(f"\n  Sample {sample['sample_num']} (#{sample_idx+1}/{question_data['num_unfaithful']}):")

        curve = await transplant_resample_sample(
            base_prompt,
            sample['cot'],
            question_data['hinted_answer'],
            num_samples
        )

        transplant_results.append({
            'sample_num': sample['sample_num'],
            'original_answer': sample['answer'],
            'curve': curve
        })

    return {
        'question_id': question_data['question_id'],
        'subject': question_data['subject'],
        'question': question_data['question'],
        'correct_answer': question_data['correct_answer'],
        'hinted_answer': question_data['hinted_answer'],
        'transplant_results': transplant_results
    }


async def main():
    import sys

    parser = argparse.ArgumentParser(description='Transplant resampling for unfaithful CoTs')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered_unfaithful_cots.json')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of resamples per truncation point (default: 20)')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Max questions to process (default: all)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    # Load unfaithful CoTs
    print(f"Loading unfaithful CoTs from {input_path}", flush=True)
    sys.stdout.flush()
    with open(input_path, 'r') as f:
        unfaithful_data = json.load(f)

    # Load original data to get base prompts
    original_path = output_dir / "cot_responses.json"
    print(f"Loading original data from {original_path}", flush=True)
    with open(original_path, 'r') as f:
        original_data = json.load(f)

    # Create mapping question_id -> base_prompt
    base_prompts = {}
    dataset_path = Path(__file__).parent.parent / "data" / "professor_hinted_mmlu.json"
    print(f"Loading dataset from {dataset_path}", flush=True)
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    for item in dataset:
        base_prompts[item['id']] = item['base_prompt']

    # Process questions
    if args.max_questions:
        unfaithful_data = unfaithful_data[:args.max_questions]

    print(f"\nProcessing {len(unfaithful_data)} questions with {args.num_samples} samples per truncation point", flush=True)
    print(f"Using model: {MODEL_NAME}", flush=True)
    print(f"Max concurrent: {MAX_CONCURRENT}\n", flush=True)

    all_results = []

    for q in unfaithful_data:
        base_prompt = base_prompts.get(q['question_id'])
        if not base_prompt:
            print(f"⚠ No base prompt found for question {q['question_id']}, skipping", flush=True)
            continue

        print(f"\nStarting question {q['question_id']}...", flush=True)
        result = await process_question(q, base_prompt, args.num_samples)
        all_results.append(result)

        # Save incrementally
        output_file = output_dir / "transplant_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved progress to {output_file}", flush=True)

    print(f"\n✓ Saved results to {output_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
