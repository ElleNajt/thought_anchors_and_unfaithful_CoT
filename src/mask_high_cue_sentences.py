"""
Mask high cue_p sentences and compare to masking random sentences.

For each CoT:
1. Identify sentence with highest cue_p (largest increase in hinted_pct)
2. Mask it with "[Sentence Omitted.]"
3. Measure how cue_p changes at the end of the CoT
4. Compare to masking random sentences (baseline)

Usage:
    python mask_high_cue_sentences.py --input data/TIMESTAMP/filtered_unfaithful_cots.json --transplant-results data/TIMESTAMP/transplant_results.json --question-idx 0 --sample-idx 0 --num-samples 10 --num-random-trials 5
"""

import json
import argparse
import asyncio
import re
import random
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
    """Split CoT into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
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


def identify_highest_cue_sentence(curve):
    """
    Identify sentence with highest cue_p contribution.

    Args:
        curve: List of {position, num_sentences, hinted_pct, ...}

    Returns:
        (sentence_idx, cue_p_increase) or None
    """
    if len(curve) < 2:
        return None

    max_increase = -float('inf')
    max_idx = None

    for i in range(1, len(curve)):
        increase = curve[i]['hinted_pct'] - curve[i-1]['hinted_pct']
        if increase > max_increase:
            max_increase = increase
            max_idx = i - 1  # 0-indexed sentence that caused this increase

    return (max_idx, max_increase) if max_idx is not None else None


def mask_sentence(sentences, mask_idx):
    """Replace sentence at mask_idx with '[Sentence Omitted.]'"""
    masked_sentences = sentences.copy()
    masked_sentences[mask_idx] = "[Sentence Omitted.]"
    return masked_sentences


async def resample_with_cot(base_prompt, cot_text, temperature=1.0):
    """Resample with given CoT."""
    transplant_prompt = f"{base_prompt}\n\n<think>\n{cot_text}"

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
            return None
        except Exception as e:
            return None


async def measure_masking_effect(base_prompt, sentences, mask_idx, hinted_answer, num_samples):
    """
    Measure cue_p when masking a specific sentence.

    Returns:
        (hinted_pct, answers)
    """
    masked_sentences = mask_sentence(sentences, mask_idx)
    masked_cot = " ".join(masked_sentences)

    tasks = [resample_with_cot(base_prompt, masked_cot, temperature=1.0)
             for _ in range(num_samples)]
    answers = await asyncio.gather(*tasks)
    answers = [a for a in answers if a is not None]
    hinted_pct = answers.count(hinted_answer) / len(answers) if answers else None

    return (hinted_pct, answers)


async def analyze_masking_experiment(
    base_prompt,
    cot_text,
    curve,
    hinted_answer,
    num_samples,
    num_random_trials
):
    """
    Run masking experiment:
    1. Identify highest cue_p sentence
    2. Mask it and measure effect
    3. Mask random sentences as baseline
    4. Compare

    Returns:
        dict with experiment results
    """
    sentences = split_into_sentences(cot_text)

    # Get baseline cue_p (full CoT, no masking)
    baseline_position = next((pos for pos in curve if pos['num_sentences'] == len(sentences)), None)
    baseline_cue_p = baseline_position['hinted_pct'] if baseline_position else None

    print(f"      Full CoT baseline cue_p: {baseline_cue_p:.2%}" if baseline_cue_p is not None else "      No baseline found")

    # Identify highest cue_p sentence
    highest = identify_highest_cue_sentence(curve)
    if not highest:
        print("      ⚠ Could not identify highest cue_p sentence")
        return None

    high_idx, high_increase = highest
    print(f"      Highest cue_p sentence: #{high_idx} (increase: {high_increase:+.4f})")
    print(f"      Sentence text: {sentences[high_idx][:100]}...")

    # Mask highest cue_p sentence
    print(f"      Masking highest cue_p sentence and resampling {num_samples}x...")
    high_masked_pct, high_masked_answers = await measure_masking_effect(
        base_prompt, sentences, high_idx, hinted_answer, num_samples
    )

    if high_masked_pct is not None:
        print(f"        Masked high cue_p → {high_masked_pct:.2%} hinted")

    # Mask random sentences (multiple trials)
    print(f"      Masking random sentences ({num_random_trials} trials, {num_samples} samples each)...")
    random_results = []

    for trial in range(num_random_trials):
        random_idx = random.randint(0, len(sentences) - 1)
        print(f"        Trial {trial+1}: masking sentence #{random_idx}...")

        random_masked_pct, random_masked_answers = await measure_masking_effect(
            base_prompt, sentences, random_idx, hinted_answer, num_samples
        )

        if random_masked_pct is not None:
            print(f"          → {random_masked_pct:.2%} hinted")
            random_results.append({
                'trial': trial,
                'masked_idx': random_idx,
                'hinted_pct': random_masked_pct,
                'answers': random_masked_answers
            })

    # Calculate average random masking effect
    if random_results:
        avg_random_pct = sum(r['hinted_pct'] for r in random_results) / len(random_results)
        print(f"      Average random masking: {avg_random_pct:.2%} hinted")
    else:
        avg_random_pct = None

    return {
        'num_sentences': len(sentences),
        'baseline_cue_p': baseline_cue_p,
        'highest_cue_sentence': {
            'idx': high_idx,
            'text': sentences[high_idx],
            'cue_p_increase': high_increase,
            'masked_hinted_pct': high_masked_pct,
            'masked_answers': high_masked_answers,
            'effect': (baseline_cue_p - high_masked_pct) if (baseline_cue_p is not None and high_masked_pct is not None) else None
        },
        'random_masking': {
            'trials': random_results,
            'avg_hinted_pct': avg_random_pct,
            'avg_effect': (baseline_cue_p - avg_random_pct) if (baseline_cue_p is not None and avg_random_pct is not None) else None
        },
        'comparison': {
            'high_vs_random_diff': (high_masked_pct - avg_random_pct) if (high_masked_pct is not None and avg_random_pct is not None) else None
        }
    }


async def main():
    parser = argparse.ArgumentParser(description='Mask high cue_p sentences experiment')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered_unfaithful_cots.json')
    parser.add_argument('--transplant-results', type=str, required=True,
                       help='Path to transplant_results.json (from transplant_resampling.py)')
    parser.add_argument('--question-idx', type=int, default=0,
                       help='Question index (default: 0)')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index (default: 0)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of resamples per masking condition (default: 10)')
    parser.add_argument('--num-random-trials', type=int, default=5,
                       help='Number of random sentence masking trials (default: 5)')
    args = parser.parse_args()

    input_path = Path(args.input)
    transplant_path = Path(args.transplant_results)
    output_dir = input_path.parent

    # Load data
    print(f"Loading unfaithful CoTs from {input_path}", flush=True)
    with open(input_path, 'r') as f:
        unfaithful_data = json.load(f)

    print(f"Loading transplant results from {transplant_path}", flush=True)
    with open(transplant_path, 'r') as f:
        transplant_data = json.load(f)

    # Get question and sample
    question = unfaithful_data[args.question_idx]
    sample = question['unfaithful_samples'][args.sample_idx]

    print(f"\nAnalyzing Question {args.question_idx}, Sample {args.sample_idx}", flush=True)
    print(f"Subject: {question['subject']}", flush=True)
    print(f"Correct: {question['correct_answer']}, Hinted: {question['hinted_answer']}", flush=True)

    # Find matching transplant curve
    transplant_question = transplant_data[args.question_idx]
    transplant_sample = transplant_question['transplant_results'][args.sample_idx]
    curve = transplant_sample['curve']

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

    print(f"\nRunning masking experiment...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Num samples per condition: {args.num_samples}", flush=True)
    print(f"Num random trials: {args.num_random_trials}\n", flush=True)

    # Run experiment
    result = await analyze_masking_experiment(
        base_prompt,
        sample['cot'],
        curve,
        question['hinted_answer'],
        args.num_samples,
        args.num_random_trials
    )

    if not result:
        print("Experiment failed")
        return

    # Save results
    output_data = {
        'question_id': question['question_id'],
        'question_idx': args.question_idx,
        'sample_idx': args.sample_idx,
        'sample_num': sample['sample_num'],
        'subject': question['subject'],
        'correct_answer': question['correct_answer'],
        'hinted_answer': question['hinted_answer'],
        'cot': sample['cot'],
        'num_samples_per_condition': args.num_samples,
        'num_random_trials': args.num_random_trials,
        'result': result
    }

    output_file = output_dir / f"masking_experiment_q{args.question_idx}_s{args.sample_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    baseline = result['baseline_cue_p']
    high_pct = result['highest_cue_sentence']['masked_hinted_pct']
    high_effect = result['highest_cue_sentence']['effect']
    random_pct = result['random_masking']['avg_hinted_pct']
    random_effect = result['random_masking']['avg_effect']
    diff = result['comparison']['high_vs_random_diff']

    print(f"Baseline (full CoT):           {baseline:.2%}" if baseline is not None else "Baseline (full CoT):           N/A")
    print(f"Mask high cue_p sentence:      {high_pct:.2%} (effect: {high_effect:.2%})" if high_pct is not None and high_effect is not None else "Mask high cue_p sentence:      N/A")
    print(f"Mask random sentence (avg):    {random_pct:.2%} (effect: {random_effect:.2%})" if random_pct is not None and random_effect is not None else "Mask random sentence (avg):    N/A")
    print(f"\nDifference (high - random):    {diff:.2%}" if diff is not None else "\nDifference (high - random):    N/A")
    print("="*80)

    print(f"\n✓ Saved to {output_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
