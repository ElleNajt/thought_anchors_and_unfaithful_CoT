"""
Calculate token-by-token logit differences between hinted and unhinted prompts.

For a given CoT, we:
1. Get logits for each token with hinted prompt
2. Get logits for each token with unhinted prompt (same CoT)
3. Calculate logit differences
4. Identify tokens with largest differences

Usage:
    python analyze_token_logits.py --input data/TIMESTAMP/filtered_unfaithful_cots.json --question-id XXXX --sample-num X
"""

import json
import argparse
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


def get_logprobs(prompt, max_tokens=4096):
    """
    Get token-by-token log probabilities from the API.

    Returns:
        List of dicts with {token, logprob, top_logprobs}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic
        logprobs=True,
        top_logprobs=5  # Get top 5 alternatives for each token
    )

    # Extract logprobs from response
    logprobs_data = []
    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
        for token_data in response.choices[0].logprobs.content:
            logprobs_data.append({
                'token': token_data.token,
                'logprob': token_data.logprob,
                'top_logprobs': [
                    {'token': alt.token, 'logprob': alt.logprob}
                    for alt in token_data.top_logprobs
                ] if token_data.top_logprobs else []
            })

    return {
        'text': response.choices[0].message.content,
        'logprobs': logprobs_data
    }


def calculate_logit_diffs(hinted_logprobs, unhinted_logprobs):
    """
    Calculate token-by-token logit differences.

    Returns:
        List of dicts with {position, token, hinted_logprob, unhinted_logprob, diff}
    """
    diffs = []

    # Match tokens by position (assuming they align)
    min_len = min(len(hinted_logprobs), len(unhinted_logprobs))

    for i in range(min_len):
        h_token = hinted_logprobs[i]
        u_token = unhinted_logprobs[i]

        # Only compare if tokens match
        if h_token['token'] == u_token['token']:
            diff = h_token['logprob'] - u_token['logprob']
            diffs.append({
                'position': i,
                'token': h_token['token'],
                'hinted_logprob': h_token['logprob'],
                'unhinted_logprob': u_token['logprob'],
                'diff': diff,
                'abs_diff': abs(diff)
            })

    return diffs


def analyze_cot_logits(question_data, sample, base_prompt, hinted_prompt):
    """
    Analyze logit differences for a specific CoT.
    """
    cot_text = sample['cot']

    print(f"Question: {question_data['question'][:80]}...")
    print(f"Sample {sample['sample_num']}, Answer: {sample['answer']}")
    print(f"CoT length: {len(cot_text)} chars\n")

    # Create prompts with the CoT continuation
    # Format: {prompt}\n\n<think>\n{cot}
    hinted_full = f"{hinted_prompt}\n\n<think>\n{cot_text}"
    unhinted_full = f"{base_prompt}\n\n<think>\n{cot_text}"

    print("Getting logprobs for hinted version...")
    hinted_result = get_logprobs(hinted_full, max_tokens=10)  # Just get next few tokens

    print("Getting logprobs for unhinted version...")
    unhinted_result = get_logprobs(unhinted_full, max_tokens=10)

    # Calculate differences
    diffs = calculate_logit_diffs(hinted_result['logprobs'], unhinted_result['logprobs'])

    # Sort by absolute difference
    diffs.sort(key=lambda x: x['abs_diff'], reverse=True)

    return {
        'question_id': question_data['question_id'],
        'sample_num': sample['sample_num'],
        'cot': cot_text,
        'hinted_continuation': hinted_result['text'],
        'unhinted_continuation': unhinted_result['text'],
        'token_diffs': diffs
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze token logit differences')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered_unfaithful_cots.json')
    parser.add_argument('--question-idx', type=int, default=0,
                       help='Question index (default: 0 = first question)')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index within question (default: 0 = first sample)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    # Load data
    print(f"Loading from {input_path}")
    with open(input_path, 'r') as f:
        unfaithful_data = json.load(f)

    # Get question and sample
    question = unfaithful_data[args.question_idx]
    sample = question['unfaithful_samples'][args.sample_idx]

    print(f"\nAnalyzing Question {args.question_idx}, Sample {args.sample_idx}")
    print(f"Subject: {question['subject']}")
    print(f"Correct: {question['correct_answer']}, Hinted: {question['hinted_answer']}\n")

    # Load dataset to get base/hinted prompts
    dataset_path = Path(__file__).parent.parent / "data" / "professor_hinted_mmlu.json"
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Find matching question
    base_prompt = None
    hinted_prompt = None
    for item in dataset:
        if item['id'] == question['question_id']:
            base_prompt = item['base_prompt']
            hinted_prompt = item['hinted_prompt']
            break

    if not base_prompt:
        print(f"Error: Could not find prompts for question {question['question_id']}")
        return

    # Analyze
    result = analyze_cot_logits(question, sample, base_prompt, hinted_prompt)

    # Print top differences
    print("\n" + "="*80)
    print("TOP TOKEN LOGIT DIFFERENCES (hinted - unhinted):")
    print("="*80)

    for i, token_diff in enumerate(result['token_diffs'][:20], 1):
        print(f"{i:2d}. Position {token_diff['position']:3d}: '{token_diff['token']}'")
        print(f"    Hinted logprob:   {token_diff['hinted_logprob']:7.3f}")
        print(f"    Unhinted logprob: {token_diff['unhinted_logprob']:7.3f}")
        print(f"    Difference:       {token_diff['diff']:+7.3f}")

    # Show continuations
    print("\n" + "="*80)
    print("CONTINUATIONS:")
    print("="*80)
    print(f"\nHinted:   {result['hinted_continuation'][:200]}...")
    print(f"\nUnhinted: {result['unhinted_continuation'][:200]}...")

    # Save results
    output_file = output_dir / f"logit_analysis_q{args.question_idx}_s{args.sample_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
