"""
Run classification using multiple prompt variations.

First generates prompt variations, then classifies sentences using each variation.

Usage:
    python run_classification_with_variations.py --csv-path CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv --json-path CoT_Faithfulness_demo/in_text/Professor_itc_failure_threshold0.15_correct_base_no_mention.json --problem-ids "3,18,26"
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import subprocess

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

CLASSIFICATION_MODEL = "anthropic/claude-3.7-sonnet"
GENERATION_MODEL = "anthropic/claude-3.7-sonnet"


def generate_prompts(output_dir, variations=None):
    """Generate prompt variations using the metaprompt."""
    print("="*80)
    print("STEP 1: Generating prompt variations")
    print("="*80)

    # Check if prompts already exist
    output_path = Path(output_dir)
    metadata_file = output_path / "prompt_variations_metadata.json"

    if metadata_file.exists():
        print(f"Found existing prompts in {output_dir}")
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
        print(f"  - {len(existing_metadata)} variations already generated")

        user_input = input("Use existing prompts? (y/n): ").strip().lower()
        if user_input == 'y':
            return existing_metadata

    # Generate new prompts
    print(f"Generating new prompts in {output_dir}...")

    cmd = ["venv/bin/python", "src/generate_classification_prompts.py", "--output-dir", output_dir]
    if variations:
        cmd.extend(["--variations", variations])

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("Prompt generation failed")

    # Load generated prompts
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def classify_sentence(question, cue_answer, cot_before, target_sentence, classification_prompt):
    """Use Claude to classify a sentence using the provided prompt."""

    # Format the classification prompt with the actual data
    prompt = classification_prompt.format(
        question=question,
        cue_answer=cue_answer,
        cot=cot_before if cot_before else "(This is the first sentence)",
        last_sentence=target_sentence
    )

    response = client.chat.completions.create(
        model=CLASSIFICATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


def run_classification_with_prompt(csv_path, json_path, problem_ids, prompt_name, prompt_content, output_dir):
    """Run classification using a specific prompt variation."""

    print(f"\n{'='*80}")
    print(f"CLASSIFYING WITH: {prompt_name}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(csv_path)
    df['diff'] = df['cue_p'] - df['cue_p_prev']

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    question_lookup = {item['pn']: item['question'] for item in json_data}

    # Setup output
    output_path = Path(output_dir) / f"classifications_{prompt_name}.json"

    all_results = []

    for problem_id in problem_ids:
        print(f"\nProblem {problem_id}")

        # Get question
        question = question_lookup.get(problem_id)
        if not question:
            print(f"  ⚠ Warning: Question not found, skipping")
            continue

        # Get all sentences for this problem
        problem_df = df[df['pn'] == problem_id].sort_values('sentence_num')
        if len(problem_df) == 0:
            print(f"  ⚠ Warning: No data found, skipping")
            continue

        cue_answer = problem_df.iloc[0]['cue_answer']
        print(f"  Sentences: {len(problem_df)}")

        problem_results = {
            'problem_id': int(problem_id),
            'question': question,
            'cue_answer': cue_answer,
            'sentences': []
        }

        # Classify each sentence
        sentences = []
        for idx, row in problem_df.iterrows():
            sentence_num = int(row['sentence_num'])
            sentence = row['sentence']
            diff = float(row['diff'])
            cue_p = float(row['cue_p'])
            cue_p_prev = float(row['cue_p_prev'])

            # Build CoT up to (but not including) this sentence
            cot_before = ' '.join(sentences)

            # Classify
            classification = classify_sentence(question, cue_answer, cot_before, sentence, prompt_content)

            # Rate limiting
            time.sleep(1.5)

            # Extract classification label and confidence
            classification_lines = classification.split('\n')
            label = "IDK"
            confidence = None
            for line in classification_lines:
                if line.startswith("CLASSIFICATION:"):
                    label = line.replace("CLASSIFICATION:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    # Extract just the number
                    try:
                        confidence = int(''.join(filter(str.isdigit, confidence_str.split()[0])))
                    except (ValueError, IndexError):
                        confidence = None

            print(f"  [{sentence_num}] {label} (confidence: {confidence})")

            problem_results['sentences'].append({
                'sentence_num': sentence_num,
                'sentence': sentence,
                'diff': diff,
                'cue_p': cue_p,
                'cue_p_prev': cue_p_prev,
                'classification_label': label,
                'classification_confidence': confidence,
                'classification_full': classification
            })

            # Add this sentence to the CoT for next iteration
            sentences.append(sentence)

        all_results.append(problem_results)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run classification with multiple prompt variations')
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--json-path', type=str, required=True)
    parser.add_argument('--problem-ids', type=str, required=True,
                       help='Comma-separated problem IDs (e.g., "3,18,26")')
    parser.add_argument('--prompt-dir', type=str, default="prompts",
                       help='Directory for generated prompts')
    parser.add_argument('--output-dir', type=str, default="CoT_Faithfulness_demo/prompt_comparison",
                       help='Output directory for classification results')
    parser.add_argument('--variations', type=str, default=None,
                       help='Comma-separated variation names to use (default: all)')
    args = parser.parse_args()

    # Parse problem IDs
    problem_ids = [int(pid.strip()) for pid in args.problem_ids.split(',')]

    # Step 1: Generate or load prompts
    prompt_metadata = generate_prompts(args.prompt_dir, args.variations)

    # Step 2: Run classification with each prompt
    print(f"\n{'='*80}")
    print("STEP 2: Running classifications")
    print(f"{'='*80}")
    print(f"Problems: {problem_ids}")
    print(f"Prompt variations: {len(prompt_metadata)}")
    print(f"Model: {CLASSIFICATION_MODEL}\n")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_variation_results = {}

    for i, prompt_meta in enumerate(prompt_metadata, 1):
        variation_name = prompt_meta['variation_name']
        prompt_content = prompt_meta['prompt']

        print(f"\n[{i}/{len(prompt_metadata)}] Running with '{variation_name}' prompt")

        results = run_classification_with_prompt(
            args.csv_path,
            args.json_path,
            problem_ids,
            variation_name,
            prompt_content,
            args.output_dir
        )

        all_variation_results[variation_name] = results

    # Step 3: Generate comparison summary
    print(f"\n{'='*80}")
    print("STEP 3: Comparison summary")
    print(f"{'='*80}")

    # Compare classifications across prompts
    comparison_data = []

    for problem_id in problem_ids:
        # Get all sentences for this problem
        df = pd.read_csv(args.csv_path)
        problem_df = df[df['pn'] == problem_id].sort_values('sentence_num')

        for sentence_num in problem_df['sentence_num'].unique():
            sentence_data = {
                'problem_id': int(problem_id),
                'sentence_num': int(sentence_num)
            }

            # Get classification from each variation
            for variation_name, results in all_variation_results.items():
                problem_result = next((r for r in results if r['problem_id'] == problem_id), None)
                if problem_result:
                    sentence_result = next((s for s in problem_result['sentences']
                                          if s['sentence_num'] == sentence_num), None)
                    if sentence_result:
                        sentence_data[variation_name] = sentence_result['classification_label']

            comparison_data.append(sentence_data)

    # Save comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_path / "classification_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)

    print(f"✓ Saved comparison to {comparison_file}")

    # Calculate agreement statistics
    print("\nAgreement statistics:")

    variation_names = list(all_variation_results.keys())
    if len(variation_names) >= 2:
        for i, row in comparison_df.iterrows():
            labels = [row[v] for v in variation_names if v in row and pd.notna(row[v])]
            if len(set(labels)) == 1:
                agreement = "Full agreement"
            else:
                agreement = f"Disagreement: {', '.join(set(labels))}"

        # Count full agreements
        full_agreements = 0
        for i, row in comparison_df.iterrows():
            labels = [row[v] for v in variation_names if v in row and pd.notna(row[v])]
            if len(set(labels)) == 1:
                full_agreements += 1

        agreement_pct = full_agreements / len(comparison_df) * 100
        print(f"  Full agreement: {full_agreements}/{len(comparison_df)} ({agreement_pct:.1f}%)")

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
