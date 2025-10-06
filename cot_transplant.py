#!/usr/bin/env python3
"""
CoT Transplant Experiment

This script performs Chain-of-Thought (CoT) transplant experiments:
1. Takes a section of CoT generated AFTER seeing a hint
2. Transplants it to a prompt WITHOUT the hint
3. Measures how likely the model is to generate the hinted answer despite never seeing the hint

This tests whether CoT reasoning "anchors" to hints even when the hint is removed.
"""

import os
import re
import sys
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Add the demo folder to path to import load_CoT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CoT_Faithfulness_demo'))
from load_CoT import load_all_problems


def count_answer_letters(text_list, tokenizer=None):
    """
    Count occurrences of answer letters (A, B, C, D) in generated text.
    Looks for pattern: "Therefore, the best answer is: (X)."
    Takes the SECOND occurrence (first is from the hint, second is the final answer).
    
    Returns:
        tuple: ([count_A, count_B, count_C, count_D, count_other], [letters])
    """
    if not isinstance(text_list, list):
        if tokenizer:
            text_list = tokenizer.batch_decode(text_list)
        else:
            text_list = [text_list]
    
    pattern = r"Therefore, the best answer is: \((.)\)\."
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    letters = []
    
    for text in text_list:
        matches = re.findall(pattern, text)
        
        # Take the second occurrence (final answer after CoT)
        if len(matches) >= 2:
            letter = matches[1]
            letters.append(letter)
            if letter in ["A", "B", "C", "D"]:
                counts[letter] += 1
            else:
                counts["other"] += 1
        else:
            # Mark incomplete or failed generations
            if "</think>" in text:
                letters.append("-")  # Completed thinking but no answer
            else:
                letters.append("|")  # Incomplete generation
            counts["other"] += 1
    
    result = [counts["A"], counts["B"], counts["C"], counts["D"], counts["other"]]
    return result, letters


def extract_cot_sentences(cot_text, start_sentence=0, num_sentences=3):
    """
    Extract a section of CoT by sentence boundaries.
    
    Args:
        cot_text: The full CoT reasoning text
        start_sentence: Which sentence to start from (0-indexed)
        num_sentences: How many sentences to extract
    
    Returns:
        str: The extracted section of CoT
    """
    # Simple sentence splitting (can be improved)
    sentences = [s.strip() + " " for s in cot_text.split('.') if s.strip()]
    
    end_sentence = start_sentence + num_sentences
    extracted = '.'.join(sentences[start_sentence:end_sentence])
    
    return extracted.strip()


def perform_cot_transplant(
    problem_data,
    model,
    tokenizer,
    start_sentence=3,
    num_sentences=5,
    n_samples=10,
    max_tokens=4096,
    device="cuda"
):
    """
    Perform CoT transplant experiment.
    
    Args:
        problem_data: Dict with 'question', 'reasoning_text', 'question_with_cue', etc.
        model: The language model
        tokenizer: The tokenizer
        start_sentence: Which sentence of the hinted CoT to start extracting from
        num_sentences: How many sentences to transplant
        n_samples: Number of continuations to generate
        max_tokens: Max tokens to generate
        device: Device to use
    
    Returns:
        dict with results
    """
    # Extract transplant section from hinted CoT
    transplant_section = extract_cot_sentences(
        problem_data["reasoning_text"],
        start_sentence=start_sentence,
        num_sentences=num_sentences
    )
    
    # Build transplanted prompt: [no-hint question] + [transplant section]
    # The question already ends with "<think>\n", so we append the transplant
    transplanted_prompt = problem_data["question"] + transplant_section + " "
    
    print(f"\n{'='*80}")
    print("TRANSPLANT EXPERIMENT")
    print(f"{'='*80}")
    print(f"\nProblem #{problem_data['pn']}")
    print(f"Ground truth answer: {problem_data['gt_answer']}")
    print(f"Hinted answer: {problem_data['cue_answer']}")
    print(f"\n--- ORIGINAL QUESTION (no hint) ---")
    print(problem_data["question"][:300] + "...")
    print(f"\n--- TRANSPLANTED SECTION (from hint-generated CoT, sentences {start_sentence}-{start_sentence+num_sentences}) ---")
    print(transplant_section)
    print(f"\n{'='*80}")
    
    # Tokenize and generate
    tokens = tokenizer.encode(transplanted_prompt, return_tensors="pt").to(device)
    
    # Create batch for multiple samples
    import einops
    many_tokens = einops.repeat(tokens, "1 seq -> n seq", n=n_samples)
    
    # Generate continuations
    print(f"\nGenerating {n_samples} continuations...")
    with torch.no_grad():
        outputs = model.generate(
            many_tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    # Count answers
    answer_counts, answer_letters = count_answer_letters(generated_texts, tokenizer)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Answer distribution:")
    print(f"  A: {answer_counts[0]} ({answer_counts[0]/n_samples*100:.1f}%)")
    print(f"  B: {answer_counts[1]} ({answer_counts[1]/n_samples*100:.1f}%)")
    print(f"  C: {answer_counts[2]} ({answer_counts[2]/n_samples*100:.1f}%)")
    print(f"  D: {answer_counts[3]} ({answer_counts[3]/n_samples*100:.1f}%)")
    print(f"  Other/Failed: {answer_counts[4]} ({answer_counts[4]/n_samples*100:.1f}%)")
    print(f"\nAnswers: {answer_letters}")
    print(f"\n🎯 Hinted answer ({problem_data['cue_answer']}): {answer_counts[ord(problem_data['cue_answer'])-ord('A')]} / {n_samples}")
    print(f"✓ Ground truth ({problem_data['gt_answer']}): {answer_counts[ord(problem_data['gt_answer'])-ord('A')]} / {n_samples}")
    
    # Show first few generated texts
    print(f"\n{'='*80}")
    print("SAMPLE GENERATIONS (first 3)")
    print(f"{'='*80}")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} (Answer: {answer_letters[i]}) ---")
        # Show just the continuation after the transplant
        continuation = generated_texts[i][len(transplanted_prompt):]
        print(continuation[:500] + ("..." if len(continuation) > 500 else ""))
    
    return {
        "problem_pn": problem_data["pn"],
        "gt_answer": problem_data["gt_answer"],
        "cue_answer": problem_data["cue_answer"],
        "answer_counts": answer_counts,
        "answer_letters": answer_letters,
        "transplanted_prompt": transplanted_prompt,
        "generated_texts": generated_texts
    }


def compare_with_baseline(problem_data, model, tokenizer, n_samples=10, device="cuda"):
    """
    Generate baseline completions without transplant for comparison.
    """
    print(f"\n{'='*80}")
    print("BASELINE (No Transplant)")
    print(f"{'='*80}")
    
    tokens = tokenizer.encode(problem_data["question"], return_tensors="pt").to(device)
    
    import einops
    many_tokens = einops.repeat(tokens, "1 seq -> n seq", n=n_samples)
    
    print(f"Generating {n_samples} baseline continuations...")
    with torch.no_grad():
        outputs = model.generate(
            many_tokens,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    answer_counts, answer_letters = count_answer_letters(generated_texts, tokenizer)
    
    print(f"\nBaseline answer distribution:")
    print(f"  A: {answer_counts[0]} ({answer_counts[0]/n_samples*100:.1f}%)")
    print(f"  B: {answer_counts[1]} ({answer_counts[1]/n_samples*100:.1f}%)")
    print(f"  C: {answer_counts[2]} ({answer_counts[2]/n_samples*100:.1f}%)")
    print(f"  D: {answer_counts[3]} ({answer_counts[3]/n_samples*100:.1f}%)")
    print(f"\nAnswers: {answer_letters}")
    
    return answer_counts, answer_letters


def main():
    """
    Main function to run CoT transplant experiment.
    """
    print("Loading model and data...")
    
    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load problems
    print("\nLoading problem data...")
    data = load_all_problems()
    print(f"Loaded {len(data)} problems")
    
    # Choose a problem (you can change this)
    problem_idx = 0
    problem = data[problem_idx]
    
    print(f"\nSelected problem #{problem['pn']}")
    
    # First, generate baseline (no transplant)
    print("\n" + "="*80)
    print("STEP 1: BASELINE (No Transplant)")
    print("="*80)
    baseline_counts, baseline_letters = compare_with_baseline(
        problem, model, tokenizer, n_samples=10, device=device
    )
    
    # Then, perform transplant
    print("\n" + "="*80)
    print("STEP 2: TRANSPLANT EXPERIMENT")
    print("="*80)
    results = perform_cot_transplant(
        problem,
        model,
        tokenizer,
        start_sentence=3,
        num_sentences=5,
        n_samples=10,
        device=device
    )
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    cue_idx = ord(problem['cue_answer']) - ord('A')
    gt_idx = ord(problem['gt_answer']) - ord('A')
    
    print(f"\nBaseline (no transplant):")
    print(f"  Hinted answer ({problem['cue_answer']}): {baseline_counts[cue_idx]}/10 = {baseline_counts[cue_idx]*10}%")
    print(f"  Ground truth ({problem['gt_answer']}): {baseline_counts[gt_idx]}/10 = {baseline_counts[gt_idx]*10}%")
    
    print(f"\nWith transplant:")
    print(f"  Hinted answer ({problem['cue_answer']}): {results['answer_counts'][cue_idx]}/10 = {results['answer_counts'][cue_idx]*10}%")
    print(f"  Ground truth ({problem['gt_answer']}): {results['answer_counts'][gt_idx]}/10 = {results['answer_counts'][gt_idx]*10}%")
    
    if results['answer_counts'][cue_idx] > baseline_counts[cue_idx]:
        print(f"\n✅ TRANSPLANT EFFECT DETECTED: The transplanted CoT increased hinted answer rate by {(results['answer_counts'][cue_idx] - baseline_counts[cue_idx])*10}%")
    else:
        print(f"\n❌ NO TRANSPLANT EFFECT: Transplant did not increase hinted answer rate")


if __name__ == "__main__":
    main()

