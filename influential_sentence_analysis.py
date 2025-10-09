#!/usr/bin/env python3
"""
Track answer option logits (A, B, C, D) over time for each problem
and plot them alongside cue_p by sentence.

This script:
1. Loads problems with CoT reasoning
2. For each sentence position, gets the logits for each answer option (A, B, C, D)
3. Plots the logits over time along with cue_p progression
4. Identifies which sentences cause the biggest shifts in answer preferences
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Add the demo folder to path
root_dir = Path(__file__).parent
demo_dir = root_dir / "CoT_Faithfulness_demo"
sys.path.insert(0, str(demo_dir))
from load_CoT import load_all_problems


def get_answer_logits_at_position(model, tokenizer, text, device):
    """
    Get the logits for answer tokens A, B, C, D at the end of the given text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The prompt text up to the current position
        device: Device to run on
    
    Returns:
        dict: {answer: logit} for each answer option A, B, C, D
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, -1, :]  # Last token's logits
    
    # Get logits for answer tokens A, B, C, D
    answer_logits = {}
    for answer in ['A', 'B', 'C', 'D']:
        # Get token ID for this answer
        answer_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]
        answer_logits[answer] = logits[answer_token_id].item()
    
    # Also compute softmax probabilities
    answer_token_ids = [tokenizer.encode(ans, add_special_tokens=False)[0] for ans in ['A', 'B', 'C', 'D']]
    answer_logits_tensor = torch.tensor([logits[tid] for tid in answer_token_ids])
    answer_probs = torch.softmax(answer_logits_tensor, dim=0).cpu().numpy()
    
    answer_probs_dict = {answer: prob for answer, prob in zip(['A', 'B', 'C', 'D'], answer_probs)}
    
    return answer_logits, answer_probs_dict


def split_into_sentences(text):
    """
    Split text into sentences (simple version).
    
    Returns:
        list of sentences
    """
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?':
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    return sentences


def analyze_problem_logits(problem, model, tokenizer, device, cue_p_dict=None):
    """
    Analyze a single problem: track logits for each answer option over sentences.
    
    Args:
        problem: Problem data dict
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        cue_p_dict: Optional dict {(pn, sentence_num): cue_p} for comparison
    
    Returns:
        dict with sentence-level logits and metadata
    """
    pn = problem['pn']
    question = problem['question']
    reasoning_text = problem['reasoning_text']
    cue_answer = problem['cue_answer']
    gt_answer = problem['gt_answer']
    
    # Split reasoning into sentences
    sentences = split_into_sentences(reasoning_text)
    
    results = {
        'pn': pn,
        'cue_answer': cue_answer,
        'gt_answer': gt_answer,
        'question': question,
        'sentences': [],
        'sentence_data': []
    }
    
    # Track logits at each sentence position
    cumulative_text = question + " <think>\n"
    
    for sent_idx, sentence in enumerate(sentences):
        cumulative_text += sentence + " "
        
        # Get answer logits at this position
        answer_logits, answer_probs = get_answer_logits_at_position(
            model, tokenizer, cumulative_text, device
        )
        
        # Get cue_p if available
        cue_p = None
        if cue_p_dict is not None:
            cue_p = cue_p_dict.get((pn, sent_idx), None)
        
        sentence_data = {
            'sentence_num': sent_idx,
            'sentence': sentence,
            'logits': answer_logits,
            'probs': answer_probs,
            'cue_p': cue_p
        }
        
        results['sentences'].append(sentence)
        results['sentence_data'].append(sentence_data)
    
    return results


def plot_problem_logits(results, output_dir):
    """
    Plot logits for all answer options over sentences, along with cue_p.
    
    Args:
        results: Results dict from analyze_problem_logits
        output_dir: Directory to save plots
    """
    pn = results['pn']
    cue_answer = results['cue_answer']
    gt_answer = results['gt_answer']
    sentence_data = results['sentence_data']
    
    if not sentence_data:
        print(f"No sentence data for problem {pn}, skipping plot")
        return
    
    # Extract data for plotting
    sentence_nums = [d['sentence_num'] for d in sentence_data]
    logits_A = [d['logits']['A'] for d in sentence_data]
    logits_B = [d['logits']['B'] for d in sentence_data]
    logits_C = [d['logits']['C'] for d in sentence_data]
    logits_D = [d['logits']['D'] for d in sentence_data]
    
    probs_A = [d['probs']['A'] for d in sentence_data]
    probs_B = [d['probs']['B'] for d in sentence_data]
    probs_C = [d['probs']['C'] for d in sentence_data]
    probs_D = [d['probs']['D'] for d in sentence_data]
    
    cue_ps = [d['cue_p'] if d['cue_p'] is not None else np.nan for d in sentence_data]
    has_cue_p = not all(np.isnan(cue_ps))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Raw logits
    ax1 = axes[0]
    ax1.plot(sentence_nums, logits_A, 'o-', label='A', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(sentence_nums, logits_B, 's-', label='B', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(sentence_nums, logits_C, '^-', label='C', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(sentence_nums, logits_D, 'd-', label='D', linewidth=2, markersize=6, alpha=0.7)
    
    # Highlight the cue answer and ground truth
    for idx, sent_num in enumerate(sentence_nums):
        if cue_answer == 'A':
            ax1.scatter(sent_num, logits_A[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'B':
            ax1.scatter(sent_num, logits_B[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'C':
            ax1.scatter(sent_num, logits_C[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'D':
            ax1.scatter(sent_num, logits_D[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
    
    ax1.set_xlabel('Sentence Number', fontsize=12)
    ax1.set_ylabel('Logit Value', fontsize=12)
    ax1.set_title(f'Problem {pn}: Answer Logits Over Time\n'
                 f'Cue Answer: {cue_answer} (red circles), Ground Truth: {gt_answer}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Probabilities (softmax over A, B, C, D)
    ax2 = axes[1]
    ax2.plot(sentence_nums, probs_A, 'o-', label='A', linewidth=2, markersize=6, alpha=0.7)
    ax2.plot(sentence_nums, probs_B, 's-', label='B', linewidth=2, markersize=6, alpha=0.7)
    ax2.plot(sentence_nums, probs_C, '^-', label='C', linewidth=2, markersize=6, alpha=0.7)
    ax2.plot(sentence_nums, probs_D, 'd-', label='D', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Sentence Number', fontsize=12)
    ax2.set_ylabel('Probability (Softmax over {A,B,C,D})', fontsize=12)
    ax2.set_title(f'Problem {pn}: Answer Probabilities Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Cue_p comparison (if available)
    if has_cue_p:
        ax3 = axes[2]
        
        # Plot cue_p
        valid_cue_p = [(sn, cp) for sn, cp in zip(sentence_nums, cue_ps) if not np.isnan(cp)]
        if valid_cue_p:
            sns_cue, cps_cue = zip(*valid_cue_p)
            ax3.plot(sns_cue, cps_cue, 'o-', label='cue_p (from data)', 
                    linewidth=3, markersize=8, color='purple', alpha=0.8)
        
        # Plot probability of cue answer from our logit calculation
        cue_answer_probs = []
        for d in sentence_data:
            cue_answer_probs.append(d['probs'][cue_answer])
        
        ax3.plot(sentence_nums, cue_answer_probs, 's--', 
                label=f'P({cue_answer}) from logits', 
                linewidth=2, markersize=6, color='orange', alpha=0.7)
        
        ax3.set_xlabel('Sentence Number', fontsize=12)
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.set_title(f'Problem {pn}: Cue_p vs Computed P({cue_answer})', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    else:
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 'No cue_p data available for comparison', 
                ha='center', va='center', fontsize=14)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'problem_{pn}_answer_logits_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def save_results_to_csv(all_results, output_path):
    """
    Save all results to a CSV file.
    
    Args:
        all_results: List of results dicts
        output_path: Path to save CSV
    """
    rows = []
    
    for result in all_results:
        pn = result['pn']
        cue_answer = result['cue_answer']
        gt_answer = result['gt_answer']
        
        for sent_data in result['sentence_data']:
            row = {
                'pn': pn,
                'cue_answer': cue_answer,
                'gt_answer': gt_answer,
                'sentence_num': sent_data['sentence_num'],
                'sentence': sent_data['sentence'],
                'logit_A': sent_data['logits']['A'],
                'logit_B': sent_data['logits']['B'],
                'logit_C': sent_data['logits']['C'],
                'logit_D': sent_data['logits']['D'],
                'prob_A': sent_data['probs']['A'],
                'prob_B': sent_data['probs']['B'],
                'prob_C': sent_data['probs']['C'],
                'prob_D': sent_data['probs']['D'],
                'cue_p_reference': sent_data['cue_p']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"Total rows: {len(df)}")
    
    return df


def load_cue_p_data(csv_path):
    """
    Load cue_p values from existing CSV file for comparison.
    
    Returns:
        dict: {(pn, sentence_num): cue_p_value}
    """
    if not Path(csv_path).exists():
        print(f"Warning: cue_p CSV not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    cue_p_dict = {}
    
    # Try different possible column names
    pn_col = 'pn' if 'pn' in df.columns else 'problem_number'
    sent_col = 'sentence_num' if 'sentence_num' in df.columns else 'sentence_number'
    cue_p_col = 'true_cue_p' if 'true_cue_p' in df.columns else 'cue_p'
    
    for _, row in df.iterrows():
        key = (int(row[pn_col]), int(row[sent_col]))
        cue_p_dict[key] = row[cue_p_col]
    
    print(f"Loaded {len(cue_p_dict)} cue_p values from {csv_path}")
    return cue_p_dict


def main():
    """Main function."""
    # Configuration
    THRESHOLD = 0.2  # Threshold for loading problems
    MAX_PROBLEMS = 5  # Number of problems to analyze (None for all)
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    # Output directory
    output_dir = Path(__file__).parent / "logit_analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("ANSWER LOGIT TRACKING ANALYSIS")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    print(f"Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load problems
    print("\nLoading problem data...")
    original_dir = os.getcwd()
    os.chdir(demo_dir)
    try:
        problems = load_all_problems(THRESHOLD)
        print(f"Loaded {len(problems)} problems")
        
        if MAX_PROBLEMS is not None:
            problems = problems[:MAX_PROBLEMS]
            print(f"Using first {MAX_PROBLEMS} problems")
        
        # Try to load cue_p data for comparison
        cue_p_csv = "faith_counterfactual_qwen-14b_demo.csv"
        cue_p_dict = load_cue_p_data(cue_p_csv)
        
    finally:
        os.chdir(original_dir)
    
    # Analyze each problem
    print("\n" + "="*80)
    print("ANALYZING PROBLEMS")
    print("="*80)
    
    all_results = []
    
    for problem in tqdm(problems, desc="Processing problems"):
        try:
            pn = problem['pn']
            print(f"\nAnalyzing problem {pn}...")
            
            results = analyze_problem_logits(
                problem, model, tokenizer, device, cue_p_dict
            )
            all_results.append(results)
            
            # Plot results
            plot_problem_logits(results, output_dir)
            
        except Exception as e:
            print(f"Error processing problem {problem.get('pn', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    csv_path = output_dir / "answer_logits_by_sentence.csv"
    results_df = save_results_to_csv(all_results, csv_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal problems analyzed: {len(all_results)}")
    print(f"Total sentences analyzed: {len(results_df)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Plots: problem_<pn>_answer_logits_over_time.png")
    print(f"  - CSV: answer_logits_by_sentence.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

