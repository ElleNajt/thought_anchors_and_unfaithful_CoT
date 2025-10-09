#!/usr/bin/env python3
"""
Compare two transplantation methods:
1. Cumulative: Transplant full CoT from hint run up to sentence n
2. Single-sentence: Transplant only the n-th sentence from hint run

This script measures the change in cue_p (probability of hinted answer) for both methods
and creates a scatterplot to visualize the comparison.
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
        answer_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]
        answer_logits[answer] = logits[answer_token_id].item()
    
    # Also compute softmax probabilities
    answer_token_ids = [tokenizer.encode(ans, add_special_tokens=False)[0] for ans in ['A', 'B', 'C', 'D']]
    answer_logits_tensor = torch.tensor([logits[tid] for tid in answer_token_ids])
    answer_probs = torch.softmax(answer_logits_tensor, dim=0).cpu().numpy()
    
    answer_probs_dict = {answer: prob for answer, prob in zip(['A', 'B', 'C', 'D'], answer_probs)}
    
    return answer_logits, answer_probs_dict


def split_into_sentences(text):
    """Split text into sentences (simple version)."""
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


def get_baseline_cue_p(problem, model, tokenizer, device):
    """
    Get baseline cue_p (probability of cue answer) with no transplantation.
    Just the question without any CoT, but at the answer decision point.
    
    Returns:
        float: baseline probability of cue answer
    """
    question = problem['question']
    cue_answer = problem['cue_answer']
    
    # Add the answer prompt to measure at the decision point (no CoT reasoning)
    baseline_prompt = question + "</think>\nTherefore, the best answer is: ("
    
    _, answer_probs = get_answer_logits_at_position(model, tokenizer, baseline_prompt, device)
    
    return answer_probs[cue_answer]


def build_cumulative_transplant(problem, num_sentences):
    """
    Build a transplanted prompt with cumulative hint sentences:
    [no-hint question] + [hint sentences 0 to num_sentences-1] + [answer prompt]
    
    Returns:
        str: transplanted prompt
    """
    hint_reasoning = problem['reasoning_text']
    hint_sentences = split_into_sentences(hint_reasoning)
    
    question = problem['question']
    transplanted_text = question  # Already has <think>\n at the end
    
    # Add cumulative hint sentences
    for idx in range(num_sentences):
        if idx < len(hint_sentences):
            transplanted_text += hint_sentences[idx] + " "
    
    # Add the closing tag and answer prompt so we measure at the decision point
    transplanted_text += "</think>\nTherefore, the best answer is: ("
    
    return transplanted_text


def build_single_sentence_transplant(problem, sentence_idx):
    """
    Build a transplanted prompt with only a single sentence from hint CoT:
    [no-hint question] + [hint sentence at sentence_idx] + [answer prompt]
    
    Returns:
        str: transplanted prompt
    """
    hint_reasoning = problem['reasoning_text']
    hint_sentences = split_into_sentences(hint_reasoning)
    
    question = problem['question']
    transplanted_text = question  # Already has <think>\n at the end
    
    # Add only the specific sentence
    if sentence_idx < len(hint_sentences):
        transplanted_text += hint_sentences[sentence_idx] + " "
    
    # Add the closing tag and answer prompt so we measure at the decision point
    transplanted_text += "</think>\nTherefore, the best answer is: ("
    
    return transplanted_text


def analyze_problem_comparison(problem, model, tokenizer, device):
    """
    Analyze a single problem: compare cumulative vs single-sentence transplantation.
    
    For each sentence position:
    - Measure cue_p with cumulative transplant (sentences 0 to n)
    - Measure cue_p with single sentence transplant (just sentence n)
    
    Returns:
        dict with comparison data for each sentence
    """
    pn = problem['pn']
    hint_reasoning_text = problem['reasoning_text']
    cue_answer = problem['cue_answer']
    gt_answer = problem['gt_answer']
    
    # Split hint reasoning into sentences
    hint_sentences = split_into_sentences(hint_reasoning_text)
    
    # Get baseline cue_p (no transplantation)
    baseline_cue_p = get_baseline_cue_p(problem, model, tokenizer, device)
    
    results = {
        'pn': pn,
        'cue_answer': cue_answer,
        'gt_answer': gt_answer,
        'baseline_cue_p': baseline_cue_p,
        'num_sentences': len(hint_sentences),
        'sentence_comparisons': []
    }
    
    # For each sentence position, compare both methods
    for sent_idx in range(len(hint_sentences)):
        # Method 1: Cumulative transplantation (sentences 0 to sent_idx)
        cumulative_prompt = build_cumulative_transplant(problem, sent_idx + 1)
        _, cumulative_probs = get_answer_logits_at_position(
            model, tokenizer, cumulative_prompt, device
        )
        cumulative_cue_p = cumulative_probs[cue_answer]
        
        # Method 2: Single-sentence transplantation (only sentence sent_idx)
        single_prompt = build_single_sentence_transplant(problem, sent_idx)
        _, single_probs = get_answer_logits_at_position(
            model, tokenizer, single_prompt, device
        )
        single_cue_p = single_probs[cue_answer]
        
        # Calculate deltas from baseline
        cumulative_delta = cumulative_cue_p - baseline_cue_p
        single_delta = single_cue_p - baseline_cue_p
        
        comparison_data = {
            'sentence_idx': sent_idx,
            'sentence_text': hint_sentences[sent_idx],
            'normalized_position': (sent_idx + 1) / len(hint_sentences),
            'baseline_cue_p': baseline_cue_p,
            'cumulative_cue_p': cumulative_cue_p,
            'single_cue_p': single_cue_p,
            'cumulative_delta': cumulative_delta,
            'single_delta': single_delta
        }
        
        results['sentence_comparisons'].append(comparison_data)
    
    return results


def plot_comparison_scatter(all_results, output_dir):
    """
    Create a scatterplot comparing cumulative vs single-sentence delta_cue_p.
    
    Each point represents one sentence from one problem.
    X-axis: delta_cue_p for cumulative transplantation
    Y-axis: delta_cue_p for single-sentence transplantation
    """
    # Collect all data points
    cumulative_deltas = []
    single_deltas = []
    problem_numbers = []
    sentence_indices = []
    normalized_positions = []
    
    for result in all_results:
        pn = result['pn']
        for comp in result['sentence_comparisons']:
            cumulative_deltas.append(comp['cumulative_delta'])
            single_deltas.append(comp['single_delta'])
            problem_numbers.append(pn)
            sentence_indices.append(comp['sentence_idx'])
            normalized_positions.append(comp['normalized_position'])
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color points by normalized position (early vs late in CoT)
    scatter = ax.scatter(
        cumulative_deltas,
        single_deltas,
        c=normalized_positions,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add diagonal line (where cumulative == single)
    all_deltas = cumulative_deltas + single_deltas
    min_val = min(all_deltas)
    max_val = max(all_deltas)
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, alpha=0.7, label='y=x (equal effect)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Position in CoT (0=start, 1=end)', fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Δ cue_p: Cumulative Transplantation\n(sentences 0 to n)', fontsize=13)
    ax.set_ylabel('Δ cue_p: Single-Sentence Transplantation\n(only sentence n)', fontsize=13)
    ax.set_title('Comparison of Transplantation Methods\n'
                 f'Total data points: {len(cumulative_deltas)} sentences from {len(all_results)} problems',
                 fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add reference lines at 0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'transplant_method_comparison_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nScatterplot saved to {output_path}")
    plt.close()
    
    return fig


def plot_comparison_by_position(all_results, output_dir):
    """
    Create additional plots showing how the two methods compare across CoT positions.
    """
    # Bin by normalized position
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    binned_data = defaultdict(lambda: {'cumulative': [], 'single': []})
    
    for result in all_results:
        for comp in result['sentence_comparisons']:
            norm_pos = comp['normalized_position']
            bin_idx = min(int(norm_pos * num_bins), num_bins - 1)
            
            binned_data[bin_idx]['cumulative'].append(comp['cumulative_delta'])
            binned_data[bin_idx]['single'].append(comp['single_delta'])
    
    # Calculate averages
    bin_centers = []
    avg_cumulative = []
    avg_single = []
    std_cumulative = []
    std_single = []
    
    for bin_idx in sorted(binned_data.keys()):
        data = binned_data[bin_idx]
        
        bin_centers.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
        avg_cumulative.append(np.mean(data['cumulative']))
        avg_single.append(np.mean(data['single']))
        std_cumulative.append(np.std(data['cumulative']))
        std_single.append(np.std(data['single']))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(bin_centers, avg_cumulative, 'o-', label='Cumulative Transplant', 
            linewidth=3, markersize=8, color='blue', alpha=0.8)
    ax.fill_between(bin_centers,
                     np.array(avg_cumulative) - np.array(std_cumulative),
                     np.array(avg_cumulative) + np.array(std_cumulative),
                     color='blue', alpha=0.2)
    
    ax.plot(bin_centers, avg_single, 's-', label='Single-Sentence Transplant', 
            linewidth=3, markersize=8, color='red', alpha=0.8)
    ax.fill_between(bin_centers,
                     np.array(avg_single) - np.array(std_single),
                     np.array(avg_single) + np.array(std_single),
                     color='red', alpha=0.2)
    
    ax.set_xlabel('Normalized CoT Position (0 to 1)', fontsize=12)
    ax.set_ylabel('Average Δ cue_p', fontsize=12)
    ax.set_title('Comparison of Transplantation Methods Across CoT Position\n'
                 f'Averaged across {len(all_results)} problems',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    output_path = output_dir / 'transplant_method_comparison_by_position.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Position comparison plot saved to {output_path}")
    plt.close()
    
    return fig


def save_results_to_csv(all_results, output_path):
    """
    Save all comparison results to a CSV file.
    """
    rows = []
    
    for result in all_results:
        pn = result['pn']
        cue_answer = result['cue_answer']
        gt_answer = result['gt_answer']
        baseline_cue_p = result['baseline_cue_p']
        
        for comp in result['sentence_comparisons']:
            row = {
                'pn': pn,
                'cue_answer': cue_answer,
                'gt_answer': gt_answer,
                'sentence_idx': comp['sentence_idx'],
                'normalized_position': comp['normalized_position'],
                'sentence_text': comp['sentence_text'],
                'baseline_cue_p': baseline_cue_p,
                'cumulative_cue_p': comp['cumulative_cue_p'],
                'single_cue_p': comp['single_cue_p'],
                'cumulative_delta': comp['cumulative_delta'],
                'single_delta': comp['single_delta']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"Total rows: {len(df)}")
    
    return df


def print_summary_statistics(results_df):
    """
    Print summary statistics about the comparison.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal problems analyzed: {results_df['pn'].nunique()}")
    print(f"Total sentences analyzed: {len(results_df)}")
    
    print("\n--- Cumulative Transplantation ---")
    print(f"Mean Δ cue_p: {results_df['cumulative_delta'].mean():.4f}")
    print(f"Median Δ cue_p: {results_df['cumulative_delta'].median():.4f}")
    print(f"Std Δ cue_p: {results_df['cumulative_delta'].std():.4f}")
    print(f"Max Δ cue_p: {results_df['cumulative_delta'].max():.4f}")
    print(f"Min Δ cue_p: {results_df['cumulative_delta'].min():.4f}")
    
    print("\n--- Single-Sentence Transplantation ---")
    print(f"Mean Δ cue_p: {results_df['single_delta'].mean():.4f}")
    print(f"Median Δ cue_p: {results_df['single_delta'].median():.4f}")
    print(f"Std Δ cue_p: {results_df['single_delta'].std():.4f}")
    print(f"Max Δ cue_p: {results_df['single_delta'].max():.4f}")
    print(f"Min Δ cue_p: {results_df['single_delta'].min():.4f}")
    
    print("\n--- Comparison ---")
    # Count how many times cumulative > single
    cumulative_larger = (results_df['cumulative_delta'] > results_df['single_delta']).sum()
    single_larger = (results_df['single_delta'] > results_df['cumulative_delta']).sum()
    equal = (results_df['cumulative_delta'] == results_df['single_delta']).sum()
    
    print(f"Cumulative > Single: {cumulative_larger} ({100*cumulative_larger/len(results_df):.1f}%)")
    print(f"Single > Cumulative: {single_larger} ({100*single_larger/len(results_df):.1f}%)")
    print(f"Equal: {equal} ({100*equal/len(results_df):.1f}%)")
    
    # Correlation
    correlation = results_df['cumulative_delta'].corr(results_df['single_delta'])
    print(f"\nCorrelation between cumulative and single deltas: {correlation:.4f}")
    
    # Identify sentences where single >> cumulative (outliers)
    # Ensure numeric types and handle any conversion issues
    results_df['single_advantage'] = pd.to_numeric(
        results_df['single_delta'], errors='coerce'
    ) - pd.to_numeric(
        results_df['cumulative_delta'], errors='coerce'
    )
    
    # Filter out NaN/inf values before sorting
    valid_df = results_df[
        np.isfinite(results_df['single_advantage']) & 
        np.isfinite(results_df['cumulative_delta']) & 
        np.isfinite(results_df['single_delta'])
    ].copy()
    
    # Ensure the column is float type (not object)
    valid_df['single_advantage'] = valid_df['single_advantage'].astype(float)
    
    if len(valid_df) > 0:
        top_single_advantage = valid_df.nlargest(5, 'single_advantage')[
            ['pn', 'sentence_idx', 'sentence_text', 'cumulative_delta', 'single_delta', 'single_advantage']
        ]
        
        print("\n--- Top 5 sentences where single-sentence had highest advantage ---")
        for idx, row in top_single_advantage.iterrows():
            print(f"\nProblem {row['pn']}, Sentence {row['sentence_idx']}:")
            print(f"  Cumulative Δ: {row['cumulative_delta']:.4f}, Single Δ: {row['single_delta']:.4f}, Advantage: {row['single_advantage']:.4f}")
            print(f"  Text: {row['sentence_text'][:100]}...")
    else:
        print("\n--- No valid data points found for comparison ---")


def main():
    """Main function."""
    # Configuration
    THRESHOLD = 0.2  # Threshold for loading problems
    MAX_PROBLEMS = None  # Number of problems to analyze (set to small number for testing, None for all)
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    # Output directory
    output_dir = Path(__file__).parent / "transplant_comparison_output"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("TRANSPLANTATION METHOD COMPARISON")
    print("Cumulative vs Single-Sentence Transplantation")
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
            print(f"Using first {MAX_PROBLEMS} problems for analysis")
        
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
            
            results = analyze_problem_comparison(
                problem, model, tokenizer, device
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing problem {problem.get('pn', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    csv_path = output_dir / "transplant_method_comparison.csv"
    results_df = save_results_to_csv(all_results, csv_path)
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    plot_comparison_scatter(all_results, output_dir)
    plot_comparison_by_position(all_results, output_dir)
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Scatterplot: transplant_method_comparison_scatter.png")
    print(f"  - Position plot: transplant_method_comparison_by_position.png")
    print(f"  - CSV: transplant_method_comparison.csv")
    print("\nKey insights:")
    print("  - Points above the diagonal: single-sentence has larger effect")
    print("  - Points below the diagonal: cumulative has larger effect")
    print("  - Color indicates position in CoT (darker = later)")


if __name__ == "__main__":
    main()

