#!/usr/bin/env python3
"""
Track answer option logits (A, B, C, D) over time for each problem
and plot them alongside cue_p by normalized CoT position.

This script uses CUMULATIVE TRANSPLANTATION:
1. Loads problems with hint-influenced CoT reasoning
2. For each position (0 to 1), transplants an increasing PREFIX of hint-influenced CoT
   - Position 0.2: transplant first 20% of hint CoT
   - Position 0.5: transplant first 50% of hint CoT  
   - Position 1.0: transplant entire hint CoT
3. Gets the logits for each answer option (A, B, C, D) after each cumulative transplantation
4. Plots the logits over normalized position along with cue_p progression
5. Shows how answer preferences build up as more hint-influenced sentences are added
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


def build_transplanted_prompt(problem, num_sentences):
    """
    Build a transplanted prompt with CUMULATIVE hint sentences:
    [no-hint question] + [hint sentences 0 to num_sentences-1]
    
    This transplants an increasing PREFIX of the hint-influenced CoT.
    
    Args:
        problem: Problem data dict with both 'reasoning_text' (hint) and 'base_reasoning_text' (non-hint)
        num_sentences: How many hint sentences to transplant (0 to num_sentences-1)
    
    Returns:
        tuple: (transplanted_prompt, transplanted_sentences, num_sentences_transplanted)
    """
    # Get the hint-influenced reasoning
    hint_reasoning = problem['reasoning_text']
    
    # Split into sentences
    hint_sentences = split_into_sentences(hint_reasoning)
    
    # Build the transplanted prompt:
    # [question] + [hint sentences 0 to num_sentences-1]
    # Note: question already ends with "<think>\n"
    question = problem['question']
    transplanted_text = question  # Already has <think>\n at the end
    
    # Add cumulative hint sentences from 0 to num_sentences-1
    transplanted_sents = []
    for idx in range(num_sentences):
        if idx < len(hint_sentences):
            transplanted_text += hint_sentences[idx] + " "
            transplanted_sents.append(hint_sentences[idx])
    
    return transplanted_text, transplanted_sents, len(transplanted_sents)


def analyze_problem_logits(problem, model, tokenizer, device, cue_p_dict=None):
    """
    Analyze a single problem: track logits for each answer option over sentences.
    
    Uses CUMULATIVE TRANSPLANTATION: For each position, transplants hint sentences 0 to n
    and measures the resulting logits. This shows how answer preferences build up as
    more hint-influenced sentences are added.
    
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
    hint_reasoning_text = problem['reasoning_text']
    cue_answer = problem['cue_answer']
    gt_answer = problem['gt_answer']
    
    # Split hint reasoning into sentences
    hint_sentences = split_into_sentences(hint_reasoning_text)
    
    results = {
        'pn': pn,
        'cue_answer': cue_answer,
        'gt_answer': gt_answer,
        'question': question,
        'num_sentences': len(hint_sentences),
        'sentence_data': []
    }
    
    # Track logits at each position using CUMULATIVE TRANSPLANTATION
    # Position 0: transplant sentences 0
    # Position 1: transplant sentences 0-1
    # Position 2: transplant sentences 0-2, etc.
    
    for num_sents in range(1, len(hint_sentences) + 1):
        # Build transplanted prompt with cumulative hint sentences 0 to num_sents-1
        transplanted_prompt, transplanted_sents, actual_num_sents = build_transplanted_prompt(
            problem, num_sents
        )
        
        if actual_num_sents == 0:  # Skip if no sentences to transplant
            continue
        
        # Get answer logits after transplanting this many sentences
        answer_logits, answer_probs = get_answer_logits_at_position(
            model, tokenizer, transplanted_prompt, device
        )
        
        # Get cue_p if available (cue_p is indexed by sentence_num, which is num_sents-1)
        cue_p = None
        if cue_p_dict is not None:
            # Use the last sentence position for cue_p lookup
            cue_p = cue_p_dict.get((pn, num_sents - 1), None)
        
        sentence_data = {
            'num_sentences_transplanted': actual_num_sents,
            'normalized_position': actual_num_sents / len(hint_sentences) if len(hint_sentences) > 0 else 0,
            'last_sentence': transplanted_sents[-1] if transplanted_sents else "",
            'logits': answer_logits,
            'probs': answer_probs,
            'cue_p': cue_p
        }
        
        results['sentence_data'].append(sentence_data)
    
    return results


def plot_problem_logits(results, output_dir):
    """
    Plot logits for all answer options over normalized sentence position, along with cue_p.
    Shows results from CUMULATIVE TRANSPLANTATION (increasing prefixes of hint CoT).
    
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
    
    # Extract data for plotting - use normalized position for x-axis
    normalized_positions = [d['normalized_position'] for d in sentence_data]
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
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Raw logits
    ax1 = axes[0]
    ax1.plot(normalized_positions, logits_A, 'o-', label='A', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(normalized_positions, logits_B, 's-', label='B', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(normalized_positions, logits_C, '^-', label='C', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(normalized_positions, logits_D, 'd-', label='D', linewidth=2, markersize=6, alpha=0.7)
    
    # Highlight the cue answer
    for idx, norm_pos in enumerate(normalized_positions):
        if cue_answer == 'A':
            ax1.scatter(norm_pos, logits_A[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'B':
            ax1.scatter(norm_pos, logits_B[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'C':
            ax1.scatter(norm_pos, logits_C[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
        elif cue_answer == 'D':
            ax1.scatter(norm_pos, logits_D[idx], s=150, facecolors='none', 
                       edgecolors='red', linewidths=2, zorder=10)
    
    ax1.set_xlabel('Normalized CoT Position (0 to 1)', fontsize=12)
    ax1.set_ylabel('Logit Value', fontsize=12)
    ax1.set_title(f'Problem {pn}: Answer Logits with Cumulative Transplantation\n'
                 f'Cue Answer: {cue_answer} (red circles), GT: {gt_answer}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Plot 2: Cue_p (if available)
    if has_cue_p:
        ax2 = axes[1]
        
        # Plot cue_p only
        valid_cue_p = [(pos, cp) for pos, cp in zip(normalized_positions, cue_ps) if not np.isnan(cp)]
        if valid_cue_p:
            positions_cue, cps_cue = zip(*valid_cue_p)
            ax2.plot(positions_cue, cps_cue, 'o-', label='cue_p (reference data)', 
                    linewidth=3, markersize=8, color='purple', alpha=0.8)
        
        ax2.set_xlabel('Normalized CoT Position (0 to 1)', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title(f'Problem {pn}: cue_p (Reference Data)', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
    else:
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'No cue_p data available', 
                ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'problem_{pn}_answer_logits_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_aggregate_trends(all_results, output_dir):
    """
    Create an aggregate plot showing average trends across all problems.
    Shows how cued answer logits/probs increase while others decrease, alongside cue_p.
    Uses normalized positions (0 to 1) to align CoTs of different lengths.
    
    Args:
        all_results: List of results dicts from all problems
        output_dir: Directory to save plots
    """
    # Collect data organized by normalized position (binned)
    # We'll bin normalized positions into 20 buckets for aggregation
    num_bins = 20
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    data_by_bin = defaultdict(lambda: {
        'cued_logits': [],
        'cued_probs': [],
        'other_logits': [],
        'other_probs': [],
        'cue_ps': []
    })
    
    for result in all_results:
        cue_answer = result['cue_answer']
        
        for sent_data in result['sentence_data']:
            norm_pos = sent_data['normalized_position']
            
            # Assign to bin
            bin_idx = min(int(norm_pos * num_bins), num_bins - 1)
            
            # Get cued answer logit/prob
            cued_logit = sent_data['logits'][cue_answer]
            cued_prob = sent_data['probs'][cue_answer]
            
            # Get other answers' logits/probs
            other_answers = [ans for ans in ['A', 'B', 'C', 'D'] if ans != cue_answer]
            other_logits = [sent_data['logits'][ans] for ans in other_answers]
            other_probs = [sent_data['probs'][ans] for ans in other_answers]
            
            data_by_bin[bin_idx]['cued_logits'].append(cued_logit)
            data_by_bin[bin_idx]['cued_probs'].append(cued_prob)
            data_by_bin[bin_idx]['other_logits'].extend(other_logits)
            data_by_bin[bin_idx]['other_probs'].extend(other_probs)
            
            if sent_data['cue_p'] is not None:
                data_by_bin[bin_idx]['cue_ps'].append(sent_data['cue_p'])
    
    # Compute averages for each bin
    bin_centers = []
    avg_cued_logits = []
    avg_cued_probs = []
    avg_other_logits = []
    avg_other_probs = []
    avg_cue_ps = []
    std_cued_probs = []
    std_other_probs = []
    std_cue_ps = []
    
    for bin_idx in sorted(data_by_bin.keys()):
        data = data_by_bin[bin_idx]
        
        # Skip bins with no data
        if not data['cued_logits']:
            continue
        
        # Bin center
        bin_centers.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
        
        avg_cued_logits.append(np.mean(data['cued_logits']))
        avg_cued_probs.append(np.mean(data['cued_probs']))
        avg_other_logits.append(np.mean(data['other_logits']))
        avg_other_probs.append(np.mean(data['other_probs']))
        
        std_cued_probs.append(np.std(data['cued_probs']))
        std_other_probs.append(np.std(data['other_probs']))
        
        if data['cue_ps']:
            avg_cue_ps.append(np.mean(data['cue_ps']))
            std_cue_ps.append(np.std(data['cue_ps']))
        else:
            avg_cue_ps.append(np.nan)
            std_cue_ps.append(np.nan)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Average logits
    ax1 = axes[0]
    ax1.plot(bin_centers, avg_cued_logits, 'o-', label='Cued Answer', 
             linewidth=3, markersize=8, color='red', alpha=0.8)
    ax1.plot(bin_centers, avg_other_logits, 's-', label='Other Answers (avg)', 
             linewidth=3, markersize=8, color='blue', alpha=0.8)
    
    ax1.set_xlabel('Normalized CoT Position (0 to 1)', fontsize=12)
    ax1.set_ylabel('Average Logit Value', fontsize=12)
    ax1.set_title('Aggregate: Cued vs Other Answer Logits with Cumulative Transplantation\n'
                 f'Averaged across {len(all_results)} problems',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Plot 2: Cue_p only
    ax2 = axes[1]
    
    # Filter out NaN values for cue_p
    valid_indices = ~np.isnan(avg_cue_ps)
    valid_bin_centers = np.array(bin_centers)[valid_indices]
    valid_avg_cue_ps = np.array(avg_cue_ps)[valid_indices]
    valid_std_cue_ps = np.array(std_cue_ps)[valid_indices]
    
    if len(valid_bin_centers) > 0:
        # Plot cue_p from reference data only
        ax2.plot(valid_bin_centers, valid_avg_cue_ps, 'o-', 
                label='cue_p (reference data)', 
                linewidth=3, markersize=8, color='purple', alpha=0.8)
        ax2.fill_between(valid_bin_centers,
                        valid_avg_cue_ps - valid_std_cue_ps,
                        valid_avg_cue_ps + valid_std_cue_ps,
                        color='purple', alpha=0.2)
        
        ax2.set_xlabel('Normalized CoT Position (0 to 1)', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Aggregate: cue_p (Reference Data)\n'
                     'Shaded region shows ±1 standard deviation',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No cue_p reference data available', 
                ha='center', va='center', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'aggregate_transplanted_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAggregate plot saved to {output_path}")
    plt.close()
    
    return fig


def save_results_to_csv(all_results, output_path):
    """
    Save all results to a CSV file.
    Records cumulative transplantation data: number of sentences transplanted and resulting logits.
    
    Args:
        all_results: List of results dicts
        output_path: Path to save CSV
    """
    rows = []
    
    for result in all_results:
        pn = result['pn']
        cue_answer = result['cue_answer']
        gt_answer = result['gt_answer']
        num_total_sentences = result['num_sentences']
        
        for sent_data in result['sentence_data']:
            row = {
                'pn': pn,
                'cue_answer': cue_answer,
                'gt_answer': gt_answer,
                'num_sentences_transplanted': sent_data['num_sentences_transplanted'],
                'normalized_position': sent_data['normalized_position'],
                'total_sentences': num_total_sentences,
                'last_sentence': sent_data['last_sentence'],
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
    THRESHOLD = 0.3  # Threshold for loading problems
    MAX_PROBLEMS = None  # Number of problems to analyze (None for all)
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    # Output directory
    output_dir = Path(__file__).parent / "logit_analysis_transplanted_output"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("ANSWER LOGIT TRACKING ANALYSIS - CUMULATIVE TRANSPLANTATION")
    print("Transplanting increasing prefixes of hint-influenced CoT")
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
    
    # Create aggregate plot
    print("\n" + "="*80)
    print("CREATING AGGREGATE PLOT")
    print("="*80)
    
    plot_aggregate_trends(all_results, output_dir)
    
    # Save results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    csv_path = output_dir / "answer_logits_by_sentence_transplanted.csv"
    results_df = save_results_to_csv(all_results, csv_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal problems analyzed: {len(all_results)}")
    print(f"Total data points: {len(results_df)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Aggregate plot: aggregate_transplanted_trends.png")
    print(f"  - Individual plots: problem_<pn>_answer_logits_over_time.png")
    print(f"  - CSV: answer_logits_by_sentence_transplanted.csv")
    print(f"\nNote: These results use CUMULATIVE TRANSPLANTATION:")
    print(f"      - At position 0.2: Transplant first 20% of hint-influenced CoT")
    print(f"      - At position 0.5: Transplant first 50% of hint-influenced CoT")
    print(f"      - At position 1.0: Transplant entire hint-influenced CoT")
    print(f"      This shows how answer preferences build up as more hint sentences are added")
    print(f"\nThe aggregate plot shows average trends across all problems:")
    print(f"  - How cued answer logits/probs increase with more transplanted sentences")
    print(f"  - How other answers' logits/probs decrease")
    print(f"  - Comparison with reference cue_p values (from original hint-influenced runs)")
    
    print("\n" + "="*80)
    print("CUMULATIVE TRANSPLANTATION ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

