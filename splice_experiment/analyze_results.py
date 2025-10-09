#!/usr/bin/env python3
"""
Analyze results from the splice experiment.

This script loads the results and provides statistical analysis and visualizations
comparing the spliced distributions to the original hinted/unhinted distributions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: Path) -> Dict:
    """Load the splice results JSON."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def load_original_distributions(cot_data_path: Path) -> Dict[int, Dict]:
    """
    Load the original hinted and unhinted distributions from CoT data.
    
    Returns:
        Dict mapping question_id to {hinted_dist, unhinted_dist, ...}
    """
    with open(cot_data_path, 'r') as f:
        cot_data = json.load(f)
    
    distributions = {}
    for item in cot_data:
        distributions[item['question_id']] = {
            'hinted_dist': item['hinted_distribution'],
            'unhinted_dist': item['no_hint_distribution'],
            'hinted_answer': item['hinted_answer'],
            'correct_answer': item['correct_answer']
        }
    
    return distributions


def compute_statistics(results: List[Dict], original_dists: Dict[int, Dict]) -> pd.DataFrame:
    """
    Compute key statistics for each splice point.
    
    Returns:
        DataFrame with statistics for each splice point
    """
    stats = []
    
    for result in results:
        problem_num = result['problem_number']
        
        # Get original distributions
        if problem_num not in original_dists:
            continue
        
        orig = original_dists[problem_num]
        hinted_answer = result['hinted_answer']
        
        # Compare splice distribution to originals
        splice_dist = result['answer_distribution']
        hinted_dist = orig['hinted_dist']
        unhinted_dist = orig['unhinted_dist']
        
        # Calculate probabilities for hinted answer
        splice_hinted_p = splice_dist.get(hinted_answer, 0)
        orig_hinted_p = hinted_dist.get(hinted_answer, 0)
        orig_unhinted_p = unhinted_dist.get(hinted_answer, 0)
        
        # Calculate KL divergence or distance metrics
        # For simplicity, we'll use L1 distance (Manhattan)
        def l1_distance(dist1, dist2):
            return sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in ['A', 'B', 'C', 'D'])
        
        dist_to_hinted = l1_distance(splice_dist, hinted_dist)
        dist_to_unhinted = l1_distance(splice_dist, unhinted_dist)
        
        # Determine which distribution splice is closer to
        closer_to = "hinted" if dist_to_hinted < dist_to_unhinted else "unhinted"
        
        stats.append({
            'problem_number': problem_num,
            'sentence_number': result['sentence_number'],
            'delta_cue_p': result['delta_cue_p'],
            'splice_hinted_answer_p': splice_hinted_p,
            'original_hinted_answer_p': orig_hinted_p,
            'original_unhinted_answer_p': orig_unhinted_p,
            'drop_from_hinted': orig_hinted_p - splice_hinted_p,
            'shift_from_unhinted': splice_hinted_p - orig_unhinted_p,
            'l1_dist_to_hinted': dist_to_hinted,
            'l1_dist_to_unhinted': dist_to_unhinted,
            'closer_to': closer_to,
            'anchor_broken': splice_hinted_p < orig_hinted_p - 0.1  # >10% drop
        })
    
    return pd.DataFrame(stats)


def plot_answer_distributions(results: List[Dict], original_dists: Dict[int, Dict], 
                               output_dir: Path):
    """
    Create visualizations comparing answer distributions.
    """
    output_dir.mkdir(exist_ok=True)
    
    # 1. For each splice point, show three distributions side by side
    for result in results[:10]:  # Limit to top 10 for readability
        problem_num = result['problem_number']
        sentence_num = result['sentence_number']
        
        if problem_num not in original_dists:
            continue
        
        orig = original_dists[problem_num]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Problem {problem_num}, Sentence {sentence_num} (delta={result['delta_cue_p']:.3f})")
        
        # Plot distributions
        answers = ['A', 'B', 'C', 'D']
        
        # Unhinted original
        axes[0].bar(answers, [orig['unhinted_dist'].get(a, 0) for a in answers], color='blue', alpha=0.7)
        axes[0].set_title('Original Unhinted')
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel('Probability')
        axes[0].axhline(y=0.25, color='gray', linestyle='--', alpha=0.3)
        
        # Spliced
        axes[1].bar(answers, [result['answer_distribution'].get(a, 0) for a in answers], color='orange', alpha=0.7)
        axes[1].set_title('Spliced (U continuation)')
        axes[1].set_ylim(0, 1)
        axes[1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.3)
        
        # Hinted original
        axes[2].bar(answers, [orig['hinted_dist'].get(a, 0) for a in answers], color='red', alpha=0.7)
        axes[2].set_title('Original Hinted')
        axes[2].set_ylim(0, 1)
        axes[2].axhline(y=0.25, color='gray', linestyle='--', alpha=0.3)
        
        # Highlight hinted answer
        hinted_answer = result['hinted_answer']
        for ax in axes:
            ax.axvline(x=answers.index(hinted_answer), color='green', linestyle='--', 
                      linewidth=2, alpha=0.5, label='Hinted answer')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"problem_{problem_num}_sent_{sentence_num}_distribution.png", dpi=150)
        plt.close()
    
    print(f"Saved individual distribution plots to {output_dir}")


def plot_summary_statistics(stats_df: pd.DataFrame, output_dir: Path):
    """Create summary plots."""
    output_dir.mkdir(exist_ok=True)
    
    # 1. Scatter: delta_cue_p vs drop from hinted
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(stats_df['delta_cue_p'], stats_df['drop_from_hinted'], 
                        c=stats_df['anchor_broken'].map({True: 'red', False: 'blue'}),
                        alpha=0.6, s=100)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='10% drop threshold')
    ax.set_xlabel('Delta cue_p (original jump)', fontsize=12)
    ax.set_ylabel('Drop from hinted probability', fontsize=12)
    ax.set_title('Anchor Breaking: Delta vs Probability Drop', fontsize=14)
    ax.legend(['0 change', '10% threshold', 'Anchor broken', 'Anchor held'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "delta_vs_drop_scatter.png", dpi=150)
    plt.close()
    
    # 2. Bar chart: average probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(stats_df))
    width = 0.25
    
    ax.bar([i - width for i in x], stats_df['original_unhinted_answer_p'], 
           width, label='Original Unhinted', color='blue', alpha=0.7)
    ax.bar([i for i in x], stats_df['splice_hinted_answer_p'], 
           width, label='Spliced', color='orange', alpha=0.7)
    ax.bar([i + width for i in x], stats_df['original_hinted_answer_p'], 
           width, label='Original Hinted', color='red', alpha=0.7)
    
    ax.set_xlabel('Splice Point', fontsize=12)
    ax.set_ylabel('Probability of Hinted Answer', fontsize=12)
    ax.set_title('Hinted Answer Probability Across Conditions', fontsize=14)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{r['problem_number']}\nS{r['sentence_number']}" 
                        for _, r in stats_df.iterrows()], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "probability_comparison_bars.png", dpi=150)
    plt.close()
    
    # 3. Distribution of which condition splice is closer to
    fig, ax = plt.subplots(figsize=(8, 6))
    closer_counts = stats_df['closer_to'].value_counts()
    ax.bar(closer_counts.index, closer_counts.values, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Which Distribution is Splice Closer To?', fontsize=14)
    ax.set_xlabel('Condition', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "splice_similarity.png", dpi=150)
    plt.close()
    
    print(f"Saved summary plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze splice experiment results')
    parser.add_argument('results_json', type=str, 
                       help='Path to splice_results.json')
    parser.add_argument('--cot-data', type=str, required=True,
                       help='Path to original CoT data (for comparison)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for visualizations (default: same as results)')
    
    args = parser.parse_args()
    
    # Load data
    results_path = Path(args.results_json)
    results = load_results(results_path)
    
    print(f"Loaded {len(results)} splice results")
    
    # Load original distributions
    original_dists = load_original_distributions(Path(args.cot_data))
    print(f"Loaded original distributions for {len(original_dists)} problems")
    
    # Compute statistics
    stats_df = compute_statistics(results, original_dists)
    
    print("\n=== Statistics Summary ===")
    print(f"Total splice points: {len(stats_df)}")
    print(f"Anchors broken (>10% drop): {stats_df['anchor_broken'].sum()} ({stats_df['anchor_broken'].mean()*100:.1f}%)")
    print(f"Average drop from hinted: {stats_df['drop_from_hinted'].mean():.3f}")
    print(f"Average shift from unhinted: {stats_df['shift_from_unhinted'].mean():.3f}")
    print(f"\nSplice closer to unhinted: {(stats_df['closer_to'] == 'unhinted').sum()} ({(stats_df['closer_to'] == 'unhinted').mean()*100:.1f}%)")
    print(f"Splice closer to hinted: {(stats_df['closer_to'] == 'hinted').sum()} ({(stats_df['closer_to'] == 'hinted').mean()*100:.1f}%)")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "visualizations"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save statistics CSV
    stats_df.to_csv(output_dir / "statistics.csv", index=False)
    print(f"\nSaved statistics to {output_dir / 'statistics.csv'}")
    
    # Create visualizations
    plot_answer_distributions(results, original_dists, output_dir)
    plot_summary_statistics(stats_df, output_dir)
    
    print(f"\n✓ Analysis complete! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

