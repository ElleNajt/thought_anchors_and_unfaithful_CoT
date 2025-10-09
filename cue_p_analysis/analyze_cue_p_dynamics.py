#!/usr/bin/env python3
"""
Comprehensive analysis of cue_p dynamics across sentence progression.

This script analyzes how cue_p (probability of outputting the hinted answer) evolves
as sentences from hint-influenced CoT are transplanted into non-hint runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "per_problem_timeseries_with_sentences_all34.csv"
DELTA_FILE = BASE_DIR / "per_problem_timeseries_with_sentences_all34_delta_analysis.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    """Load both timeseries and delta datasets."""
    df = pd.read_csv(DATA_FILE)
    delta_df = pd.read_csv(DELTA_FILE)
    print(f"Loaded {len(df)} sentences from {df['pn'].nunique()} problems")
    print(f"Delta dataset has {len(delta_df)} sentences")
    return df, delta_df


def plot_delta_distribution(delta_df):
    """Plot distribution of delta_cue_p values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(delta_df['delta_cue_p'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero change')
    axes[0].set_xlabel('Delta cue_p (change from previous sentence)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Delta cue_p\n(How much does cue_p change per sentence?)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(delta_df['delta_cue_p'], vert=True)
    axes[1].set_ylabel('Delta cue_p')
    axes[1].set_title('Delta cue_p Distribution Summary')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '01_delta_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / '01_delta_distribution.png'}")
    plt.close()
    
    # Print statistics
    print("\nDelta cue_p Statistics:")
    print(f"  Mean: {delta_df['delta_cue_p'].mean():.4f}")
    print(f"  Median: {delta_df['delta_cue_p'].median():.4f}")
    print(f"  Std: {delta_df['delta_cue_p'].std():.4f}")
    print(f"  Min: {delta_df['delta_cue_p'].min():.4f}")
    print(f"  Max: {delta_df['delta_cue_p'].max():.4f}")
    print(f"  Positive deltas: {(delta_df['delta_cue_p'] > 0).sum()} ({(delta_df['delta_cue_p'] > 0).sum() / len(delta_df) * 100:.1f}%)")
    print(f"  Negative deltas: {(delta_df['delta_cue_p'] < 0).sum()} ({(delta_df['delta_cue_p'] < 0).sum() / len(delta_df) * 100:.1f}%)")


def plot_delta_by_sentence_position(delta_df):
    """Plot delta_cue_p by sentence position."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Scatter plot
    axes[0].scatter(delta_df['sentence_number'], delta_df['delta_cue_p'], 
                   alpha=0.4, s=30, c=delta_df['delta_cue_p'], 
                   cmap='RdYlGn', vmin=-0.3, vmax=0.3)
    axes[0].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Sentence Number in CoT')
    axes[0].set_ylabel('Delta cue_p')
    axes[0].set_title('Delta cue_p vs Sentence Position\n(Each point is a sentence from some problem)')
    axes[0].grid(True, alpha=0.3)
    
    # Average by sentence position
    avg_by_position = delta_df.groupby('sentence_number')['delta_cue_p'].agg(['mean', 'std', 'count']).reset_index()
    avg_by_position = avg_by_position[avg_by_position['count'] >= 3]  # At least 3 samples
    
    axes[1].plot(avg_by_position['sentence_number'], avg_by_position['mean'], 
                marker='o', linewidth=2, markersize=6, label='Mean delta_cue_p')
    axes[1].fill_between(avg_by_position['sentence_number'],
                        avg_by_position['mean'] - avg_by_position['std'],
                        avg_by_position['mean'] + avg_by_position['std'],
                        alpha=0.3, label='±1 std')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Sentence Number in CoT')
    axes[1].set_ylabel('Average Delta cue_p')
    axes[1].set_title('Average Delta cue_p by Sentence Position\n(Are early/late sentences more impactful?)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '02_delta_by_sentence_position.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / '02_delta_by_sentence_position.png'}")
    plt.close()


def plot_cue_p_trajectories(df):
    """Plot cue_p trajectories for individual problems."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    problems = sorted(df['pn'].unique())
    
    # Plot 1: All trajectories (spaghetti plot)
    ax = axes[0, 0]
    for pn in problems:
        pn_data = df[df['pn'] == pn].sort_values('sentence_num')
        ax.plot(pn_data['sentence_num'], pn_data['true_cue_p'], 
               alpha=0.3, linewidth=1)
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('True cue_p')
    ax.set_title(f'All {len(problems)} Problem Trajectories\n(How does cue_p evolve?)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Average trajectory with confidence band
    ax = axes[0, 1]
    avg_by_sentence = df.groupby('sentence_num')['true_cue_p'].agg(['mean', 'std', 'count']).reset_index()
    avg_by_sentence = avg_by_sentence[avg_by_sentence['count'] >= 3]
    
    ax.plot(avg_by_sentence['sentence_num'], avg_by_sentence['mean'], 
           linewidth=3, color='blue', label='Mean cue_p')
    ax.fill_between(avg_by_sentence['sentence_num'],
                   avg_by_sentence['mean'] - avg_by_sentence['std'],
                   avg_by_sentence['mean'] + avg_by_sentence['std'],
                   alpha=0.3, color='blue', label='±1 std')
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('True cue_p')
    ax.set_title('Average cue_p Trajectory Across All Problems')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: Heatmap of cue_p evolution
    ax = axes[1, 0]
    
    # Create a matrix: problems x sentences
    max_sentences = int(df['sentence_num'].max()) + 1
    matrix = np.full((len(problems), max_sentences), np.nan)
    
    for i, pn in enumerate(problems):
        pn_data = df[df['pn'] == pn].sort_values('sentence_num')
        for _, row in pn_data.iterrows():
            sent_idx = int(row['sentence_num'])
            if sent_idx < max_sentences:
                matrix[i, sent_idx] = row['true_cue_p']
    
    # Only show first 30 sentences for clarity
    im = ax.imshow(matrix[:, :min(30, max_sentences)], aspect='auto', cmap='RdYlGn', 
                   vmin=0, vmax=1, interpolation='none')
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('Problem Index')
    ax.set_title('Heatmap: cue_p Evolution per Problem\n(Green = high cue_p, Red = low cue_p)')
    plt.colorbar(im, ax=ax, label='cue_p')
    
    # Plot 4: Final cue_p distribution
    ax = axes[1, 1]
    final_cue_p = df.groupby('pn')['true_cue_p'].last()
    ax.hist(final_cue_p, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(final_cue_p.mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {final_cue_p.mean():.2f}')
    ax.axvline(final_cue_p.median(), color='orange', linestyle='--', 
              linewidth=2, label=f'Median: {final_cue_p.median():.2f}')
    ax.set_xlabel('Final cue_p (last sentence)')
    ax.set_ylabel('Number of Problems')
    ax.set_title('Distribution of Final cue_p Values\n(Where do problems end up?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '03_cue_p_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / '03_cue_p_trajectories.png'}")
    plt.close()


def plot_anchor_points_analysis(delta_df):
    """Analyze 'anchor points' - sentences with large delta jumps."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define thresholds
    high_threshold = 0.3
    low_threshold = -0.2
    
    high_anchors = delta_df[delta_df['delta_cue_p'] >= high_threshold].copy()
    low_anchors = delta_df[delta_df['delta_cue_p'] <= low_threshold].copy()
    
    # Plot 1: Position of high anchor points
    ax = axes[0, 0]
    ax.hist(high_anchors['sentence_number'], bins=range(0, 30), 
           edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('Count')
    ax.set_title(f'Position of Large Increases (delta_cue_p ≥ {high_threshold})\n'
                f'Total: {len(high_anchors)} sentences from {high_anchors["problem_number"].nunique()} problems')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Position of low anchor points (large decreases)
    ax = axes[0, 1]
    ax.hist(low_anchors['sentence_number'], bins=range(0, 30), 
           edgecolor='black', alpha=0.7, color='red')
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('Count')
    ax.set_title(f'Position of Large Decreases (delta_cue_p ≤ {low_threshold})\n'
                f'Total: {len(low_anchors)} sentences from {low_anchors["problem_number"].nunique()} problems')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Delta magnitude by sentence position
    ax = axes[1, 0]
    delta_df['abs_delta'] = delta_df['delta_cue_p'].abs()
    avg_abs_delta = delta_df.groupby('sentence_number')['abs_delta'].agg(['mean', 'count']).reset_index()
    avg_abs_delta = avg_abs_delta[avg_abs_delta['count'] >= 3]
    
    ax.bar(avg_abs_delta['sentence_number'], avg_abs_delta['mean'], 
          alpha=0.7, edgecolor='black')
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('Average |delta_cue_p|')
    ax.set_title('Average Absolute Change by Sentence Position\n(Which positions see most change?)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative effect
    ax = axes[1, 1]
    # For each problem, compute cumulative delta
    problems = sorted(delta_df['problem_number'].unique())
    for pn in problems[:20]:  # Show first 20 for clarity
        pn_data = delta_df[delta_df['problem_number'] == pn].sort_values('sentence_number')
        cumulative = pn_data['delta_cue_p'].cumsum()
        ax.plot(pn_data['sentence_number'], cumulative, alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Sentence Number')
    ax.set_ylabel('Cumulative Delta cue_p')
    ax.set_title('Cumulative Delta cue_p Over Time\n(First 20 problems shown)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '04_anchor_points_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / '04_anchor_points_analysis.png'}")
    plt.close()
    
    print(f"\nAnchor Point Statistics:")
    print(f"  High anchors (Δ ≥ {high_threshold}): {len(high_anchors)}")
    print(f"  Low anchors (Δ ≤ {low_threshold}): {len(low_anchors)}")


def plot_convergence_analysis(df):
    """Analyze when problems 'converge' to high or low cue_p."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define convergence as reaching cue_p > 0.8 or < 0.2
    convergence_data = []
    
    for pn in sorted(df['pn'].unique()):
        pn_data = df[df['pn'] == pn].sort_values('sentence_num')
        
        # Find first sentence where cue_p > 0.8
        high_conv = pn_data[pn_data['true_cue_p'] > 0.8]
        high_sent = high_conv.iloc[0]['sentence_num'] if len(high_conv) > 0 else None
        
        # Find first sentence where cue_p < 0.2  
        low_conv = pn_data[pn_data['true_cue_p'] < 0.2]
        low_sent = low_conv.iloc[0]['sentence_num'] if len(low_conv) > 0 else None
        
        final_cue_p = pn_data.iloc[-1]['true_cue_p']
        total_sentences = len(pn_data)
        
        convergence_data.append({
            'problem': pn,
            'converge_high_sent': high_sent,
            'converge_low_sent': low_sent,
            'final_cue_p': final_cue_p,
            'total_sentences': total_sentences
        })
    
    conv_df = pd.DataFrame(convergence_data)
    
    # Plot 1: When do problems converge to high cue_p?
    ax = axes[0]
    high_convergers = conv_df[conv_df['converge_high_sent'].notna()]
    ax.hist(high_convergers['converge_high_sent'], bins=range(0, 25), 
           edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Sentence Number at First cue_p > 0.8')
    ax.set_ylabel('Number of Problems')
    ax.set_title(f'When Do Problems Lock Into High cue_p?\n'
                f'{len(high_convergers)}/{len(conv_df)} problems reach cue_p > 0.8')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Initial vs final cue_p
    ax = axes[1]
    # Get initial cue_p for each problem
    initial_cue_p = []
    for pn in sorted(df['pn'].unique()):
        pn_data = df[df['pn'] == pn].sort_values('sentence_num')
        initial_cue_p.append(pn_data.iloc[0]['true_cue_p'])
    
    conv_df['initial_cue_p'] = initial_cue_p
    
    ax.scatter(conv_df['initial_cue_p'], conv_df['final_cue_p'], 
              s=80, alpha=0.6, edgecolors='black', linewidth=1)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No change')
    ax.set_xlabel('Initial cue_p (sentence 0)')
    ax.set_ylabel('Final cue_p (last sentence)')
    ax.set_title('Initial vs Final cue_p\n(Do problems with high initial cue_p stay high?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '05_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / '05_convergence_analysis.png'}")
    plt.close()


def analyze_top_delta_sentences(delta_df):
    """Detailed look at sentences with highest deltas."""
    print("\n" + "="*80)
    print("TOP 20 SENTENCES WITH HIGHEST DELTA CUE_P:")
    print("="*80)
    
    top_20 = delta_df.nlargest(20, 'delta_cue_p')
    
    for idx, (i, row) in enumerate(top_20.iterrows(), 1):
        print(f"\n{idx}. Problem {row['problem_number']}, Sentence {row['sentence_number']}")
        print(f"   Delta: {row['delta_cue_p']:.3f} | cue_p: {row['true_cue_p']:.3f}")
        print(f"   \"{row['sentence'][:120]}{'...' if len(row['sentence']) > 120 else ''}\"")
    
    print("\n" + "="*80)
    print("TOP 20 SENTENCES WITH LARGEST DECREASES:")
    print("="*80)
    
    bottom_20 = delta_df.nsmallest(20, 'delta_cue_p')
    
    for idx, (i, row) in enumerate(bottom_20.iterrows(), 1):
        print(f"\n{idx}. Problem {row['problem_number']}, Sentence {row['sentence_number']}")
        print(f"   Delta: {row['delta_cue_p']:.3f} | cue_p: {row['true_cue_p']:.3f}")
        print(f"   \"{row['sentence'][:120]}{'...' if len(row['sentence']) > 120 else ''}\"")


def main():
    """Run all analyses."""
    print("="*80)
    print("CUE_P DYNAMICS ANALYSIS")
    print("="*80)
    print(f"\nData file: {DATA_FILE}")
    print(f"Output directory: {FIG_DIR}")
    
    # Load data
    df, delta_df = load_data()
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    print("\n1. Delta distribution...")
    plot_delta_distribution(delta_df)
    
    print("\n2. Delta by sentence position...")
    plot_delta_by_sentence_position(delta_df)
    
    print("\n3. cue_p trajectories...")
    plot_cue_p_trajectories(df)
    
    print("\n4. Anchor points analysis...")
    plot_anchor_points_analysis(delta_df)
    
    print("\n5. Convergence analysis...")
    plot_convergence_analysis(df)
    
    # Text analysis
    analyze_top_delta_sentences(delta_df)
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE! All plots saved to {FIG_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

