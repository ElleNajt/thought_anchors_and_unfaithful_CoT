"""
Visualize bias mechanism classifications from high diff sentences.

This script analyzes the classification results to understand:
- Distribution of bias mechanisms
- Confidence patterns
- Tier 1 vs Tier 2 prevalence
- Context dependency
- Relationship with diff magnitude
- Co-occurrence patterns

Usage:
    python plot_classification_analysis.py --json-path sentence_classifications/high_diff_sentence_classifications.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_classifications(json_path):
    """Load classification data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out failed classifications
    valid_data = [item for item in data if item.get('classification') is not None]
    print(f"Loaded {len(valid_data)} valid classifications ({len(data) - len(valid_data)} failed)")
    
    return valid_data


def extract_statistics(data):
    """Extract key statistics from classification data."""
    stats = {
        'primary_mechanisms': [],
        'all_mechanisms': [],
        'primary_tiers': [],
        'primary_confidences': [],
        'context_dependencies': [],
        'diffs': [],
        'mechanism_counts_per_sentence': [],
        'tier_by_diff': [],
        'confidence_by_tier': defaultdict(list),
        'mechanism_by_diff': defaultdict(list),
        'mechanism_confidences': defaultdict(list),
        'mechanism_cooccurrence': defaultdict(Counter),
    }
    
    for item in data:
        classification = item['classification']
        classifications = classification.get('classifications', [])
        
        if not classifications:
            continue
        
        diff = item['diff']
        context_dep = classification.get('context_dependency', 'Unknown')
        
        # Primary mechanism (first in list)
        primary = classifications[0]
        primary_mech = primary['mechanism']
        primary_tier = primary['tier']
        primary_conf = primary['confidence']
        
        stats['primary_mechanisms'].append(primary_mech)
        stats['primary_tiers'].append(primary_tier)
        stats['primary_confidences'].append(primary_conf)
        stats['context_dependencies'].append(context_dep)
        stats['diffs'].append(diff)
        stats['mechanism_counts_per_sentence'].append(len(classifications))
        stats['tier_by_diff'].append((diff, primary_tier))
        stats['confidence_by_tier'][primary_tier].append(primary_conf)
        stats['mechanism_by_diff'][primary_mech].append(diff)
        stats['mechanism_confidences'][primary_mech].append(primary_conf)
        
        # All mechanisms
        for c in classifications:
            stats['all_mechanisms'].append(c['mechanism'])
        
        # Co-occurrence (what mechanisms appear together with primary)
        if len(classifications) > 1:
            for secondary in classifications[1:]:
                stats['mechanism_cooccurrence'][primary_mech][secondary['mechanism']] += 1
    
    return stats


def plot_mechanism_distribution(stats, output_dir):
    """Plot distribution of primary mechanisms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count primary mechanisms
    mechanism_counts = Counter(stats['primary_mechanisms'])
    mechanisms = sorted(mechanism_counts.keys(), key=lambda x: mechanism_counts[x], reverse=True)
    counts = [mechanism_counts[m] for m in mechanisms]
    
    # Define tier colors
    tier1_mechanisms = [
        'False Framing', 'False Categorization', 'Definitional Stretch',
        'Separation Fallacy', 'Misreading', 'Vague Principle Invocation',
        'False Memory/Confabulation', 'Recognition Signal'
    ]
    colors = ['#e74c3c' if m in tier1_mechanisms else '#3498db' for m in mechanisms]
    
    # Plot 1: Horizontal bar chart
    y_pos = np.arange(len(mechanisms))
    ax1.barh(y_pos, counts, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(mechanisms, fontsize=9)
    ax1.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Primary Mechanism Distribution', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add count labels
    for i, (count, color) in enumerate(zip(counts, colors)):
        ax1.text(count + 0.3, i, str(count), va='center', fontweight='bold')
    
    # Add legend
    tier1_patch = Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.8)
    tier2_patch = Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.8)
    ax1.legend([tier1_patch, tier2_patch], ['Tier 1: Context-Independent', 'Tier 2: Context-Dependent'],
               loc='lower right', fontsize=9)
    
    # Plot 2: Tier distribution pie chart
    tier_counts = Counter(stats['primary_tiers'])
    tier_labels = [f"Tier {t}\n(n={tier_counts[t]})" for t in sorted(tier_counts.keys())]
    tier_values = [tier_counts[t] for t in sorted(tier_counts.keys())]
    
    ax2.pie(tier_values, labels=tier_labels, autopct='%1.1f%%',
            colors=['#e74c3c', '#3498db'], startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Tier Distribution (Primary Mechanism)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mechanism_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved mechanism_distribution.png")
    plt.close()


def plot_confidence_analysis(stats, output_dir):
    """Plot confidence level analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall confidence distribution
    conf_counts = Counter(stats['primary_confidences'])
    conf_order = ['High', 'Medium', 'Low']
    conf_counts_ordered = [conf_counts.get(c, 0) for c in conf_order]
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    
    ax1.bar(conf_order, conf_counts_ordered, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Confidence Distribution (Primary Mechanism)', fontsize=13, fontweight='bold')
    for i, count in enumerate(conf_counts_ordered):
        ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold', fontsize=11)
    
    # Plot 2: Confidence by tier
    tier_conf_data = []
    tier_labels = []
    for tier in sorted(stats['confidence_by_tier'].keys()):
        confs = stats['confidence_by_tier'][tier]
        tier_conf_data.append([confs.count('High'), confs.count('Medium'), confs.count('Low')])
        tier_labels.append(f'Tier {tier}')
    
    tier_conf_data = np.array(tier_conf_data).T
    x = np.arange(len(tier_labels))
    width = 0.25
    
    for i, (conf, color) in enumerate(zip(['High', 'Medium', 'Low'], colors)):
        ax2.bar(x + i*width, tier_conf_data[i], width, label=conf, color=color, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Tier', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Distribution by Tier', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(tier_labels)
    ax2.legend(title='Confidence', fontsize=10)
    
    # Plot 3: Top mechanisms by confidence
    mechanism_high_conf = {}
    for mech in set(stats['primary_mechanisms']):
        confs = stats['mechanism_confidences'][mech]
        if confs:
            high_pct = confs.count('High') / len(confs) * 100
            mechanism_high_conf[mech] = (high_pct, len(confs))
    
    # Filter mechanisms with at least 3 instances
    filtered_mechs = {k: v for k, v in mechanism_high_conf.items() if v[1] >= 3}
    top_mechs = sorted(filtered_mechs.items(), key=lambda x: x[1][0], reverse=True)[:10]
    
    if top_mechs:
        mech_names = [m[0] for m in top_mechs]
        high_pcts = [m[1][0] for m in top_mechs]
        
        y_pos = np.arange(len(mech_names))
        bars = ax3.barh(y_pos, high_pcts, color='#27ae60', alpha=0.8, edgecolor='black')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(mech_names, fontsize=9)
        ax3.set_xlabel('% High Confidence', fontsize=11, fontweight='bold')
        ax3.set_title('Top Mechanisms by High Confidence %\n(≥3 instances)', fontsize=13, fontweight='bold')
        ax3.invert_yaxis()
        
        # Add percentage labels
        for i, pct in enumerate(high_pcts):
            ax3.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold')
    
    # Plot 4: Mechanisms with multiple classifications per sentence
    count_dist = Counter(stats['mechanism_counts_per_sentence'])
    counts = sorted(count_dist.keys())
    frequencies = [count_dist[c] for c in counts]
    
    ax4.bar(counts, frequencies, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Number of Mechanisms per Sentence', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Multiple Mechanism Classifications', fontsize=13, fontweight='bold')
    ax4.set_xticks(counts)
    
    for i, (count, freq) in enumerate(zip(counts, frequencies)):
        ax4.text(count, freq + 0.5, str(freq), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confidence_analysis.png")
    plt.close()


def plot_diff_analysis(stats, output_dir):
    """Plot relationship between diff magnitude and mechanisms."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Diff distribution histogram
    ax1.hist(stats['diffs'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(stats['diffs']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(stats['diffs']):.3f}")
    ax1.axvline(np.median(stats['diffs']), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(stats['diffs']):.3f}")
    ax1.set_xlabel('Cue_p Diff', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Diff Values', fontsize=13, fontweight='bold')
    ax1.legend()
    
    # Plot 2: Diff by tier
    tier_diffs = defaultdict(list)
    for diff, tier in stats['tier_by_diff']:
        tier_diffs[tier].append(diff)
    
    tier_data = [tier_diffs[t] for t in sorted(tier_diffs.keys())]
    tier_labels = [f'Tier {t}' for t in sorted(tier_diffs.keys())]
    
    bp = ax2.boxplot(tier_data, tick_labels=tier_labels, patch_artist=True,
                      boxprops=dict(facecolor='#3498db', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Cue_p Diff', fontsize=11, fontweight='bold')
    ax2.set_title('Diff Distribution by Tier', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Top mechanisms by avg diff
    mech_avg_diff = {}
    for mech in set(stats['primary_mechanisms']):
        diffs = stats['mechanism_by_diff'][mech]
        if len(diffs) >= 3:  # At least 3 instances
            mech_avg_diff[mech] = np.mean(diffs)
    
    top_diff_mechs = sorted(mech_avg_diff.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_diff_mechs:
        mech_names = [m[0] for m in top_diff_mechs]
        avg_diffs = [m[1] for m in top_diff_mechs]
        
        y_pos = np.arange(len(mech_names))
        bars = ax3.barh(y_pos, avg_diffs, color='#e67e22', alpha=0.8, edgecolor='black')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(mech_names, fontsize=9)
        ax3.set_xlabel('Average Cue_p Diff', fontsize=11, fontweight='bold')
        ax3.set_title('Top Mechanisms by Average Diff\n(≥3 instances)', fontsize=13, fontweight='bold')
        ax3.invert_yaxis()
        
        for i, diff in enumerate(avg_diffs):
            ax3.text(diff + 0.01, i, f'{diff:.3f}', va='center', fontweight='bold')
    
    # Plot 4: Context dependency
    context_counts = Counter(stats['context_dependencies'])
    labels = list(context_counts.keys())
    values = list(context_counts.values())
    colors_ctx = ['#e74c3c' if label == 'No' else '#3498db' for label in labels]
    
    wedges, texts, autotexts = ax4.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors_ctx, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax4.set_title(f'Context Dependency\n(n={sum(values)})', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diff_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved diff_analysis.png")
    plt.close()


def plot_cooccurrence_heatmap(stats, output_dir):
    """Plot co-occurrence heatmap for mechanisms."""
    # Get top 10 most common primary mechanisms
    primary_counts = Counter(stats['primary_mechanisms'])
    top_primary = [m for m, _ in primary_counts.most_common(10)]
    
    # Get all mechanisms that co-occur with these
    all_secondary = set()
    for primary in top_primary:
        all_secondary.update(stats['mechanism_cooccurrence'][primary].keys())
    
    if not all_secondary:
        print("⚠ No co-occurrence data to plot")
        return
    
    # Build co-occurrence matrix
    secondary_list = sorted(all_secondary)
    matrix = np.zeros((len(top_primary), len(secondary_list)))
    
    for i, primary in enumerate(top_primary):
        for j, secondary in enumerate(secondary_list):
            matrix[i, j] = stats['mechanism_cooccurrence'][primary].get(secondary, 0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(secondary_list)))
    ax.set_yticks(np.arange(len(top_primary)))
    ax.set_xticklabels(secondary_list, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_primary, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Co-occurrence Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(top_primary)):
        for j in range(len(secondary_list)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('Secondary Mechanism', fontsize=11, fontweight='bold')
    ax.set_ylabel('Primary Mechanism', fontsize=11, fontweight='bold')
    ax.set_title('Mechanism Co-occurrence Patterns\n(Top 10 Primary Mechanisms)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mechanism_cooccurrence.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved mechanism_cooccurrence.png")
    plt.close()


def plot_detailed_mechanism_profiles(stats, output_dir):
    """Plot detailed profiles for top mechanisms."""
    # Get top 8 most common mechanisms
    mechanism_counts = Counter(stats['primary_mechanisms'])
    top_mechanisms = [m for m, _ in mechanism_counts.most_common(8)]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, mechanism in enumerate(top_mechanisms):
        ax = axes[idx]
        
        # Get data for this mechanism
        diffs = stats['mechanism_by_diff'][mechanism]
        confs = stats['mechanism_confidences'][mechanism]
        
        # Create scatter plot of diff vs confidence
        conf_to_y = {'High': 3, 'Medium': 2, 'Low': 1}
        y_values = [conf_to_y[c] + np.random.uniform(-0.15, 0.15) for c in confs]
        
        colors_map = {'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        colors = [colors_map[c] for c in confs]
        
        ax.scatter(diffs, y_values, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        # Add mean line
        ax.axvline(np.mean(diffs), color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.set_xlabel('Cue_p Diff', fontsize=10, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=10, fontweight='bold')
        
        # Title with stats
        n = len(diffs)
        mean_diff = np.mean(diffs)
        high_pct = confs.count('High') / n * 100
        ax.set_title(f'{mechanism}\n(n={n}, μ={mean_diff:.3f}, {high_pct:.0f}% high conf)',
                    fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mechanism_profiles.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved mechanism_profiles.png")
    plt.close()


def print_summary_statistics(stats):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal classified sentences: {len(stats['primary_mechanisms'])}")
    
    # Tier distribution
    tier_counts = Counter(stats['primary_tiers'])
    print(f"\nTier Distribution:")
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        pct = count / len(stats['primary_tiers']) * 100
        print(f"  Tier {tier}: {count} ({pct:.1f}%)")
    
    # Confidence distribution
    conf_counts = Counter(stats['primary_confidences'])
    print(f"\nConfidence Distribution:")
    for conf in ['High', 'Medium', 'Low']:
        count = conf_counts.get(conf, 0)
        pct = count / len(stats['primary_confidences']) * 100 if stats['primary_confidences'] else 0
        print(f"  {conf}: {count} ({pct:.1f}%)")
    
    # Context dependency
    context_counts = Counter(stats['context_dependencies'])
    print(f"\nContext Dependency:")
    for ctx in sorted(context_counts.keys()):
        count = context_counts[ctx]
        pct = count / len(stats['context_dependencies']) * 100
        print(f"  {ctx}: {count} ({pct:.1f}%)")
    
    # Top mechanisms
    mechanism_counts = Counter(stats['primary_mechanisms'])
    print(f"\nTop 10 Primary Mechanisms:")
    for i, (mech, count) in enumerate(mechanism_counts.most_common(10), 1):
        pct = count / len(stats['primary_mechanisms']) * 100
        print(f"  {i}. {mech}: {count} ({pct:.1f}%)")
    
    # Diff statistics
    print(f"\nDiff Statistics:")
    print(f"  Mean: {np.mean(stats['diffs']):.4f}")
    print(f"  Median: {np.median(stats['diffs']):.4f}")
    print(f"  Std: {np.std(stats['diffs']):.4f}")
    print(f"  Min: {np.min(stats['diffs']):.4f}")
    print(f"  Max: {np.max(stats['diffs']):.4f}")
    
    # Multiple classifications
    multi_class = [c for c in stats['mechanism_counts_per_sentence'] if c > 1]
    if multi_class:
        pct = len(multi_class) / len(stats['mechanism_counts_per_sentence']) * 100
        print(f"\nMultiple Mechanism Classifications:")
        print(f"  Sentences with >1 mechanism: {len(multi_class)} ({pct:.1f}%)")
        print(f"  Average mechanisms per sentence: {np.mean(stats['mechanism_counts_per_sentence']):.2f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize bias mechanism classifications')
    parser.add_argument('--json-path', type=str, required=True,
                       help='Path to classification JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as JSON)')
    args = parser.parse_args()
    
    # Setup paths
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent / 'classification_plots'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading classifications from: {json_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Load and analyze data
    data = load_classifications(json_path)
    stats = extract_statistics(data)
    
    # Print summary statistics
    print_summary_statistics(stats)
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_mechanism_distribution(stats, output_dir)
    plot_confidence_analysis(stats, output_dir)
    plot_diff_analysis(stats, output_dir)
    plot_cooccurrence_heatmap(stats, output_dir)
    plot_detailed_mechanism_profiles(stats, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - mechanism_distribution.png")
    print(f"  - confidence_analysis.png")
    print(f"  - diff_analysis.png")
    print(f"  - mechanism_cooccurrence.png")
    print(f"  - mechanism_profiles.png")


if __name__ == "__main__":
    main()

