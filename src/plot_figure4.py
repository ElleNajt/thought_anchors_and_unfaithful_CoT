"""
Create Figure 4: Transplant resampling visualization.

Shows cumulative hint effect as sentences are added to the CoT.

Usage:
    python plot_figure4.py --input data/TIMESTAMP/transplant_results.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_transplant_curves(results, output_path):
    """
    Plot cumulative hint effect curves for all samples.

    Creates a line plot showing how hint effect accumulates
    sentence-by-sentence through the CoT.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    all_curves = []

    # Collect all curves from all questions
    for question in results:
        for sample_result in question['transplant_results']:
            curve = sample_result['curve']

            # Extract positions and hint percentages
            positions = [c['num_sentences'] for c in curve]
            hint_pcts = [c['hinted_pct'] * 100 for c in curve]  # Convert to percentage

            all_curves.append((positions, hint_pcts))

            # Plot individual curve (semi-transparent)
            ax.plot(positions, hint_pcts, 'o-', alpha=0.3, color='steelblue', linewidth=1)

    # Plot average curve
    if all_curves:
        # Find max position to normalize
        max_pos = max(max(pos) for pos, _ in all_curves)

        # Interpolate all curves to same positions
        common_positions = range(max_pos + 1)
        interpolated = []

        for positions, hint_pcts in all_curves:
            # Interpolate to common positions
            interp_pcts = np.interp(common_positions, positions, hint_pcts)
            interpolated.append(interp_pcts)

        # Calculate mean and std
        mean_pcts = np.mean(interpolated, axis=0)
        std_pcts = np.std(interpolated, axis=0)

        # Plot mean curve
        ax.plot(common_positions, mean_pcts, 'o-', color='darkred', linewidth=3,
                label='Average', markersize=8, zorder=10)

        # Add confidence interval
        ax.fill_between(common_positions,
                        mean_pcts - std_pcts,
                        mean_pcts + std_pcts,
                        alpha=0.2, color='darkred')

    ax.set_xlabel('Number of CoT sentences included', fontsize=12)
    ax.set_ylabel('% giving hinted answer', fontsize=12)
    ax.set_title('Transplant Resampling: Cumulative Hint Effect', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Set y-axis limits
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    return fig


def plot_question_breakdown(results, output_dir):
    """
    Create separate plots for each question.
    """
    for q_idx, question in enumerate(results):
        fig, ax = plt.subplots(figsize=(10, 6))

        for sample_result in question['transplant_results']:
            curve = sample_result['curve']
            sample_num = sample_result['sample_num']

            positions = [c['num_sentences'] for c in curve]
            hint_pcts = [c['hinted_pct'] * 100 for c in curve]

            ax.plot(positions, hint_pcts, 'o-', alpha=0.7, linewidth=2,
                   label=f"Sample {sample_num}")

        ax.set_xlabel('Number of CoT sentences included', fontsize=12)
        ax.set_ylabel('% giving hinted answer', fontsize=12)
        ax.set_title(f"Q{q_idx+1}: {question['subject']}\n"
                    f"Correct={question['correct_answer']}, Hinted={question['hinted_answer']}",
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(-5, 105)

        plt.tight_layout()
        output_path = output_dir / f"question_{q_idx+1}_transplant.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 4: Transplant resampling results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to transplant_results.json')
    parser.add_argument('--breakdown', action='store_true',
                       help='Also create per-question breakdown plots')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    print(f"Loading transplant results from {input_path}")
    with open(input_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} questions")
    total_samples = sum(len(q['transplant_results']) for q in results)
    print(f"Total unfaithful samples: {total_samples}\n")

    # Create main figure
    print("Creating Figure 4 (combined)...")
    output_path = output_dir / "figure4_transplant_resampling.png"
    plot_transplant_curves(results, output_path)

    # Create per-question breakdown
    if args.breakdown:
        print("\nCreating per-question breakdowns...")
        plot_question_breakdown(results, output_dir)

    print(f"\n✓ Done! Saved to {output_dir}")


if __name__ == "__main__":
    main()
