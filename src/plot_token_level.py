"""
Plot token-by-token transplant resampling results.

Usage:
    python plot_token_level.py --input data/TIMESTAMP/token_level_qX_sY.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def plot_token_level_curve(result, output_path):
    """
    Plot cumulative hint effect token-by-token.
    """
    curve = result['curve']

    positions = [c['num_tokens'] for c in curve]
    hint_pcts = [c['hinted_pct'] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot curve
    ax.plot(positions, hint_pcts, 'o-', linewidth=2, markersize=6, color='steelblue')

    # Highlight key regions
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    ax.set_xlabel('Number of CoT tokens included', fontsize=12)
    ax.set_ylabel('% giving hinted answer', fontsize=12)
    ax.set_title(f"Token-Level Transplant Resampling\n"
                f"Q{result['question_idx']}, Sample {result['sample_idx']} ({result['subject']})\n"
                f"Correct={result['correct_answer']}, Hinted={result['hinted_answer']}",
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax.set_ylim(-5, 105)
    ax.set_xlim(-5, max(positions) + 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot token-level transplant results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to token_level_qX_sY.json')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    print(f"Loading from {input_path}")
    with open(input_path, 'r') as f:
        result = json.load(f)

    print(f"Question {result['question_idx']}, Sample {result['sample_idx']}")
    print(f"CoT has {result['num_tokens']} tokens")
    print(f"Sampled every {result['stride']} tokens")
    print(f"{result['num_samples_per_position']} samples per position\n")

    # Create plot
    output_path = output_dir / f"token_level_plot_q{result['question_idx']}_s{result['sample_idx']}.png"
    plot_token_level_curve(result, output_path)

    # Print summary statistics
    curve = result['curve']
    baseline = curve[0]['hinted_pct'] * 100
    final = curve[-1]['hinted_pct'] * 100
    max_pct = max(c['hinted_pct'] * 100 for c in curve)

    print(f"\nSummary:")
    print(f"  Baseline (0 tokens): {baseline:.1f}%")
    print(f"  Final ({result['num_tokens']} tokens): {final:.1f}%")
    print(f"  Maximum: {max_pct:.1f}%")
    print(f"  Total change: {final - baseline:+.1f}%")


if __name__ == "__main__":
    main()
