"""
Find transplant curves that show smooth, monotonic increases in hint effect.

These are the clearest examples of unfaithful reasoning where the hint's
influence accumulates gradually through the CoT.

Usage:
    python find_smooth_curves.py --input data/TIMESTAMP/transplant_results.json
"""

import json
import argparse
from pathlib import Path
import numpy as np


def calculate_smoothness(positions, hint_pcts):
    """
    Calculate how smooth/monotonic a curve is.

    Returns a smoothness score (higher = smoother).
    - Penalizes large drops
    - Rewards monotonic increases
    """
    if len(positions) < 3:
        return 0.0

    # Calculate differences between consecutive points
    diffs = np.diff(hint_pcts)

    # Count monotonic increases (positive diffs)
    increases = np.sum(diffs > 0)
    total_steps = len(diffs)

    # Calculate average slope when increasing
    positive_diffs = diffs[diffs > 0]
    avg_slope = np.mean(positive_diffs) if len(positive_diffs) > 0 else 0

    # Penalize large drops
    negative_diffs = diffs[diffs < 0]
    max_drop = abs(np.min(negative_diffs)) if len(negative_diffs) > 0 else 0

    # Final effect should be positive
    final_effect = hint_pcts[-1] - hint_pcts[0]

    # Smoothness score:
    # - Reward high proportion of increases
    # - Reward positive final effect
    # - Penalize large drops
    smoothness = (increases / total_steps) * final_effect - (max_drop * 0.5)

    return smoothness


def analyze_curves(results):
    """
    Analyze all curves and rank by smoothness.
    """
    curve_info = []

    for question in results:
        q_id = question['question_id']
        subject = question['subject']

        for sample_result in question['transplant_results']:
            curve = sample_result['curve']
            sample_num = sample_result['sample_num']

            positions = [c['num_sentences'] for c in curve]
            hint_pcts = [c['hinted_pct'] * 100 for c in curve]

            # Calculate metrics
            smoothness = calculate_smoothness(positions, hint_pcts)
            final_effect = hint_pcts[-1] - hint_pcts[0]
            max_value = max(hint_pcts)

            # Check if generally increasing
            diffs = np.diff(hint_pcts)
            pct_increasing = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else 0

            curve_info.append({
                'question_id': q_id,
                'subject': subject,
                'sample_num': sample_num,
                'smoothness': smoothness,
                'final_effect': final_effect,
                'max_value': max_value,
                'pct_increasing': pct_increasing,
                'num_sentences': len(positions) - 1,  # Exclude position 0
                'positions': positions,
                'hint_pcts': hint_pcts,
                'original_answer': sample_result['original_answer']
            })

    # Sort by smoothness
    curve_info.sort(key=lambda x: x['smoothness'], reverse=True)

    return curve_info


def main():
    parser = argparse.ArgumentParser(description='Find smooth transplant curves')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to transplant_results.json')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top curves to show (default: 10)')
    parser.add_argument('--min-effect', type=float, default=20,
                       help='Minimum final effect % (default: 20)')
    parser.add_argument('--min-increasing', type=float, default=0.5,
                       help='Minimum % of steps that increase (default: 0.5)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent

    print(f"Loading transplant results from {input_path}")
    with open(input_path, 'r') as f:
        results = json.load(f)

    print(f"Analyzing {sum(len(q['transplant_results']) for q in results)} curves...\n")

    curve_info = analyze_curves(results)

    # Filter by criteria
    filtered = [
        c for c in curve_info
        if c['final_effect'] >= args.min_effect
        and c['pct_increasing'] >= args.min_increasing
    ]

    print(f"Found {len(filtered)} curves meeting criteria:")
    print(f"  - Final effect >= {args.min_effect}%")
    print(f"  - {args.min_increasing*100:.0f}% of steps increasing\n")

    print("=" * 80)
    print(f"TOP {args.top} SMOOTHEST CURVES:")
    print("=" * 80)

    for i, curve in enumerate(filtered[:args.top], 1):
        print(f"\n{i}. Question {curve['question_id']} ({curve['subject']}), Sample {curve['sample_num']}")
        print(f"   Smoothness score: {curve['smoothness']:.1f}")
        print(f"   Final effect: {curve['final_effect']:.1f}% (0 → {curve['hint_pcts'][-1]:.0f}%)")
        print(f"   Max value: {curve['max_value']:.0f}%")
        print(f"   % steps increasing: {curve['pct_increasing']*100:.0f}%")
        print(f"   CoT length: {curve['num_sentences']} sentences")
        print(f"   Original answer: {curve['original_answer']}")

        # Show curve progression
        print(f"   Curve: ", end="")
        for pos, pct in zip(curve['positions'][:10], curve['hint_pcts'][:10]):
            print(f"{pos}:{pct:.0f}% ", end="")
        if len(curve['positions']) > 10:
            print("...")
        else:
            print()

    # Save top curves
    output_file = output_dir / "smooth_curves.json"
    with open(output_file, 'w') as f:
        json.dump(filtered[:args.top], f, indent=2)

    print(f"\n✓ Saved top {args.top} curves to {output_file}")


if __name__ == "__main__":
    main()
