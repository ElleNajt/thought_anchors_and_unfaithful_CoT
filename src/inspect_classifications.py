"""
Inspect classification results by mechanism.

Usage:
    python inspect_classifications.py --json-path sentence_classifications/high_diff_sentence_classifications.json --mechanism "False Framing"
    python inspect_classifications.py --json-path sentence_classifications/high_diff_sentence_classifications.json --list-mechanisms
    python inspect_classifications.py --json-path sentence_classifications/high_diff_sentence_classifications.json --mechanism "False Framing" --output classifications.md
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_classifications(json_path: str) -> List[Dict]:
    """Load classification data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [item for item in data if item.get('classification') is not None]


def list_mechanisms(data: List[Dict]) -> None:
    """List all unique mechanisms with counts."""
    mechanism_counts = {}
    
    for item in data:
        classifications = item['classification'].get('classifications', [])
        if classifications:
            primary = classifications[0]['mechanism']
            mechanism_counts[primary] = mechanism_counts.get(primary, 0) + 1
    
    print("\n" + "="*70)
    print("AVAILABLE MECHANISMS")
    print("="*70)
    for mechanism, count in sorted(mechanism_counts.items(), key=lambda x: -x[1]):
        print(f"  • {mechanism}: {count} cases")
    print("="*70)


def filter_by_mechanism(data: List[Dict], mechanism: str, position: str = "primary") -> List[Dict]:
    """
    Filter cases by mechanism.
    
    Args:
        data: List of classification items
        mechanism: Mechanism name to filter by
        position: "primary" (first mechanism) or "any" (anywhere in list)
    """
    filtered = []
    
    for item in data:
        classifications = item['classification'].get('classifications', [])
        
        if position == "primary":
            if classifications and classifications[0]['mechanism'] == mechanism:
                filtered.append(item)
        else:  # any
            if any(c['mechanism'] == mechanism for c in classifications):
                filtered.append(item)
    
    return filtered


def format_case_terminal(item: Dict, index: int, total: int) -> str:
    """Format a single case for terminal display."""
    classification = item['classification']
    classifications = classification.get('classifications', [])
    
    output = []
    output.append("\n" + "="*80)
    output.append(f"CASE {index}/{total} - Problem {item['problem_id']}")
    output.append("="*80)
    
    output.append(f"\n📊 DIFF: {item['diff']:.4f}")
    output.append(f"🎯 Correct Answer: {item['gt_answer']}")
    output.append(f"🔴 Biased Hint: {item['cue_answer']}")
    
    output.append(f"\n💭 PIVOTAL SENTENCE:")
    output.append(f"   \"{item['high_diff_sentence']}\"")
    
    output.append(f"\n🏷️  PRIMARY CLASSIFICATION:")
    if classifications:
        primary = classifications[0]
        output.append(f"   Mechanism: {primary['mechanism']}")
        output.append(f"   Tier: {primary['tier']}")
        output.append(f"   Confidence: {primary['confidence']}")
        output.append(f"   Evidence: {primary['evidence']}")
    
    if len(classifications) > 1:
        output.append(f"\n📋 SECONDARY MECHANISMS:")
        for i, c in enumerate(classifications[1:], 1):
            output.append(f"   {i}. {c['mechanism']} (Tier {c['tier']}, {c['confidence']} conf)")
    
    output.append(f"\n💡 REASONING:")
    output.append(f"   {classification.get('reasoning', 'N/A')}")
    
    output.append(f"\n🔗 CONTEXT DEPENDENCY: {classification.get('context_dependency', 'Unknown')}")
    
    return "\n".join(output)


def format_case_markdown(item: Dict, index: int) -> str:
    """Format a single case for markdown export."""
    classification = item['classification']
    classifications = classification.get('classifications', [])
    
    output = []
    output.append(f"\n---\n")
    output.append(f"## Case {index}: Problem {item['problem_id']}\n")
    
    output.append(f"**Diff:** {item['diff']:.4f} | ")
    output.append(f"**Correct Answer:** {item['gt_answer']} | ")
    output.append(f"**Biased Hint:** {item['cue_answer']}\n")
    
    output.append(f"### Pivotal Sentence\n")
    output.append(f"> {item['high_diff_sentence']}\n")
    
    output.append(f"### Primary Classification\n")
    if classifications:
        primary = classifications[0]
        output.append(f"- **Mechanism:** {primary['mechanism']}")
        output.append(f"- **Tier:** {primary['tier']}")
        output.append(f"- **Confidence:** {primary['confidence']}")
        output.append(f"- **Evidence:** {primary['evidence']}\n")
    
    if len(classifications) > 1:
        output.append(f"### Secondary Mechanisms\n")
        for i, c in enumerate(classifications[1:], 1):
            output.append(f"{i}. **{c['mechanism']}** (Tier {c['tier']}, {c['confidence']} confidence)")
            output.append(f"   - {c['evidence']}\n")
    
    output.append(f"### Reasoning\n")
    output.append(f"{classification.get('reasoning', 'N/A')}\n")
    
    output.append(f"**Context Dependency:** {classification.get('context_dependency', 'Unknown')}\n")
    
    return "\n".join(output)


def format_detailed_case(item: Dict, index: int, json_data: List[Dict] = None) -> str:
    """Format a case with full question context (requires original JSON)."""
    classification = item['classification']
    classifications = classification.get('classifications', [])
    
    output = []
    output.append("\n" + "="*80)
    output.append(f"DETAILED CASE {index} - Problem {item['problem_id']}")
    output.append("="*80)
    
    # If we have the original questions JSON, show the full question
    if json_data:
        prob_data = next((p for p in json_data if p['pn'] == item['problem_id']), None)
        if prob_data:
            output.append(f"\n❓ QUESTION:")
            output.append(f"   {prob_data['question']}")
    
    output.append(f"\n📊 STATISTICS:")
    output.append(f"   Diff: {item['diff']:.4f}")
    output.append(f"   Correct Answer: {item['gt_answer']}")
    output.append(f"   Biased Hint: {item['cue_answer']}")
    
    output.append(f"\n💭 PIVOTAL SENTENCE:")
    output.append(f"   \"{item['high_diff_sentence']}\"")
    
    output.append(f"\n🏷️  CLASSIFICATIONS:")
    for i, c in enumerate(classifications, 1):
        label = "PRIMARY" if i == 1 else f"SECONDARY {i-1}"
        output.append(f"\n   [{label}] {c['mechanism']}")
        output.append(f"   • Tier: {c['tier']}")
        output.append(f"   • Confidence: {c['confidence']}")
        output.append(f"   • Evidence: {c['evidence']}")
    
    output.append(f"\n💡 OVERALL REASONING:")
    output.append(f"   {classification.get('reasoning', 'N/A')}")
    
    output.append(f"\n🔗 CONTEXT DEPENDENCY: {classification.get('context_dependency', 'Unknown')}")
    output.append("\n" + "="*80)
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Inspect classification results')
    parser.add_argument('--json-path', type=str, required=True,
                       help='Path to classifications JSON')
    parser.add_argument('--questions-json', type=str, default=None,
                       help='Path to original questions JSON for full context')
    parser.add_argument('--mechanism', type=str, default=None,
                       help='Mechanism to filter by')
    parser.add_argument('--position', type=str, default='primary', choices=['primary', 'any'],
                       help='Filter by primary mechanism or any mention')
    parser.add_argument('--list-mechanisms', action='store_true',
                       help='List all available mechanisms')
    parser.add_argument('--output', type=str, default=None,
                       help='Output markdown file (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed view with full question context')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading classifications from: {args.json_path}")
    data = load_classifications(args.json_path)
    print(f"✓ Loaded {len(data)} valid classifications\n")
    
    # Load questions if provided
    questions_data = None
    if args.questions_json:
        print(f"Loading questions from: {args.questions_json}")
        with open(args.questions_json, 'r') as f:
            questions_data = json.load(f)
        print(f"✓ Loaded {len(questions_data)} questions\n")
    
    # List mechanisms if requested
    if args.list_mechanisms:
        list_mechanisms(data)
        return
    
    # Filter by mechanism
    if not args.mechanism:
        print("ERROR: Please specify --mechanism or use --list-mechanisms")
        return
    
    filtered = filter_by_mechanism(data, args.mechanism, args.position)
    
    if not filtered:
        print(f"\n❌ No cases found for mechanism: {args.mechanism}")
        print(f"\nUse --list-mechanisms to see available mechanisms")
        return
    
    print(f"\n✓ Found {len(filtered)} case(s) for mechanism: '{args.mechanism}'\n")
    
    # Display cases
    if args.output:
        # Export to markdown
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(f"# Classification Results: {args.mechanism}\n")
            f.write(f"\nFound {len(filtered)} cases\n")
            
            for i, item in enumerate(filtered, 1):
                f.write(format_case_markdown(item, i))
        
        print(f"✓ Exported to: {output_path}")
    else:
        # Display to terminal
        for i, item in enumerate(filtered, 1):
            if args.detailed and questions_data:
                print(format_detailed_case(item, i, questions_data))
            else:
                print(format_case_terminal(item, i, len(filtered)))
        
        print("\n" + "="*80)
        print(f"SUMMARY: {len(filtered)} cases with '{args.mechanism}' as {args.position} mechanism")
        print("="*80)
        
        # Show diff statistics for this mechanism
        diffs = [item['diff'] for item in filtered]
        print(f"\nDiff statistics:")
        print(f"  Mean: {sum(diffs)/len(diffs):.4f}")
        print(f"  Min: {min(diffs):.4f}")
        print(f"  Max: {max(diffs):.4f}")


if __name__ == "__main__":
    main()

