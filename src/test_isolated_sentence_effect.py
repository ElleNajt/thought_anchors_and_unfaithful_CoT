"""
Test if high cue_p diff sentences alone (without CoT context) can shift answers.

For each problem:
1. Find the sentence with highest diff (cue_p - cue_p_prev)
2. Test just that sentence appended to the question
3. Compare to baseline (no sentence)

Usage:
    python test_isolated_sentence_effect.py --csv-path CoT_Faithfulness_demo/faith_counterfactual_qwen-14b_demo.csv --json-path CoT_Faithfulness_demo/in_text/Professor_itc_failure_threshold0.15_correct_base_no_mention.json --num-samples 20
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Rate limiting
MAX_CONCURRENT = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


def extract_answer(response_text):
    """Extract letter answer from response."""
    text_upper = response_text.strip().upper()

    # Look for explicit answer patterns
    lines = response_text.strip().split("\n")
    for line in reversed(lines[-5:]):
        line_clean = line.strip()
        if len(line_clean) == 1 and line_clean in ["A", "B", "C", "D"]:
            return line_clean
        for letter in ["A", "B", "C", "D"]:
            if f"answer: {letter.lower()}" in line_clean.lower():
                return letter
            if f"answer is {letter.lower()}" in line_clean.lower():
                return letter

    # Fallback
    for char in reversed(text_upper):
        if char in ["A", "B", "C", "D"]:
            return char

    return None


async def resample_with_sentence(base_prompt, sentence, temperature=1.0, mode="think"):
    """Resample with just one sentence appended.

    Args:
        mode: 'baseline', 'think', or 'forced'
            - baseline: just the question
            - think: sentence in <think> tags
            - forced: sentence followed by "Therefore the answer is:"

    Returns:
        tuple: (answer, response_text, reasoning_trace) or (None, None, None) on failure
    """
    if mode == "baseline":
        prompt = base_prompt
    elif mode == "think":
        prompt = f"{base_prompt}\n<think>\n{sentence}\n"
    elif mode == "forced":
        prompt = (
            f"{base_prompt}\n <think>\n {sentence}\n</think> Therefore the answer is:"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"reasoning": {"effort": "high", "exclude": False}},
                ),
                timeout=180.0,
            )
            response_text = response.choices[0].message.content

            # Extract reasoning trace if available
            reasoning_trace = None
            if hasattr(response.choices[0].message, "reasoning"):
                reasoning_trace = response.choices[0].message.reasoning

            answer = extract_answer(response_text)

            if answer is None:
                print(f"        ⚠ Failed to extract answer", flush=True)

            return (answer, response_text, reasoning_trace)
        except asyncio.TimeoutError:
            print(f"        ⚠ Timeout", flush=True)
            return (None, None, None)
        except Exception as e:
            print(f"        ⚠ Error: {type(e).__name__}: {str(e)}", flush=True)
            return (None, None, None)


async def test_high_diff_sentence(
    problem_id, base_prompt, high_diff_sentence, cue_answer, diff, num_samples=20
):
    """
    Test high diff sentence in three conditions: baseline, think tags, and forced immediate answer.

    Returns dict with results from all three conditions.
    """
    print(f"\n  Testing problem {problem_id}")
    print(f"  High diff sentence (diff={diff:+.4f}):")
    print(
        f"    '{high_diff_sentence[:80]}{'...' if len(high_diff_sentence) > 80 else ''}'"
    )

    # Baseline: no sentence (SKIPPED)
    baseline_answers = []
    baseline_rollouts = []
    baseline_reasoning = []
    baseline_cue_pct = None

    # With high diff sentence in think tags
    print(f"  With sentence in <think> tags...", flush=True)
    tasks = [
        resample_with_sentence(
            base_prompt, high_diff_sentence, temperature=1.0, mode="think"
        )
        for _ in range(num_samples)
    ]
    think_results = await asyncio.gather(*tasks)
    think_answers = [a for a, r, rt in think_results if a is not None]
    think_rollouts = [r for a, r, rt in think_results if a is not None]
    think_reasoning = [rt for a, r, rt in think_results if a is not None]
    think_cue_pct = (
        think_answers.count(cue_answer) / len(think_answers) if think_answers else None
    )
    print(
        f"    {think_cue_pct:.2%} cue answer ({len(think_answers)} samples)"
        if think_cue_pct is not None
        else f"    FAILED - no valid samples"
    )

    # With forced immediate answer (SKIPPED)
    forced_answers = []
    forced_rollouts = []
    forced_reasoning = []
    forced_cue_pct = None

    # Calculate deltas
    delta_think = (
        (think_cue_pct - baseline_cue_pct)
        if (think_cue_pct is not None and baseline_cue_pct is not None)
        else None
    )
    delta_forced = (
        (forced_cue_pct - baseline_cue_pct)
        if (forced_cue_pct is not None and baseline_cue_pct is not None)
        else None
    )

    print(
        f"  Delta (think): {delta_think:+.2%}"
        if delta_think is not None
        else "  Delta (think): FAILED"
    )
    print(
        f"  Delta (forced): {delta_forced:+.2%}"
        if delta_forced is not None
        else "  Delta (forced): FAILED"
    )

    return {
        "problem_id": int(problem_id) if hasattr(problem_id, "item") else problem_id,
        "high_diff_sentence": high_diff_sentence,
        "diff": float(diff) if hasattr(diff, "item") else diff,
        "cue_answer": cue_answer,
        "baseline_cue_pct": baseline_cue_pct,
        "baseline_answers": baseline_answers,
        "baseline_rollouts": baseline_rollouts,
        "baseline_reasoning": baseline_reasoning,
        "think_cue_pct": think_cue_pct,
        "think_answers": think_answers,
        "think_rollouts": think_rollouts,
        "think_reasoning": think_reasoning,
        "forced_cue_pct": forced_cue_pct,
        "forced_answers": forced_answers,
        "forced_rollouts": forced_rollouts,
        "forced_reasoning": forced_reasoning,
        "delta_think": delta_think,
        "delta_forced": delta_forced,
        "num_samples": num_samples,
    }


async def main():
    parser = argparse.ArgumentParser(description="Test high diff sentences alone")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--min-diff",
        type=float,
        default=0.35,
        help="Minimum diff to consider (default: 0.35)",
    )
    args = parser.parse_args()

    # Load CSV
    print(f"Loading CSV from {args.csv_path}", flush=True)
    df = pd.read_csv(args.csv_path)

    # Calculate diff
    df["diff"] = df["cue_p"] - df["cue_p_prev"]

    # Load JSON
    print(f"Loading JSON from {args.json_path}", flush=True)
    with open(args.json_path, "r") as f:
        json_data = json.load(f)

    # Create lookup dict for questions
    question_lookup = {item["pn"]: item["question"] for item in json_data}

    # Find high diff sentences for each problem
    print(f"\nFinding high diff sentences (min diff: {args.min_diff})...", flush=True)

    high_diff_problems = []
    for pn in df["pn"].unique():
        problem_df = df[df["pn"] == pn].sort_values("sentence_num")

        # Find max diff sentence
        max_diff_idx = problem_df["diff"].idxmax()
        max_diff_row = problem_df.loc[max_diff_idx]

        if max_diff_row["diff"] >= args.min_diff:
            high_diff_problems.append(
                {
                    "pn": pn,
                    "sentence": max_diff_row["sentence"],
                    "diff": max_diff_row["diff"],
                    "cue_answer": max_diff_row["cue_answer"],
                    "gt_answer": max_diff_row["gt_answer"],
                }
            )

    print(f"Found {len(high_diff_problems)} problems with diff >= {args.min_diff}")

    # Run experiments
    print(f"\nRunning experiments...")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples per condition: {args.num_samples}\n")

    results = []
    for i, prob in enumerate(high_diff_problems, 1):
        print(f"\n[{i}/{len(high_diff_problems)}] Problem {prob['pn']}", flush=True)

        base_prompt = question_lookup.get(prob["pn"])
        if not base_prompt:
            print(f"  ⚠ Question not found for problem {prob['pn']}, skipping")
            continue

        result = await test_high_diff_sentence(
            prob["pn"],
            base_prompt,
            prob["sentence"],
            prob["cue_answer"],
            prob["diff"],
            args.num_samples,
        )
        result["gt_answer"] = prob["gt_answer"]
        results.append(result)

    # Save results - split into summary and detailed rollouts
    output_dir = Path(args.csv_path).parent
    output_file = output_dir / "isolated_sentence_effect_results.json"
    rollouts_file = output_dir / "isolated_sentence_effect_rollouts.json"

    # Create summary results (without rollouts/reasoning)
    summary_results = []
    detailed_rollouts = []

    for result in results:
        # Summary version
        summary = {
            "problem_id": result["problem_id"],
            "high_diff_sentence": result["high_diff_sentence"],
            "diff": result["diff"],
            "cue_answer": result["cue_answer"],
            "gt_answer": result["gt_answer"],
            "baseline_cue_pct": result["baseline_cue_pct"],
            "baseline_answers": result["baseline_answers"],
            "think_cue_pct": result["think_cue_pct"],
            "think_answers": result["think_answers"],
            "forced_cue_pct": result["forced_cue_pct"],
            "forced_answers": result["forced_answers"],
            "delta_think": result["delta_think"],
            "delta_forced": result["delta_forced"],
            "num_samples": result["num_samples"],
        }
        summary_results.append(summary)

        # Detailed rollouts version
        rollout = {
            "problem_id": result["problem_id"],
            "baseline_rollouts": result["baseline_rollouts"],
            "baseline_reasoning": result["baseline_reasoning"],
            "think_rollouts": result["think_rollouts"],
            "think_reasoning": result["think_reasoning"],
            "forced_rollouts": result["forced_rollouts"],
            "forced_reasoning": result["forced_reasoning"],
        }
        detailed_rollouts.append(rollout)

    with open(output_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    with open(rollouts_file, "w") as f:
        json.dump(detailed_rollouts, f, indent=2)

    print(f"\n\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total problems tested: {len(results)}")
    failed_think_count = len([r for r in results if r["delta_think"] is None])
    failed_forced_count = len([r for r in results if r["delta_forced"] is None])
    if failed_think_count > 0 or failed_forced_count > 0:
        print(f"Failed (think): {failed_think_count}")
        print(f"Failed (forced): {failed_forced_count}")

    # Calculate statistics for THINK condition (filter out None values)
    valid_think = [r for r in results if r["delta_think"] is not None]
    if valid_think:
        positive_think = sum(1 for r in valid_think if r["delta_think"] > 0)
        negative_think = sum(1 for r in valid_think if r["delta_think"] < 0)
        zero_think = sum(1 for r in valid_think if r["delta_think"] == 0)

        mean_delta_think = sum(r["delta_think"] for r in valid_think) / len(valid_think)
        mean_baseline = sum(r["baseline_cue_pct"] for r in valid_think) / len(
            valid_think
        )
        mean_think = sum(r["think_cue_pct"] for r in valid_think) / len(valid_think)

        print(f"\n=== THINK TAG CONDITION ===")
        print(f"Mean baseline cue%: {mean_baseline:.2%}")
        print(f"Mean think cue%: {mean_think:.2%}")
        print(f"Mean delta: {mean_delta_think:+.2%}")
        print(f"\nDelta distribution (of {len(valid_think)} valid results):")
        print(
            f"  Positive (think > baseline): {positive_think} ({positive_think / len(valid_think):.1%})"
        )
        print(
            f"  Negative (think < baseline): {negative_think} ({negative_think / len(valid_think):.1%})"
        )
        print(f"  Zero: {zero_think} ({zero_think / len(valid_think):.1%})")

    # Calculate statistics for FORCED condition
    valid_forced = [r for r in results if r["delta_forced"] is not None]
    if valid_forced:
        positive_forced = sum(1 for r in valid_forced if r["delta_forced"] > 0)
        negative_forced = sum(1 for r in valid_forced if r["delta_forced"] < 0)
        zero_forced = sum(1 for r in valid_forced if r["delta_forced"] == 0)

        mean_delta_forced = sum(r["delta_forced"] for r in valid_forced) / len(
            valid_forced
        )
        mean_baseline_forced = sum(r["baseline_cue_pct"] for r in valid_forced) / len(
            valid_forced
        )
        mean_forced = sum(r["forced_cue_pct"] for r in valid_forced) / len(valid_forced)

        print(f"\n=== FORCED IMMEDIATE ANSWER CONDITION ===")
        print(f"Mean baseline cue%: {mean_baseline_forced:.2%}")
        print(f"Mean forced cue%: {mean_forced:.2%}")
        print(f"Mean delta: {mean_delta_forced:+.2%}")
        print(f"\nDelta distribution (of {len(valid_forced)} valid results):")
        print(
            f"  Positive (forced > baseline): {positive_forced} ({positive_forced / len(valid_forced):.1%})"
        )
        print(
            f"  Negative (forced < baseline): {negative_forced} ({negative_forced / len(valid_forced):.1%})"
        )
        print(f"  Zero: {zero_forced} ({zero_forced / len(valid_forced):.1%})")

    print(f"\n✓ Saved summary to {output_file}")
    print(f"✓ Saved detailed rollouts to {rollouts_file}")


if __name__ == "__main__":
    asyncio.run(main())
