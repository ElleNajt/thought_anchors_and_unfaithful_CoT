#!/usr/bin/env python3
"""
CoT Transplant Ablation Experiment (Randomized Multi-Problem Version w/ Baseline Filter)

This script ablates individual model components and measures their impact on
steering towards the hinted answer. It supports running N randomly-selected
problems and **skips** problems whose baseline steering is below a threshold.

New in this version:
- `--num-problems N` and `--seed` to run N randomly-selected problems.
- **True** no-ablation baseline (fix from earlier version).
- `--min-baseline` (default 0.1): skip problems with baseline < threshold and
  keep sampling until N valid problems are collected or the dataset is exhausted.
- Reproducible RNG for both problem selection and head sampling.

Per selected problem:
1. Run baseline (no ablation).
2. If baseline >= `--min-baseline`, run the ablation sweep.
3. Save per-problem CSV + plot; when N>1, also save a combined CSV.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import einops
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the demo folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CoT_Faithfulness_demo'))
from load_CoT import load_all_problems

# Import from original file
from cot_transplant import count_answer_letters, extract_cot_sentences


class ComponentAblator:
    """Handles ablation of model components (attention heads and MLP layers)."""

    def __init__(self, model):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def ablate_attention_head(self, layer_idx: int, head_idx: int):
        def hook_fn(module, input, output):
            # output often is (attn_output, attn_weights, ...)
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            attn_output = attn_output.clone()

            # Support both (batch, seq, n_heads, head_dim) and (batch, seq, hidden_dim)
            if attn_output.ndim == 4:  # (B, S, H, D)
                attn_output[:, :, head_idx, :] = 0
            elif attn_output.ndim == 3:  # (B, S, D)
                hidden_dim = attn_output.shape[-1]
                n_heads = self._get_num_heads(layer_idx)
                head_dim = hidden_dim // n_heads
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim
                attn_output[:, :, start_idx:end_idx] = 0
            else:
                raise RuntimeError("Unexpected attention output shape: " + str(attn_output.shape))

            if isinstance(output, tuple):
                return (attn_output,) + output[1:]
            else:
                return attn_output

        layer = self._get_layer(layer_idx)
        attn_module = self._get_attention_module(layer)
        hook = attn_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)

    def ablate_mlp(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            else:
                return torch.zeros_like(output)

        layer = self._get_layer(layer_idx)
        mlp_module = self._get_mlp_module(layer)
        hook = mlp_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)

    # ----- helpers to navigate model structure -----
    def _get_layer(self, layer_idx: int):
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model

        if hasattr(base_model, 'layers'):
            return base_model.layers[layer_idx]
        elif hasattr(base_model, 'h'):
            return base_model.h[layer_idx]
        elif hasattr(base_model, 'transformer'):
            if hasattr(base_model.transformer, 'h'):
                return base_model.transformer.h[layer_idx]
            elif hasattr(base_model.transformer, 'layers'):
                return base_model.transformer.layers[layer_idx]
        raise ValueError("Could not find transformer layers in model")

    def _get_attention_module(self, layer):
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        elif hasattr(layer, 'attn'):
            return layer.attn
        elif hasattr(layer, 'attention'):
            return layer.attention
        raise ValueError("Could not find attention module in layer")

    def _get_mlp_module(self, layer):
        if hasattr(layer, 'mlp'):
            return layer.mlp
        elif hasattr(layer, 'feed_forward'):
            return layer.feed_forward
        elif hasattr(layer, 'ffn'):
            return layer.ffn
        raise ValueError("Could not find MLP module in layer")

    def _get_num_heads(self, layer_idx: int):
        layer = self._get_layer(layer_idx)
        attn = self._get_attention_module(layer)
        if hasattr(attn, 'num_heads'):
            return attn.num_heads
        elif hasattr(attn, 'n_head'):
            return attn.n_head
        elif hasattr(attn, 'num_attention_heads'):
            return attn.num_attention_heads
        # fallback to config
        config = self.model.config
        if hasattr(config, 'num_attention_heads'):
            return config.num_attention_heads
        elif hasattr(config, 'n_head'):
            return config.n_head
        raise ValueError("Could not determine number of attention heads")

    def get_num_layers(self):
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model

        if hasattr(base_model, 'layers'):
            return len(base_model.layers)
        elif hasattr(base_model, 'h'):
            return len(base_model.h)
        elif hasattr(base_model, 'transformer'):
            if hasattr(base_model.transformer, 'h'):
                return len(base_model.transformer.h)
            elif hasattr(base_model.transformer, 'layers'):
                return len(base_model.transformer.layers)
        # fallback to config
        config = self.model.config
        if hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            return config.n_layer
        raise ValueError("Could not determine number of layers")


def run_transplant_with_ablation(
    problem_data: Dict,
    model,
    tokenizer,
    ablator: ComponentAblator,
    component_type: str = 'none',  # 'attn_head' | 'mlp' | 'none'
    component_idx: Tuple[int, ...] = (),
    start_sentence: int = 3,
    num_sentences: int = 5,
    n_samples: int = 5,
    max_tokens: int = 1024,
    device: str = "cuda"
) -> Tuple[List[int], List[str]]:
    """Run transplant experiment with a specific component ablated (or none)."""
    ablator.clear_hooks()

    if component_type == 'attn_head':
        layer_idx, head_idx = component_idx
        ablator.ablate_attention_head(layer_idx, head_idx)
    elif component_type == 'mlp':
        layer_idx = component_idx[0]
        ablator.ablate_mlp(layer_idx)
    elif component_type == 'none':
        pass
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    transplant_section = extract_cot_sentences(
        problem_data["reasoning_text"],
        start_sentence=start_sentence,
        num_sentences=num_sentences
    )

    transplanted_prompt = problem_data["question"] + transplant_section + " "

    tokens = tokenizer.encode(transplanted_prompt, return_tensors="pt").to(device)
    many_tokens = einops.repeat(tokens, "1 seq -> n seq", n=n_samples)

    with torch.no_grad():
        outputs = model.generate(
            many_tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    answer_counts, answer_letters = count_answer_letters(generated_texts, tokenizer)

    ablator.clear_hooks()
    return answer_counts, answer_letters


def compute_steering_score(answer_counts: List[int], cue_answer: str, n_samples: int) -> float:
    cue_idx = ord(cue_answer) - ord('A')
    return answer_counts[cue_idx] / n_samples


def compute_answer_metrics(answer_letters: List[str], cue_answer: str, gt_answer: str, n_samples: int) -> Dict[str, float]:
    """
    Compute comprehensive metrics about answer distribution.
    
    Returns:
        Dict with:
        - hinted_rate: fraction choosing hinted answer
        - gt_rate: fraction choosing ground truth answer  
        - valid_rate: fraction with valid answers (not - or |)
        - restoration_quality: gt_rate - hinted_rate (positive = restoring correct answer)
    """
    hinted_count = sum(1 for a in answer_letters if a == cue_answer)
    gt_count = sum(1 for a in answer_letters if a == gt_answer)
    valid_count = sum(1 for a in answer_letters if a not in ['-', '|'])
    
    return {
        'hinted_rate': hinted_count / n_samples,
        'gt_rate': gt_count / n_samples,
        'valid_rate': valid_count / n_samples,
        'restoration_quality': (gt_count / n_samples) - (hinted_count / n_samples)
    }


def compute_baseline_steering(
    problem_data: Dict,
    model,
    tokenizer,
    n_samples: int,
    max_tokens: int,
    device: str
) -> Tuple[float, List[str]]:
    """Compute baseline steering (no ablation) for a problem."""
    ablator = ComponentAblator(model)
    counts, letters = run_transplant_with_ablation(
        problem_data, model, tokenizer, ablator,
        component_type='none', component_idx=(),
        n_samples=n_samples, max_tokens=max_tokens, device=device
    )
    steering = compute_steering_score(counts, problem_data['cue_answer'], n_samples)
    return steering, letters


def run_ablation_experiment(
    problem_data: Dict,
    model,
    tokenizer,
    n_samples: int = 5,
    max_tokens: int = 1024,
    device: str = "cuda",
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    sample_rate: float = 1.0,
    test_mlps: bool = True,
    rng: Optional[np.random.Generator] = None,
    baseline_steering_override: Optional[float] = None,
    baseline_letters_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run ablation experiment on all components for a single problem."""
    ablator = ComponentAblator(model)

    # Baseline (no ablation) — use override if provided
    if baseline_steering_override is None:
        print("Running baseline (no ablation)...")
        baseline_counts, baseline_letters = run_transplant_with_ablation(
            problem_data, model, tokenizer, ablator,
            component_type='none', component_idx=(),
            n_samples=n_samples, max_tokens=max_tokens, device=device
        )
        baseline_steering = compute_steering_score(baseline_counts, problem_data['cue_answer'], n_samples)
    else:
        baseline_steering = float(baseline_steering_override)
        baseline_letters = baseline_letters_override if baseline_letters_override is not None else []
        print("Using precomputed baseline (no ablation)...")

    print(f"Baseline steering towards {problem_data['cue_answer']}: {baseline_steering:.3f}")
    if baseline_letters_override is None:
        print(f"Baseline answers: {baseline_letters}")

    num_layers = ablator.get_num_layers()
    num_heads = ablator._get_num_heads(0)  # assume uniform across layers

    if layer_end is None:
        layer_end = num_layers

    layers_to_test = list(range(layer_start, layer_end))
    heads_per_layer = max(1, int(num_heads * sample_rate))

    if sample_rate < 1.0:
        print(f"\n⚡ SAMPLING MODE: Testing {heads_per_layer}/{num_heads} heads per layer")

    total_attn = len(layers_to_test) * heads_per_layer
    total_mlp = len(layers_to_test) if test_mlps else 0
    total_components = total_attn + total_mlp

    print(f"\nModel has {num_layers} layers, {num_heads} heads per layer")
    print(f"Testing layers {layer_start} to {layer_end-1}")
    print(f"Total components to test: {total_attn} attention heads + {total_mlp} MLPs = {total_components}")

    results = []

    # Attention heads
    print("\nTesting attention heads...")
    for layer_idx in layers_to_test:
        if sample_rate < 1.0:
            if rng is None:
                heads_to_test = np.random.choice(num_heads, heads_per_layer, replace=False)
            else:
                heads_to_test = rng.choice(num_heads, heads_per_layer, replace=False)
            heads_to_test = sorted(map(int, heads_to_test))
        else:
            heads_to_test = range(num_heads)

        for head_idx in tqdm(heads_to_test, desc=f"Layer {layer_idx}/{layer_end-1}", leave=True):
            try:
                answer_counts, answer_letters = run_transplant_with_ablation(
                    problem_data, model, tokenizer, ablator,
                    component_type='attn_head', component_idx=(layer_idx, head_idx),
                    n_samples=n_samples, max_tokens=max_tokens, device=device
                )
                steering = compute_steering_score(answer_counts, problem_data['cue_answer'], n_samples)
                steering_change = steering - baseline_steering
                
                # Compute comprehensive metrics
                metrics = compute_answer_metrics(
                    answer_letters, 
                    problem_data['cue_answer'], 
                    problem_data['gt_answer'],
                    n_samples
                )
                
                results.append({
                    'component_type': 'attn_head',
                    'layer': layer_idx,
                    'head': head_idx,
                    'component_name': f'L{layer_idx}H{head_idx}',
                    'baseline_steering': baseline_steering,
                    'ablated_steering': steering,
                    'steering_change': steering_change,
                    'steering_drop': -steering_change,
                    'gt_rate': metrics['gt_rate'],
                    'hinted_rate': metrics['hinted_rate'],
                    'valid_rate': metrics['valid_rate'],
                    'restoration_quality': metrics['restoration_quality'],
                    'answer_letters': ''.join(answer_letters),
                })
            except Exception as e:
                print(f"\nError testing L{layer_idx}H{head_idx}: {e}")
                continue

    # MLP layers
    if test_mlps:
        print("\nTesting MLP layers...")
        for layer_idx in tqdm(layers_to_test, desc="MLP layers"):
            try:
                answer_counts, answer_letters = run_transplant_with_ablation(
                    problem_data, model, tokenizer, ablator,
                    component_type='mlp', component_idx=(layer_idx,),
                    n_samples=n_samples, max_tokens=max_tokens, device=device
                )
                steering = compute_steering_score(answer_counts, problem_data['cue_answer'], n_samples)
                steering_change = steering - baseline_steering
                
                # Compute comprehensive metrics
                metrics = compute_answer_metrics(
                    answer_letters, 
                    problem_data['cue_answer'], 
                    problem_data['gt_answer'],
                    n_samples
                )
                
                results.append({
                    'component_type': 'mlp',
                    'layer': layer_idx,
                    'head': -1,
                    'component_name': f'L{layer_idx}_MLP',
                    'baseline_steering': baseline_steering,
                    'ablated_steering': steering,
                    'steering_change': steering_change,
                    'steering_drop': -steering_change,
                    'gt_rate': metrics['gt_rate'],
                    'hinted_rate': metrics['hinted_rate'],
                    'valid_rate': metrics['valid_rate'],
                    'restoration_quality': metrics['restoration_quality'],
                    'answer_letters': ''.join(answer_letters),
                })
            except Exception as e:
                print(f"\nError testing MLP layer {layer_idx}: {e}")
                continue

    df = pd.DataFrame(results)
    if not df.empty:
        # Sort by restoration_quality (higher = more restoration toward ground truth)
        df = df.sort_values('restoration_quality', ascending=False).reset_index(drop=True)
    return df


def plot_top_components(df: pd.DataFrame, top_n: int = 10, output_path: str = 'cot_transplant_ablation.png'):
    top_df = df.head(top_n)
    if top_df.empty:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#27ae60' if q > 0 else '#e74c3c' for q in top_df['restoration_quality']]
    bars = ax.bar(range(len(top_df)), top_df['restoration_quality'], color=colors)

    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Restoration Quality (GT Rate - Hinted Rate)', fontsize=12)
    ax.set_title(f'Top {top_n} Components by Ground Truth Restoration When Ablated', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_df)))
    ax.set_xticklabels(top_df['component_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Restores GT (positive)'), 
        Patch(facecolor='#e74c3c', label='Breaks model (negative)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    for bar, value in zip(bars, top_df['restoration_quality']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='CoT Transplant Ablation Experiment (Randomized Multi-Problem + Baseline Filter)')
    # existing args
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples per component (default: 5)')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Max tokens to generate (default: 1024)')
    parser.add_argument('--layer-start', type=int, default=0, help='First layer to test (default: 0)')
    parser.add_argument('--layer-end', type=int, default=None, help='Last layer to test, exclusive (default: all layers)')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of attention heads to sample per layer (default: 1.0 = all)')
    parser.add_argument('--no-mlps', action='store_true', help='Skip testing MLP layers')
    parser.add_argument('--problem-idx', type=int, default=0, help='Problem index to use if not random (default: 0)')
    parser.add_argument('--output-prefix', type=str, default='cot_transplant_ablation', help='Prefix for output files')

    # new args
    parser.add_argument('--num-problems', type=int, default=1, help='Number of randomly selected valid problems to run (default: 1)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for problem and head sampling (default: 123)')
    parser.add_argument('--min-baseline', type=float, default=0.1, help='Minimum baseline steering required to include a problem (default: 0.1)')

    args = parser.parse_args()

    print("="*80)
    print("CoT TRANSPLANT ABLATION EXPERIMENT — RANDOMIZED MULTI-PROBLEM + BASELINE FILTER")
    print("="*80)

    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load problems
    print("\nLoading problem data...")
    original_dir = os.getcwd()
    demo_dir = os.path.join(os.path.dirname(__file__), 'CoT_Faithfulness_demo')
    os.chdir(demo_dir)
    try:
        data = load_all_problems()
        print(f"Loaded {len(data)} problems")
    finally:
        os.chdir(original_dir)

    # Determine which problems to run
    n_total = len(data)
    target_n = max(1, min(args.num_problems, n_total))
    rng = np.random.default_rng(args.seed)

    # Always randomize to find valid problems that meet baseline threshold
    # We'll keep trying until we find target_n valid problems or exhaust all candidates
    candidate_order = list(range(n_total))
    indices_planned = rng.permutation(candidate_order).tolist()
    print(f"\nRandomly sampling problems (seed={args.seed}) to find {target_n} with baseline >= {args.min_baseline:.3f}")
    print(f"Candidate order: {indices_planned[:10]}{'...' if len(indices_planned) > 10 else ''}")

    valid_indices: List[int] = []
    baseline_cache: Dict[int, Tuple[float, List[str]]] = {}
    skipped_indices: List[int] = []

    # Sweep through the randomized candidate list, computing baseline and filtering
    for idx in indices_planned:
        if len(valid_indices) >= target_n:
            break
        problem = data[idx]
        pn = problem.get('pn', idx)
        print("\n" + "-"*80)
        print(f"Evaluating baseline for candidate problem #{pn} (index {idx})...")
        baseline_steering, baseline_letters = compute_baseline_steering(
            problem, model, tokenizer, args.n_samples, args.max_tokens, device
        )
        print(f"Baseline steering: {baseline_steering:.3f} (threshold {args.min_baseline:.3f})")
        if baseline_steering >= args.min_baseline:
            valid_indices.append(idx)
            baseline_cache[idx] = (baseline_steering, baseline_letters)
            print(f"✓ Accepted problem #{pn} (index {idx})")
        else:
            skipped_indices.append(idx)
            print(f"✗ Skipping problem #{pn} (index {idx}) due to low baseline")

    if len(valid_indices) < target_n:
        print("\nWARNING: Not enough valid problems met the baseline threshold.")
        print(f"Requested: {target_n}, Accepted: {len(valid_indices)}, Skipped: {len(skipped_indices)}")
        print("Proceeding with the accepted set.")

    combined_results: List[pd.DataFrame] = []

    for idx in valid_indices:
        problem = data[idx]
        pn = problem.get('pn', idx)
        base_steer, base_letters = baseline_cache[idx]

        print(f"\nSelected problem #{pn} (index {idx})")
        print(f"Ground truth answer: {problem['gt_answer']}")
        print(f"Hinted answer: {problem['cue_answer']}")
        
        # Show baseline metrics
        baseline_metrics = compute_answer_metrics(
            base_letters, 
            problem['cue_answer'], 
            problem['gt_answer'],
            args.n_samples
        )
        print(f"\nBaseline metrics (with transplanted CoT):")
        print(f"  Hinted answer rate: {baseline_metrics['hinted_rate']:.3f}")
        print(f"  Ground truth rate: {baseline_metrics['gt_rate']:.3f}")
        print(f"  Valid answer rate: {baseline_metrics['valid_rate']:.3f}")
        print(f"  Baseline answers: {(''.join(base_letters))}")

        print("\n" + "="*80)
        print("RUNNING ABLATION EXPERIMENT")
        print("="*80)

        results_df = run_ablation_experiment(
            problem, model, tokenizer,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            device=device,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            sample_rate=args.sample_rate,
            test_mlps=not args.no_mlps,
            rng=rng,
            baseline_steering_override=base_steer,
            baseline_letters_override=base_letters,
        )

        # Annotate problem identity for aggregation
        results_df.insert(0, 'problem_index', idx)
        results_df.insert(1, 'problem_pn', pn)

        # Save per-problem results
        output_csv = f"{args.output_prefix}_pn{pn}_idx{idx}_results.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")

        # Display top components
        print("\n" + "="*80)
        print("TOP 20 COMPONENTS BY GROUND TRUTH RESTORATION")
        print("="*80)
        print("Legend: restoration_quality = gt_rate - hinted_rate (positive = restoring GT)")
        if not results_df.empty:
            print(results_df.head(20).to_string(index=False))
        else:
            print("No results (empty DataFrame)")

        # Plot top 10 per problem
        plot_output = f"{args.output_prefix}_pn{pn}_idx{idx}_plot.png"
        plot_top_components(results_df, top_n=10, output_path=plot_output)

        # Basic stats
        if not results_df.empty:
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            print(f"Baseline steering towards hinted answer: {results_df['baseline_steering'].iloc[0]:.3f}")
            print(f"\nTop component: {results_df.iloc[0]['component_name']}")
            print(f"  Steering drop: {results_df.iloc[0]['steering_drop']:.3f}")
            print(f"  Ablated steering: {results_df.iloc[0]['ablated_steering']:.3f}")

            attn_results = results_df[results_df['component_type'] == 'attn_head']
            mlp_results = results_df[results_df['component_type'] == 'mlp']

            if len(attn_results) > 0:
                print(f"\nAttention heads:")
                print(f"  Mean steering drop: {attn_results['steering_drop'].mean():.3f}")
                print(f"  Std steering drop: {attn_results['steering_drop'].std():.3f}")
                print(f"  Max steering drop: {attn_results['steering_drop'].max():.3f}")

            if len(mlp_results) > 0:
                print(f"\nMLP layers:")
                print(f"  Mean steering drop: {mlp_results['steering_drop'].mean():.3f}")
                print(f"  Std steering drop: {mlp_results['steering_drop'].std():.3f}")
                print(f"  Max steering drop: {mlp_results['steering_drop'].max():.3f}")

        combined_results.append(results_df)

    # Save combined if we ran > 1 problem
    if len(combined_results) > 1:
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_csv = f"{args.output_prefix}_combined_results.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✓ Combined results saved to: {combined_csv}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
