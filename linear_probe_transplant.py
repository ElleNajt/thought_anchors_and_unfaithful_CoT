#!/usr/bin/env python3
"""
Linear Probe Analysis for TRANSPLANTED CoT

This script trains linear probes to predict the hint answer from model activations
in TRANSPLANTED examples - where hint-influenced CoT is placed after a no-hint prompt.

Key Question: Does the transplanted CoT carry latent representations of the hint
that was used to generate it, even when the hint is never shown?
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Add the demo folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CoT_Faithfulness_demo'))
from load_CoT import load_all_problems


def extract_cot_sentences(cot_text, start_sentence=0, num_sentences=3):
    """
    Extract a section of CoT by sentence boundaries.
    
    Args:
        cot_text: The full CoT reasoning text
        start_sentence: Which sentence to start from (0-indexed)
        num_sentences: How many sentences to extract
    
    Returns:
        str: The extracted section of CoT
    """
    # Simple sentence splitting
    sentences = [s.strip() + " " for s in cot_text.split('.') if s.strip()]
    
    end_sentence = start_sentence + num_sentences
    extracted = '.'.join(sentences[start_sentence:end_sentence])
    
    return extracted.strip()


def build_transplanted_prompt(problem_data, start_sentence=3, num_sentences=5):
    """
    Build a transplanted prompt: [no-hint question] + [hint-influenced CoT snippet]
    
    Returns:
        transplanted_prompt (str): The combined prompt
        num_transplanted_sentences (int): How many sentences were transplanted
    """
    # Extract transplant section from hint-influenced CoT
    transplant_section = extract_cot_sentences(
        problem_data["reasoning_text"],
        start_sentence=start_sentence,
        num_sentences=num_sentences
    )
    
    # Build: no-hint question + transplanted section
    transplanted_prompt = problem_data["question"] + transplant_section + " "
    
    return transplanted_prompt, num_sentences


def get_layer_activations(model, tokens, layer_idx):
    """
    Get activations from a specific layer for given tokens.
    
    Args:
        model: The language model
        tokens: Input token tensor
        layer_idx: Which layer to extract from
    
    Returns:
        Tensor of activations [batch, seq_len, hidden_dim]
    """
    activations = {}
    
    def hook_fn(module, input, output):
        # Handle different output formats
        if isinstance(output, tuple):
            activations['hidden'] = output[0].detach()
        else:
            activations['hidden'] = output.detach()
    
    # Register hook on the specified layer
    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            model(tokens)
        return activations['hidden']
    finally:
        hook.remove()


def extract_sentence_positions_in_prompt(transplanted_prompt, question_only, tokenizer):
    """
    Find token positions corresponding to sentence boundaries in the transplanted CoT.
    
    Returns:
        List of (sentence_num, end_token_pos) tuples, relative to start of transplanted section
    """
    # Get length of question-only part
    question_tokens = tokenizer.encode(question_only, return_tensors="pt")
    question_len = question_tokens.shape[1]
    
    # Get full transplanted prompt
    full_tokens = tokenizer.encode(transplanted_prompt, return_tensors="pt")
    
    # Extract just the transplanted section
    transplanted_section = transplanted_prompt[len(question_only):]
    
    # Split transplanted section by sentences
    sentences = [s.strip() for s in transplanted_section.split('.') if s.strip()]
    
    positions = []
    accumulated_text = question_only
    
    for i, sentence in enumerate(sentences):
        accumulated_text += sentence + ". "
        tokens = tokenizer.encode(accumulated_text, return_tensors="pt")
        end_pos = tokens.shape[1] - 1
        positions.append((i, end_pos))
    
    return positions


def collect_activations_for_transplants(
    problems,
    model,
    tokenizer,
    device,
    start_sentence=3,
    num_sentences=5,
    max_problems=None
):
    """
    Collect activations from TRANSPLANTED prompts across all layers and sentence positions.
    
    Returns:
        dict with structure:
            {
                layer_idx: {
                    sentence_num: {
                        'activations': list of activation vectors,
                        'labels': list of hint answers (0=A, 1=B, 2=C, 3=D),
                        'pns': list of problem numbers
                    }
                }
            }
    """
    n_layers = model.config.num_hidden_layers
    
    # Initialize data structure
    data = defaultdict(lambda: defaultdict(lambda: {
        'activations': [],
        'labels': [],
        'pns': []
    }))
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"\nCollecting activations from {len(problems)} TRANSPLANTED examples across {n_layers} layers...")
    print(f"Transplanting sentences {start_sentence} to {start_sentence+num_sentences} from hint-influenced CoT")
    
    for problem in tqdm(problems, desc="Processing transplanted prompts"):
        try:
            # Build transplanted prompt
            transplanted_prompt, n_sentences = build_transplanted_prompt(
                problem,
                start_sentence=start_sentence,
                num_sentences=num_sentences
            )
            
            cue_answer = problem['cue_answer']
            label = ord(cue_answer) - ord('A')
            
            # Tokenize transplanted prompt
            tokens = tokenizer.encode(transplanted_prompt, return_tensors="pt").to(device)
            
            # Get sentence positions in the transplanted section
            sentence_positions = extract_sentence_positions_in_prompt(
                transplanted_prompt,
                problem["question"],
                tokenizer
            )
            
            # For each layer
            for layer_idx in range(n_layers):
                # Get activations for this layer
                layer_acts = get_layer_activations(model, tokens, layer_idx)
                
                # For each sentence position in the transplanted section
                for sent_num, token_pos in sentence_positions:
                    # Get activation at this position
                    if token_pos < layer_acts.shape[1]:
                        act_vector = layer_acts[0, token_pos, :].cpu().numpy()
                        
                        data[layer_idx][sent_num]['activations'].append(act_vector)
                        data[layer_idx][sent_num]['labels'].append(label)
                        data[layer_idx][sent_num]['pns'].append(problem['pn'])
        
        except Exception as e:
            print(f"\nError processing problem {problem.get('pn', 'unknown')}: {e}")
            continue
    
    return data


def train_probes_and_evaluate(data, test_size=0.2, random_state=42):
    """
    Train linear probes for each layer and sentence position, evaluate on test set.
    
    Returns:
        DataFrame with columns: layer, sentence, train_acc, test_acc, n_samples
    """
    results = []
    
    print("\nTraining linear probes on transplanted examples...")
    
    for layer_idx in tqdm(sorted(data.keys()), desc="Layers"):
        for sent_num in sorted(data[layer_idx].keys()):
            layer_sent_data = data[layer_idx][sent_num]
            
            X = np.array(layer_sent_data['activations'])
            y = np.array(layer_sent_data['labels'])
            
            # Need at least 10 samples and at least 2 classes
            if len(X) < 10 or len(np.unique(y)) < 2:
                continue
            
            try:
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train logistic regression probe
                probe = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    multi_class='multinomial'
                )
                probe.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_acc = probe.score(X_train_scaled, y_train)
                test_acc = probe.score(X_test_scaled, y_test)
                
                results.append({
                    'layer': layer_idx,
                    'sentence': sent_num,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'n_samples': len(X),
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
                
            except Exception as e:
                print(f"\nError training probe for layer {layer_idx}, sentence {sent_num}: {e}")
                continue
    
    return pd.DataFrame(results)


def plot_probe_results(results_df, save_path='linear_probe_transplant_heatmap.png'):
    """
    Create a heatmap showing probe accuracy across layers and sentences.
    """
    # Pivot to create heatmap data
    heatmap_data = results_df.pivot(
        index='layer',
        columns='sentence',
        values='test_acc'
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Test accuracy heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0.25,
        vmax=1.0,
        center=0.5,
        ax=ax1,
        cbar_kws={'label': 'Test Accuracy'}
    )
    ax1.set_xlabel('Sentence Position in Transplanted CoT', fontsize=12)
    ax1.set_ylabel('Layer', fontsize=12)
    ax1.set_title('Linear Probe Test Accuracy: Predicting Hint Answer\nfrom TRANSPLANTED CoT Activations (Hint Never Shown)', 
                  fontsize=14, fontweight='bold')
    
    # Sample count heatmap
    sample_data = results_df.pivot(
        index='layer',
        columns='sentence',
        values='n_samples'
    )
    sns.heatmap(
        sample_data,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        ax=ax2,
        cbar_kws={'label': 'Number of Samples'}
    )
    ax2.set_xlabel('Sentence Position in Transplanted CoT', fontsize=12)
    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title('Sample Counts per Layer and Sentence', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig


def plot_comparison_charts(results_df, save_path_prefix='linear_probe_transplant'):
    """
    Create comparison plots showing layer and sentence trends.
    """
    # Layer trends
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    layer_avg = results_df.groupby('layer')['test_acc'].agg(['mean', 'std', 'count']).reset_index()
    
    ax1.plot(layer_avg['layer'], layer_avg['mean'], marker='o', linewidth=2, markersize=8)
    ax1.fill_between(
        layer_avg['layer'],
        layer_avg['mean'] - layer_avg['std'],
        layer_avg['mean'] + layer_avg['std'],
        alpha=0.3
    )
    ax1.axhline(y=0.25, color='r', linestyle='--', label='Chance (25%)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Test Accuracy (averaged across sentences)', fontsize=12)
    ax1.set_title('Linear Probe Accuracy by Layer\nTransplanted CoT (Hint Never Shown)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_layers.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path_prefix}_layers.png")
    
    # Sentence trends
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sent_avg = results_df.groupby('sentence')['test_acc'].agg(['mean', 'std', 'count']).reset_index()
    
    ax2.plot(sent_avg['sentence'], sent_avg['mean'], marker='o', linewidth=2, markersize=8)
    ax2.fill_between(
        sent_avg['sentence'],
        sent_avg['mean'] - sent_avg['std'],
        sent_avg['mean'] + sent_avg['std'],
        alpha=0.3
    )
    ax2.axhline(y=0.25, color='r', linestyle='--', label='Chance (25%)')
    ax2.set_xlabel('Sentence Position in Transplanted CoT', fontsize=12)
    ax2.set_ylabel('Test Accuracy (averaged across layers)', fontsize=12)
    ax2.set_title('Linear Probe Accuracy by Sentence Position\nTransplanted CoT (Hint Never Shown)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_sentences.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path_prefix}_sentences.png")
    
    return fig1, fig2


def main():
    """
    Main function to run linear probe analysis on transplanted examples.
    """
    print("="*80)
    print("LINEAR PROBE ANALYSIS: TRANSPLANTED CoT")
    print("Testing if hint is decodable when hint-influenced CoT is transplanted")
    print("="*80)
    
    # Configuration
    MAX_PROBLEMS = 30  # Use subset for faster analysis
    START_SENTENCE = 3  # Which sentence to start transplanting from
    NUM_SENTENCES = 5   # How many sentences to transplant
    
    # Load model
    print("\nLoading model...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model has {model.config.num_hidden_layers} layers")
    
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
    
    # Collect activations from transplanted examples
    activation_data = collect_activations_for_transplants(
        data,
        model,
        tokenizer,
        device,
        start_sentence=START_SENTENCE,
        num_sentences=NUM_SENTENCES,
        max_problems=MAX_PROBLEMS
    )
    
    # Train probes and evaluate
    results_df = train_probes_and_evaluate(activation_data)
    
    # Save results
    results_csv = 'linear_probe_transplant_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # Check if we got any results
    if len(results_df) == 0:
        print("\n" + "="*80)
        print("ERROR: No probes were successfully trained!")
        print("="*80)
        print("\nPossible issues:")
        print("  - Not enough samples collected")
        print("  - Activation extraction failed")
        print("  - Not enough diversity in labels")
        return
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal probe configurations: {len(results_df)}")
    print(f"\nTest Accuracy Statistics:")
    print(results_df['test_acc'].describe())
    
    print(f"\nBest performing configurations:")
    print(results_df.nlargest(10, 'test_acc')[['layer', 'sentence', 'test_acc', 'n_samples']])
    
    print(f"\nWorst performing configurations:")
    print(results_df.nsmallest(10, 'test_acc')[['layer', 'sentence', 'test_acc', 'n_samples']])
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_probe_results(results_df, 'linear_probe_transplant_heatmap.png')
    plot_comparison_charts(results_df, 'linear_probe_transplant')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - linear_probe_transplant_results.csv")
    print("  - linear_probe_transplant_heatmap.png")
    print("  - linear_probe_transplant_layers.png")
    print("  - linear_probe_transplant_sentences.png")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS: THOUGHT ANCHORS IN TRANSPLANTED CoT")
    print("="*80)
    
    # Find where hint becomes most decodable
    best_result = results_df.loc[results_df['test_acc'].idxmax()]
    print(f"\nHighest probe accuracy:")
    print(f"  Layer {best_result['layer']}, Sentence {best_result['sentence']}")
    print(f"  Test Accuracy: {best_result['test_acc']:.2%}")
    print(f"  N samples: {best_result['n_samples']}")
    
    # Count high-accuracy configurations
    high_acc = results_df[results_df['test_acc'] > 0.6]
    print(f"\nConfigurations with >60% accuracy: {len(high_acc)}/{len(results_df)}")
    
    if len(high_acc) > 0:
        earliest_high = high_acc.loc[high_acc['sentence'].idxmin()]
        print(f"\nEarliest high accuracy (>60%):")
        print(f"  Sentence {earliest_high['sentence']} in transplanted CoT")
        print(f"  Layer {earliest_high['layer']}, Accuracy: {earliest_high['test_acc']:.2%}")
        
        print(f"\n🔬 INTERPRETATION:")
        print(f"  The hint answer IS decodable from transplanted CoT, even though")
        print(f"  the hint was never shown! This suggests 'thought anchors' - the")
        print(f"  reasoning carries latent representations of the hint.")
    else:
        print("\n❌ No configurations achieved >60% accuracy")
        print("   Hint may not be strongly encoded in transplanted activations")
    
    # Compare to chance
    above_chance = results_df[results_df['test_acc'] > 0.35]  # Well above 25% chance
    print(f"\nConfigurations above chance (>35%): {len(above_chance)}/{len(results_df)}")


if __name__ == "__main__":
    main()

