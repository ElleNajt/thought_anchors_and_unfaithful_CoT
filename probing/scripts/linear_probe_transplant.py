#!/usr/bin/env python3
"""
Linear Probe Analysis for TRANSPLANTED CoT

This script trains linear probes to predict the hint answer from model activations
in TRANSPLANTED examples - where hint-influenced CoT is placed after a no-hint prompt.

Key Question: Does the transplanted CoT carry latent representations of the hint
that was used to generate it, even when the hint is never shown?
"""

import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the demo folder to path (go up two levels from scripts/ to root, then into demo folder)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
demo_dir = os.path.join(root_dir, "CoT_Faithfulness_demo")
sys.path.insert(0, demo_dir)
from load_CoT import load_all_problems


def load_cue_p_data(csv_path):
    """
    Load cue_p values from the faithfulness CSV file.

    Returns:
        dict: {(pn, sentence_num): cue_p_value}
    """
    df = pd.read_csv(csv_path)
    cue_p_dict = {}
    for _, row in df.iterrows():
        key = (row["pn"], row["sentence_num"])
        cue_p_dict[key] = row["cue_p"]
    return cue_p_dict


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
    sentences = [s.strip() + " " for s in cot_text.split(".") if s.strip()]

    end_sentence = start_sentence + num_sentences
    extracted = ".".join(sentences[start_sentence:end_sentence])

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
        num_sentences=num_sentences,
    )

    # Build: no-hint question + transplanted section
    transplanted_prompt = problem_data["question"] + transplant_section + " "

    return transplanted_prompt, num_sentences


def get_layer_activations_chunked(model, tokens, chunk_size=12):
    """
    Get activations from all layers by processing them in chunks.
    This balances efficiency (fewer forward passes than N) with memory constraints.

    Args:
        model: The language model
        tokens: Input token tensor
        chunk_size: Number of layers to process per forward pass

    Returns:
        List of tensors, one per layer [batch, seq_len, hidden_dim]
    """
    n_layers = model.config.num_hidden_layers
    all_activations = [None] * n_layers

    # Process layers in chunks
    for chunk_start in range(0, n_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_layers)

        activations = {}
        hooks = []

        def make_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().cpu()
                else:
                    activations[layer_idx] = output.detach().cpu()

            return hook_fn

        # Register hooks only for this chunk of layers
        for layer_idx in range(chunk_start, chunk_end):
            hook = model.model.layers[layer_idx].register_forward_hook(
                make_hook_fn(layer_idx)
            )
            hooks.append(hook)

        try:
            with torch.no_grad():
                model(tokens)

            # Store activations from this chunk
            for layer_idx in range(chunk_start, chunk_end):
                all_activations[layer_idx] = activations[layer_idx]
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            activations.clear()

    return all_activations


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
    transplanted_section = transplanted_prompt[len(question_only) :]

    # Split transplanted section by sentences
    sentences = [s.strip() for s in transplanted_section.split(".") if s.strip()]

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
    cue_p_dict=None,
    start_sentence=3,
    num_sentences=5,
    max_problems=None,
):
    """
    Collect activations from TRANSPLANTED prompts across all layers and sentence positions.
    Uses a single forward pass per problem to extract all layer activations.

    Args:
        cue_p_dict: Optional dict {(pn, sentence_num): cue_p_value} for regression targets

    Returns:
        dict with structure:
            {
                layer_idx: {
                    sentence_num: {
                        'activations': list of activation vectors,
                        'labels': list of hint answers (0=A, 1=B, 2=C, 3=D),
                        'pns': list of problem numbers,
                        'cue_ps': list of cue_p values (if cue_p_dict provided)
                    }
                }
            }
    """
    n_layers = model.config.num_hidden_layers

    # Initialize data structure
    if cue_p_dict is not None:
        data = defaultdict(
            lambda: defaultdict(
                lambda: {"activations": [], "labels": [], "pns": [], "cue_ps": []}
            )
        )
    else:
        data = defaultdict(
            lambda: defaultdict(lambda: {"activations": [], "labels": [], "pns": []})
        )

    if max_problems:
        problems = problems[:max_problems]

    print(
        f"\nCollecting activations from {len(problems)} TRANSPLANTED examples across {n_layers} layers..."
    )
    print(
        f"Transplanting sentences {start_sentence} to {start_sentence + num_sentences} from hint-influenced CoT"
    )

    for problem in tqdm(problems, desc="Processing transplanted prompts"):
        try:
            # Build transplanted prompt
            transplanted_prompt, n_sentences = build_transplanted_prompt(
                problem, start_sentence=start_sentence, num_sentences=num_sentences
            )

            cue_answer = problem["cue_answer"]
            label = ord(cue_answer) - ord("A")

            # Tokenize transplanted prompt
            tokens = tokenizer.encode(transplanted_prompt, return_tensors="pt").to(
                device
            )

            # Get sentence positions in the transplanted section
            sentence_positions = extract_sentence_positions_in_prompt(
                transplanted_prompt, problem["question"], tokenizer
            )

            # Get all layer activations using chunked approach (48 layers / 12 per chunk = 4 forward passes)
            all_layer_acts = get_layer_activations_chunked(model, tokens, chunk_size=12)

            # Extract activations for each layer and sentence position
            for layer_idx in range(n_layers):
                layer_acts = all_layer_acts[layer_idx]

                # For each sentence position in the transplanted section
                for sent_num, token_pos in sentence_positions:
                    # Get activation at this position
                    if token_pos < layer_acts.shape[1]:
                        act_vector = layer_acts[0, token_pos, :].numpy()

                        data[layer_idx][sent_num]["activations"].append(act_vector)
                        data[layer_idx][sent_num]["labels"].append(label)
                        data[layer_idx][sent_num]["pns"].append(problem["pn"])

                        # Add cue_p if available
                        if cue_p_dict is not None:
                            key = (problem["pn"], sent_num)
                            cue_p_value = cue_p_dict.get(key, np.nan)
                            data[layer_idx][sent_num]["cue_ps"].append(cue_p_value)

            # Clear the layer activations and CUDA cache after each problem
            del all_layer_acts
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing problem {problem.get('pn', 'unknown')}: {e}")
            continue

    return data


def train_probes_and_evaluate(data, test_size=0.2, random_state=42, loaded_split=None):
    """
    Train linear probes for each layer and sentence position, evaluate on test set.

    Args:
        loaded_split: Optional tuple of (train_pns_set, test_pns_set) to reuse an existing split

    Returns:
        results_df: DataFrame with columns: layer, sentence, train_acc, test_acc, n_samples
        predictions_df: DataFrame with per-example predictions: layer, sentence, pn, true_label, predicted_label, correct, split
        splits_df: DataFrame with train/test split info: layer, sentence, pn, split
    """
    results = []
    predictions = []
    splits = []

    print("\nTraining linear probes on transplanted examples...")
    
    # CRITICAL FIX: Determine global train/test split at problem level FIRST
    # This prevents data leakage where a problem could be in train for some 
    # layer/sentence combos and test for others
    
    if loaded_split is not None:
        train_pns_set, test_pns_set = loaded_split
        print(f"Using loaded split: {len(train_pns_set)} train problems, {len(test_pns_set)} test problems")
    else:
        print("Determining global train/test split at problem level...")
        all_pns = set()
        all_labels = {}
        for layer_idx in data.keys():
            for sent_num in data[layer_idx].keys():
                for pn, label in zip(data[layer_idx][sent_num]["pns"], 
                                    data[layer_idx][sent_num]["labels"]):
                    all_pns.add(pn)
                    all_labels[pn] = label
        
        all_pns = sorted(list(all_pns))
        labels_for_split = np.array([all_labels[pn] for pn in all_pns])
        
        # Do stratified split at problem level
        train_pns, test_pns = train_test_split(
            all_pns, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels_for_split
        )
        train_pns_set = set(train_pns)
        test_pns_set = set(test_pns)
        print(f"Global split: {len(train_pns)} train problems, {len(test_pns)} test problems")

    for layer_idx in tqdm(sorted(data.keys()), desc="Layers"):
        for sent_num in sorted(data[layer_idx].keys()):
            layer_sent_data = data[layer_idx][sent_num]

            X = np.array(layer_sent_data["activations"])
            y = np.array(layer_sent_data["labels"])
            pns = np.array(layer_sent_data["pns"])

            # Need at least 10 samples and at least 2 classes
            if len(X) < 10 or len(np.unique(y)) < 2:
                continue

            # Check if we can do stratified split (need at least 2 samples per class)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = class_counts.min()

            # Skip if any class has only 1 sample (can't stratify)
            if min_class_count < 2:
                continue

            try:
                # Use global train/test split instead of per-layer/sentence split
                train_mask = np.array([pn in train_pns_set for pn in pns])
                test_mask = np.array([pn in test_pns_set for pn in pns])
                
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                pns_train = pns[train_mask]
                pns_test = pns[test_mask]
                
                # Skip if either split is too small
                if len(X_train) < 2 or len(X_test) < 2:
                    continue
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                # Store which problems are in train vs test for this layer/sentence
                for pn in pns_train:
                    splits.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "split": "train",
                        }
                    )
                for pn in pns_test:
                    splits.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "split": "test",
                        }
                    )

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train logistic regression probe
                probe = LogisticRegression(max_iter=1000, random_state=random_state)
                probe.fit(X_train_scaled, y_train)

                # Get predictions
                y_train_pred = probe.predict(X_train_scaled)
                y_test_pred = probe.predict(X_test_scaled)

                # Evaluate
                train_acc = probe.score(X_train_scaled, y_train)
                test_acc = probe.score(X_test_scaled, y_test)

                results.append(
                    {
                        "layer": layer_idx,
                        "sentence": sent_num,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "n_samples": len(X),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                )

                # Store per-example predictions for train set
                for pn, true, pred in zip(pns_train, y_train, y_train_pred):
                    predictions.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "true_label": int(true),
                            "predicted_label": int(pred),
                            "correct": int(true == pred),
                            "split": "train",
                        }
                    )

                # Store per-example predictions for test set
                for pn, true, pred in zip(pns_test, y_test, y_test_pred):
                    predictions.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "true_label": int(true),
                            "predicted_label": int(pred),
                            "correct": int(true == pred),
                            "split": "test",
                        }
                    )

            except Exception as e:
                print(
                    f"\nError training probe for layer {layer_idx}, sentence {sent_num}: {e}"
                )
                continue

    return pd.DataFrame(results), pd.DataFrame(predictions), pd.DataFrame(splits)


def train_regression_probes_and_evaluate(data, test_size=0.2, random_state=42, loaded_split=None):
    """
    Train linear regression probes to predict cue_p for each layer and sentence position.

    Args:
        loaded_split: Optional tuple of (train_pns_set, test_pns_set) to reuse an existing split

    Returns:
        results_df: DataFrame with columns: layer, sentence, train_mse, test_mse, train_r2, test_r2, n_samples
        predictions_df: DataFrame with per-example predictions: layer, sentence, pn, true_cue_p, predicted_cue_p, split
        splits_df: DataFrame with train/test split info: layer, sentence, pn, split
    """
    results = []
    predictions = []
    splits = []

    print("\nTraining linear regression probes on transplanted examples...")
    
    # CRITICAL FIX: Determine global train/test split at problem level FIRST
    # This prevents data leakage where a problem could be in train for some 
    # layer/sentence combos and test for others
    
    if loaded_split is not None:
        train_pns_set, test_pns_set = loaded_split
        print(f"Using loaded split: {len(train_pns_set)} train problems, {len(test_pns_set)} test problems")
    else:
        print("Determining global train/test split at problem level...")
        all_pns = set()
        for layer_idx in range(len(data)):
            for sent_num in data[layer_idx].keys():
                all_pns.update(data[layer_idx][sent_num]["pns"])
        
        all_pns = sorted(list(all_pns))
        
        # Do random split at problem level (no stratification needed for regression)
        train_pns, test_pns = train_test_split(
            all_pns, 
            test_size=test_size, 
            random_state=random_state
        )
        train_pns_set = set(train_pns)
        test_pns_set = set(test_pns)
        print(f"Global split: {len(train_pns)} train problems, {len(test_pns)} test problems")

    # Iterate through all layer-sentence combinations
    for layer_idx in tqdm(range(len(data)), desc="Layers"):
        for sent_num in sorted(data[layer_idx].keys()):
            try:
                # Get data for this layer and sentence
                activations = np.array(data[layer_idx][sent_num]["activations"])
                cue_ps = np.array(data[layer_idx][sent_num]["cue_ps"])
                pns = np.array(data[layer_idx][sent_num]["pns"])

                # Skip if not enough samples or if we have NaN values
                valid_mask = ~np.isnan(cue_ps)
                if valid_mask.sum() < 10:  # Need at least 10 valid samples
                    continue

                # Filter out NaN values
                activations = activations[valid_mask]
                cue_ps = cue_ps[valid_mask]
                pns = pns[valid_mask]

                # Use global train/test split instead of per-layer/sentence split
                train_mask = np.array([pn in train_pns_set for pn in pns])
                test_mask = np.array([pn in test_pns_set for pn in pns])
                
                X_train = activations[train_mask]
                X_test = activations[test_mask]
                y_train = cue_ps[train_mask]
                y_test = cue_ps[test_mask]
                pns_train = pns[train_mask]
                pns_test = pns[test_mask]
                
                # Skip if either split is too small
                if len(X_train) < 2 or len(X_test) < 2:
                    continue

                # Store which problems are in train vs test for this layer/sentence
                for pn in pns_train:
                    splits.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "split": "train",
                        }
                    )
                for pn in pns_test:
                    splits.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "split": "test",
                        }
                    )

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train linear regression probe
                probe = LinearRegression()
                probe.fit(X_train_scaled, y_train)

                # Get predictions
                y_train_pred = probe.predict(X_train_scaled)
                y_test_pred = probe.predict(X_test_scaled)

                # Evaluate
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                results.append(
                    {
                        "layer": layer_idx,
                        "sentence": sent_num,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "n_samples": len(activations),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                )

                # Store per-example predictions for train set
                for pn, true, pred in zip(pns_train, y_train, y_train_pred):
                    predictions.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "true_cue_p": float(true),
                            "predicted_cue_p": float(pred),
                            "error": float(abs(true - pred)),
                            "split": "train",
                        }
                    )

                # Store per-example predictions for test set
                for pn, true, pred in zip(pns_test, y_test, y_test_pred):
                    predictions.append(
                        {
                            "layer": layer_idx,
                            "sentence": sent_num,
                            "pn": pn,
                            "true_cue_p": float(true),
                            "predicted_cue_p": float(pred),
                            "error": float(abs(true - pred)),
                            "split": "test",
                        }
                    )

            except Exception as e:
                print(
                    f"\nError training regression probe for layer {layer_idx}, sentence {sent_num}: {e}"
                )
                continue

    return pd.DataFrame(results), pd.DataFrame(predictions), pd.DataFrame(splits)


def plot_probe_results(results_df, save_path="linear_probe_transplant_heatmap.png"):
    """
    Create a heatmap showing probe accuracy across layers and sentences.
    """
    # Pivot to create heatmap data
    heatmap_data = results_df.pivot(
        index="layer", columns="sentence", values="test_acc"
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Test accuracy heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.25,
        vmax=1.0,
        center=0.5,
        ax=ax1,
        cbar_kws={"label": "Test Accuracy"},
    )
    ax1.set_xlabel("Sentence Position in Transplanted CoT", fontsize=12)
    ax1.set_ylabel("Layer", fontsize=12)
    ax1.set_title(
        "Linear Probe Test Accuracy: Predicting Hint Answer\nfrom TRANSPLANTED CoT Activations (Hint Never Shown)",
        fontsize=14,
        fontweight="bold",
    )

    # Sample count heatmap
    sample_data = results_df.pivot(
        index="layer", columns="sentence", values="n_samples"
    )
    sns.heatmap(
        sample_data,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax2,
        cbar_kws={"label": "Number of Samples"},
    )
    ax2.set_xlabel("Sentence Position in Transplanted CoT", fontsize=12)
    ax2.set_ylabel("Layer", fontsize=12)
    ax2.set_title(
        "Sample Counts per Layer and Sentence", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    return fig


def plot_comparison_charts(results_df, save_path_prefix="linear_probe_transplant"):
    """
    Create comparison plots showing layer and sentence trends.
    """
    # Layer trends
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    layer_avg = (
        results_df.groupby("layer")["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    ax1.plot(
        layer_avg["layer"], layer_avg["mean"], marker="o", linewidth=2, markersize=8
    )
    ax1.fill_between(
        layer_avg["layer"],
        layer_avg["mean"] - layer_avg["std"],
        layer_avg["mean"] + layer_avg["std"],
        alpha=0.3,
    )
    ax1.axhline(y=0.25, color="r", linestyle="--", label="Chance (25%)")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Test Accuracy (averaged across sentences)", fontsize=12)
    ax1.set_title(
        "Linear Probe Accuracy by Layer\nTransplanted CoT (Hint Never Shown)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_layers.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path_prefix}_layers.png")

    # Sentence trends
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sent_avg = (
        results_df.groupby("sentence")["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    ax2.plot(
        sent_avg["sentence"], sent_avg["mean"], marker="o", linewidth=2, markersize=8
    )
    ax2.fill_between(
        sent_avg["sentence"],
        sent_avg["mean"] - sent_avg["std"],
        sent_avg["mean"] + sent_avg["std"],
        alpha=0.3,
    )
    ax2.axhline(y=0.25, color="r", linestyle="--", label="Chance (25%)")
    ax2.set_xlabel("Sentence Position in Transplanted CoT", fontsize=12)
    ax2.set_ylabel("Test Accuracy (averaged across layers)", fontsize=12)
    ax2.set_title(
        "Linear Probe Accuracy by Sentence Position\nTransplanted CoT (Hint Never Shown)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_sentences.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path_prefix}_sentences.png")

    return fig1, fig2


def main():
    """
    Main function to run linear probe analysis on transplanted examples.
    """
    print("=" * 80)
    print("LINEAR PROBE ANALYSIS: TRANSPLANTED CoT")
    print("Testing if hint is decodable when hint-influenced CoT is transplanted")
    print("=" * 80)

    # Configuration
    MAX_PROBLEMS = 30  # Use subset for faster analysis
    START_SENTENCE = 0  # Which sentence to start transplanting from
    NUM_SENTENCES = 100  # Use large number to get entire CoT
    # How many sentences to transplant
    
    # Optional: Specify custom test problems (set to None to load from file or auto-generate)
    CUSTOM_TEST_PROBLEMS = [700, 119, 26, 59, 715]
    
    # Optional: Load existing train/test split from a previous run
    # Set to None to create a new split, or provide path to reuse an existing split
    # LOAD_SPLIT_FROM = "probing/scripts/results/7e4afeb_historical/data/linear_probe_transplant_regression_splits.csv"
    LOAD_SPLIT_FROM = None  # Using custom test problems instead

    # Get git hash and timestamp for output directory
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode("utf-8")
            .strip()
        )
    except:
        git_hash = "unknown"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{git_hash}_{timestamp}"

    # Output directories with timestamped subdirectory
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", run_id)
    DATA_DIR = os.path.join(RESULTS_DIR, "data")
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")

    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"\nResults will be saved to: {RESULTS_DIR}")

    # Load model
    print("\nLoading model...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model has {model.config.num_hidden_layers} layers")

    # Load problems
    print("\nLoading problem data...")
    original_dir = os.getcwd()
    # Demo dir is already set up in the imports section
    os.chdir(demo_dir)
    try:
        data = load_all_problems()
        print(f"Loaded {len(data)} problems")

        # Load cue_p data for regression probes
        cue_p_csv = "faith_counterfactual_qwen-14b_demo.csv"
        print(f"\nLoading cue_p data from {cue_p_csv}...")
        cue_p_dict = load_cue_p_data(cue_p_csv)
        print(f"Loaded cue_p values for {len(cue_p_dict)} (problem, sentence) pairs")
    finally:
        os.chdir(original_dir)

    # Collect activations from transplanted examples
    activation_data = collect_activations_for_transplants(
        data,
        model,
        tokenizer,
        device,
        cue_p_dict=cue_p_dict,
        start_sentence=START_SENTENCE,
        num_sentences=NUM_SENTENCES,
        max_problems=MAX_PROBLEMS,
    )

    # Load existing split if specified, or use custom test problems
    loaded_split = None
    if CUSTOM_TEST_PROBLEMS is not None:
        print(f"\nUsing custom test problems: {CUSTOM_TEST_PROBLEMS}")
        test_pns = set(CUSTOM_TEST_PROBLEMS)
        # Get all available problem numbers from the data
        all_pns = set(p['pn'] for p in data)
        train_pns = all_pns - test_pns
        loaded_split = (train_pns, test_pns)
        print(f"Custom split: {len(train_pns)} train problems, {len(test_pns)} test problems")
        print(f"Test problems: {sorted(test_pns)}")
    elif LOAD_SPLIT_FROM is not None:
        print(f"\nLoading train/test split from: {LOAD_SPLIT_FROM}")
        split_df = pd.read_csv(LOAD_SPLIT_FROM)
        # Extract unique problem numbers for train and test
        train_pns = set(split_df[split_df['split'] == 'train']['pn'].unique())
        test_pns = set(split_df[split_df['split'] == 'test']['pn'].unique())
        loaded_split = (train_pns, test_pns)
        print(f"Loaded split: {len(train_pns)} train problems, {len(test_pns)} test problems")
        print(f"Test problems: {sorted(test_pns)}")

    # Train classification probes and evaluate
    print("\n" + "=" * 80)
    print("CLASSIFICATION PROBES: Predicting Hint Answer")
    print("=" * 80)
    results_df, predictions_df, splits_df = train_probes_and_evaluate(activation_data, loaded_split=loaded_split)

    # Save classification results
    results_csv = os.path.join(DATA_DIR, "linear_probe_transplant_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nClassification results saved to: {results_csv}")

    # Save train and test predictions separately to avoid confusion
    predictions_train_csv = os.path.join(DATA_DIR, "linear_probe_transplant_predictions_train.csv")
    predictions_test_csv = os.path.join(DATA_DIR, "linear_probe_transplant_predictions_test.csv")
    
    predictions_df[predictions_df['split'] == 'train'].to_csv(predictions_train_csv, index=False)
    predictions_df[predictions_df['split'] == 'test'].to_csv(predictions_test_csv, index=False)
    
    print(f"Classification predictions (train) saved to: {predictions_train_csv}")
    print(f"Classification predictions (test) saved to: {predictions_test_csv}")

    splits_csv = os.path.join(DATA_DIR, "linear_probe_transplant_splits.csv")
    splits_df.to_csv(splits_csv, index=False)
    print(f"Classification train/test splits saved to: {splits_csv}")

    # Train regression probes for cue_p prediction
    print("\n" + "=" * 80)
    print("REGRESSION PROBES: Predicting cue_p")
    print("=" * 80)
    regression_results_df, regression_predictions_df, regression_splits_df = (
        train_regression_probes_and_evaluate(activation_data, loaded_split=loaded_split)
    )

    # Save regression results
    regression_results_csv = os.path.join(
        DATA_DIR, "linear_probe_transplant_regression_results.csv"
    )
    regression_results_df.to_csv(regression_results_csv, index=False)
    print(f"\nRegression results saved to: {regression_results_csv}")

    # Define file paths for regression predictions
    regression_predictions_train_csv = os.path.join(
        DATA_DIR, "linear_probe_transplant_regression_predictions_train.csv"
    )
    regression_predictions_test_csv = os.path.join(
        DATA_DIR, "linear_probe_transplant_regression_predictions_test.csv"
    )
    
    # Check if we have any regression predictions before saving
    if len(regression_predictions_df) > 0:
        # Save train and test predictions separately to avoid confusion
        regression_predictions_df[regression_predictions_df['split'] == 'train'].to_csv(
            regression_predictions_train_csv, index=False
        )
        regression_predictions_df[regression_predictions_df['split'] == 'test'].to_csv(
            regression_predictions_test_csv, index=False
        )
        
        print(f"Regression predictions (train) saved to: {regression_predictions_train_csv}")
        print(f"Regression predictions (test) saved to: {regression_predictions_test_csv}")
    else:
        print("WARNING: No regression predictions were generated!")

    regression_splits_csv = os.path.join(
        DATA_DIR, "linear_probe_transplant_regression_splits.csv"
    )
    regression_splits_df.to_csv(regression_splits_csv, index=False)
    print(f"Regression train/test splits saved to: {regression_splits_csv}")

    # Check if we got any results
    if len(results_df) == 0:
        print("\n" + "=" * 80)
        print("ERROR: No probes were successfully trained!")
        print("=" * 80)
        print("\nPossible issues:")
        print("  - Not enough samples collected")
        print("  - Activation extraction failed")
        print("  - Not enough diversity in labels")
        return

    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal probe configurations: {len(results_df)}")
    print(f"\nTest Accuracy Statistics:")
    print(results_df["test_acc"].describe())

    print(f"\nBest performing configurations:")
    print(
        results_df.nlargest(10, "test_acc")[
            ["layer", "sentence", "test_acc", "n_samples"]
        ]
    )

    print(f"\nWorst performing configurations:")
    print(
        results_df.nsmallest(10, "test_acc")[
            ["layer", "sentence", "test_acc", "n_samples"]
        ]
    )

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    plot_probe_results(
        results_df, os.path.join(FIG_DIR, "linear_probe_transplant_heatmap.png")
    )
    plot_comparison_charts(results_df, os.path.join(FIG_DIR, "linear_probe_transplant"))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {results_csv}")
    print(f"  - {predictions_train_csv}")
    print(f"  - {predictions_test_csv}")
    print(f"  - {splits_csv}")
    print(f"  - {regression_results_csv}")
    if len(regression_predictions_df) > 0:
        print(f"  - {regression_predictions_train_csv}")
        print(f"  - {regression_predictions_test_csv}")
    print(f"  - {regression_splits_csv}")
    print(f"  - {os.path.join(FIG_DIR, 'linear_probe_transplant_heatmap.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'linear_probe_transplant_layers.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'linear_probe_transplant_sentences.png')}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS: THOUGHT ANCHORS IN TRANSPLANTED CoT")
    print("=" * 80)

    # Find where hint becomes most decodable
    best_result = results_df.loc[results_df["test_acc"].idxmax()]
    print(f"\nHighest probe accuracy:")
    print(f"  Layer {best_result['layer']}, Sentence {best_result['sentence']}")
    print(f"  Test Accuracy: {best_result['test_acc']:.2%}")
    print(f"  N samples: {best_result['n_samples']}")

    # Count high-accuracy configurations
    high_acc = results_df[results_df["test_acc"] > 0.6]
    print(f"\nConfigurations with >60% accuracy: {len(high_acc)}/{len(results_df)}")

    if len(high_acc) > 0:
        earliest_high = high_acc.loc[high_acc["sentence"].idxmin()]
        print(f"\nEarliest high accuracy (>60%):")
        print(f"  Sentence {earliest_high['sentence']} in transplanted CoT")
        print(
            f"  Layer {earliest_high['layer']}, Accuracy: {earliest_high['test_acc']:.2%}"
        )

        print(f"\n🔬 INTERPRETATION:")
        print(f"  The hint answer IS decodable from transplanted CoT, even though")
        print(f"  the hint was never shown! This suggests 'thought anchors' - the")
        print(f"  reasoning carries latent representations of the hint.")
    else:
        print("\n❌ No configurations achieved >60% accuracy")
        print("   Hint may not be strongly encoded in transplanted activations")

    # Compare to chance
    above_chance = results_df[results_df["test_acc"] > 0.35]  # Well above 25% chance
    print(
        f"\nConfigurations above chance (>35%): {len(above_chance)}/{len(results_df)}"
    )


if __name__ == "__main__":
    main()
