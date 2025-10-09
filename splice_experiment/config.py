"""
Configuration for the splice experiment.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SPLICE_EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CUE_P_ANALYSIS_DIR = PROJECT_ROOT / "cue_p_analysis"

# Input files
DELTA_ANALYSIS_CSV = CUE_P_ANALYSIS_DIR / "per_problem_timeseries_with_sentences_all34_delta_analysis.csv"
PROFESSOR_HINTED_MMLU = DATA_DIR / "professor_hinted_mmlu.json"

# API Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Experiment parameters
DEFAULT_DELTA_THRESHOLD = 0.5  # Only splice at sentences with delta_cue_p > this
DEFAULT_NUM_SAMPLES = 5  # Number of unhinted continuations to sample per splice point
DEFAULT_CONCURRENCY = 5  # Max concurrent API calls
TEMPERATURE = 1.0  # Sampling temperature

# Analysis parameters
MIN_PROBLEM_SENTENCES = 3  # Skip problems with fewer sentences than this

# Output
RESULTS_DIR = SPLICE_EXPERIMENT_DIR / "results"

