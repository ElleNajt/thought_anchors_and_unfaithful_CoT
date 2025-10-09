"""
Test configuration for the splice experiment with matching data.
This uses test delta analysis for problems that have hinted CoTs.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SPLICE_EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_problems"

# Input files - USE TEST DATA
DELTA_ANALYSIS_CSV = SPLICE_EXPERIMENT_DIR / "delta_analysis_test.csv"
PROFESSOR_HINTED_MMLU = DATA_DIR / "professor_hinted_mmlu.json"

# Model Configuration (local)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Inference Configuration
MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.7

# Experiment Configuration
DEFAULT_DELTA_THRESHOLD = 0.7
DEFAULT_NUM_SAMPLES = 2
DEFAULT_CONCURRENCY = 1

# Output
RESULTS_DIR = SPLICE_EXPERIMENT_DIR / "results"

