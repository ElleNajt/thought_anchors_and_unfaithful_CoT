#!/usr/bin/env python3
"""
Test that the setup is working correctly before running the full experiment.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from dotenv import load_dotenv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")
    
    import config
    
    files_to_check = [
        ("Delta analysis CSV", config.DELTA_ANALYSIS_CSV),
        ("MMLU dataset", config.PROFESSOR_HINTED_MMLU),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        if path.exists():
            print(f"✓ {name} exists: {path}")
        else:
            print(f"✗ {name} NOT FOUND: {path}")
            all_exist = False
    
    return all_exist


def test_model_availability():
    """Test that PyTorch and model libraries are available."""
    print("\nTesting model availability...")
    
    try:
        import torch
        import transformers
        
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"✓ Transformers {transformers.__version__} installed")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (will use CPU - slower)")
        
        return True
    except Exception as e:
        print(f"✗ Error checking model availability: {e}")
        return False


def test_load_data():
    """Test loading the delta analysis data."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        import config
        
        df = pd.read_csv(config.DELTA_ANALYSIS_CSV)
        print(f"✓ Loaded delta analysis: {len(df)} rows")
        
        import json
        with open(config.PROFESSOR_HINTED_MMLU, 'r') as f:
            questions = json.load(f)
        print(f"✓ Loaded questions: {len(questions)} questions")
        
        # Show some high delta sentences
        high_delta = df[df['delta_cue_p'] >= 0.5].sort_values('delta_cue_p', ascending=False)
        print(f"\nFound {len(high_delta)} sentences with delta_cue_p >= 0.5")
        print("\nTop 5:")
        for _, row in high_delta.head(5).iterrows():
            print(f"  Problem {row['problem_number']}, Sentence {row['sentence_number']}: delta={row['delta_cue_p']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("SPLICE EXPERIMENT SETUP TEST")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data files", test_data_files()))
    results.append(("Model availability", test_model_availability()))
    results.append(("Data loading", test_load_data()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the experiment.")
        print("\nNext steps:")
        print("1. Find your CoT data file:")
        print("   ls -la ../data/*/cot_responses.json")
        print("\n2. Run a test experiment:")
        print("   python run_splice_experiment.py \\")
        print("     --cot-data ../data/YOUR_DATA/cot_responses.json \\")
        print("     --delta-threshold 0.7 \\")
        print("     --num-samples 2 \\")
        print("     --problems 3,288")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

