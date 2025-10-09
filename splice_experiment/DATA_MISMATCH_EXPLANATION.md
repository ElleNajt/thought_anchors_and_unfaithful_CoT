# Data Mismatch Issue and Solution

## The Problem

You encountered a `ZeroDivisionError` because of a **data mismatch** between two datasets:

### Dataset 1: Hinted CoT Responses
- **File**: `../data_problems/20251006_142931_deepseek_r1/cot_responses.json`
- **Contains**: Only 2 problems with hinted CoTs
- **Question IDs**: 10476 and 1824  
- **Array indices**: 0 and 1 (in `professor_hinted_mmlu.json`)

### Dataset 2: Delta Analysis
- **File**: `../cue_p_analysis/per_problem_timeseries_with_sentences_all34_delta_analysis.csv`
- **Contains**: 34 problems with sentence-level delta_cue_p analysis
- **Problem numbers**: 3, 18, 26, 37, 59, 62, 68, ... (does NOT include 0 or 1)

### Why the Original Example Failed

The `run_example.sh` script tried to process problems **3, 288, and 59** which:
- ✅ ARE in the delta analysis CSV
- ❌ DO NOT have hinted CoT data

The experiment needs BOTH:
1. Delta analysis (to identify high-delta sentences)
2. Hinted CoT data (to know what to splice)

Since problems 3, 288, and 59 had no hinted CoTs, the results list stayed empty, causing division by zero in the summary generation.

## The Solution

I've created two fixes:

### Fix 1: Handle Empty Results (Already Applied)
Updated `generate_summary()` in `run_splice_experiment.py` to handle empty results gracefully without crashing.

### Fix 2: Create Test Data with Matching Problems

Created test files that work with your existing hinted CoT data:

1. **`delta_analysis_test.csv`** - Mock delta analysis for problems 0 and 1
2. **`config_test.py`** - Configuration that uses the test delta file
3. **`run_test.sh`** - Simple script that runs with matching data

## How to Run a Working Test

```bash
cd /root/thought_anchors_and_unfaithful_CoT/splice_experiment
./run_test.sh
```

This will:
1. Auto-generate `delta_analysis_test.csv` with mock data for problem 0
2. Temporarily use test configuration
3. Run the experiment on problem 0 (which HAS hinted CoT data)
4. Restore original configuration

## Long-Term Solutions

To run the experiment on the 34 problems in your delta analysis, you have two options:

### Option A: Generate Hinted CoTs for Delta Analysis Problems
Run your CoT generation pipeline on problems 3, 18, 26, 37, 59, etc. to create hinted CoT samples for them.

### Option B: Generate Delta Analysis for Problems 0 and 1
Run the cue_p analysis on problems 0 and 1 (question IDs 10476 and 1824) to create proper delta analysis data.

## Quick Reference

- **Working test script**: `./run_test.sh`
- **Original example**: `./run_example.sh` (won't work due to data mismatch)
- **Test config**: `config_test.py`
- **Production config**: `config.py`
- **Generated test data**: `delta_analysis_test.csv`

## Data Requirements Checklist

For the splice experiment to work, you need:

- [ ] Delta analysis CSV with sentence-level delta_cue_p values
- [ ] Hinted CoT responses (in cot_responses.json format)
- [ ] **SAME problem numbers** in both datasets
- [ ] Questions file (professor_hinted_mmlu.json)

The test setup satisfies all requirements for problem 0.

