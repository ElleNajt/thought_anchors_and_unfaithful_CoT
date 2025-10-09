# Splice Experiment - Start Here

## What is This?

This is a **self-contained experiment** to test whether high-delta cue_p sentences act as "anchors" in chain-of-thought reasoning.

## The Core Idea

We've identified sentences that cause **massive jumps** in the probability of outputting the hinted answer (up to 0.8 increase!). This experiment tests:

**Do these sentences actually cause the model to lock onto the hinted answer?**

We test this by:
1. Taking a hinted reasoning chain
2. Right before a high-delta sentence, **switch to unhinted reasoning**
3. See if the final answer changes

## Quick Navigation

### 📖 New to this experiment?
→ Start with **[README.md](README.md)** for the experiment overview

### 🚀 Ready to run it?
→ Follow **[QUICKSTART.md](QUICKSTART.md)** for step-by-step instructions

### 🔬 Want scientific details?
→ Read **[EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md)** for hypotheses and methodology

### 📂 Need to find a specific file?
→ Check **[FILE_GUIDE.md](FILE_GUIDE.md)** for complete file documentation

## Three-Step Getting Started

```bash
# Step 1: Test your setup
python test_setup.py

# Step 2: Run a small example
./run_example.sh
# (or manually: python run_splice_experiment.py --cot-data PATH --delta-threshold 0.7 --problems 3,288,59)

# Step 3: Analyze results
python analyze_results.py results/TIMESTAMP/splice_results.json --cot-data PATH
```

## File Structure

```
splice_experiment/
├── INDEX.md                    ← You are here
├── README.md                   ← Experiment overview
├── QUICKSTART.md               ← Step-by-step guide
├── EXPERIMENT_OVERVIEW.md      ← Scientific details
├── FILE_GUIDE.md               ← Complete file reference
│
├── run_splice_experiment.py    ← Main experiment script
├── analyze_results.py          ← Analysis script
├── test_setup.py               ← Setup verification
├── run_example.sh              ← Quick test script
│
├── config.py                   ← Configuration
├── requirements.txt            ← Dependencies
└── .gitignore                  ← Git ignore rules
```

## What You Need

**Data Requirements:**
- ✅ Delta analysis CSV (already exists in parent project)
- ✅ MMLU questions (already exists in parent project)
- ⚠️ **Original CoT responses** (you need to specify the path)

**Environment:**
- ✅ Python 3.7+
- ✅ OpenRouter API key (in `.env`)
- ✅ Dependencies from `requirements.txt`

## Expected Output

After running the experiment, you'll get:

```
results/
└── YYYYMMDD_HHMMSS/
    ├── splice_results.json           # Raw experimental data
    ├── metadata.json                  # Configuration used
    ├── summary.json                   # Quick summary stats
    └── visualizations/
        ├── statistics.csv             # Detailed statistics
        ├── delta_vs_drop_scatter.png  # Does delta predict drop?
        ├── probability_comparison_bars.png
        ├── splice_similarity.png
        └── problem_*_distribution.png # Individual plots
```

## Key Insight You're Testing

**If high-delta sentences are true anchors:**
- Removing them should break the anchoring effect
- Answer distribution should shift away from hinted answer
- Shift should be proportional to the original delta size

**If they're not anchors:**
- Removing them won't matter much
- Model will still output similar distributions
- No relationship between delta size and effect

## Next Steps

1. **First time?** → [QUICKSTART.md](QUICKSTART.md)
2. **Want details?** → [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md)
3. **Need help?** → [FILE_GUIDE.md](FILE_GUIDE.md) has troubleshooting

## Questions?

This is a complete, self-contained experiment. All the infrastructure you need is here:
- ✅ Data loading and preprocessing
- ✅ API integration for sampling
- ✅ Statistical analysis
- ✅ Visualization generation
- ✅ Documentation and examples

Just point it at your CoT data and run!

