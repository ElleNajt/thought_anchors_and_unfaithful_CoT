#!/usr/bin/env python3
"""
Splice Experiment: Test if high delta_cue_p sentences act as anchors.

This script takes hinted CoTs, identifies high delta_cue_p sentences,
and replaces them with unhinted continuations to test if they're critical anchors.
"""

import json
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from dotenv import load_dotenv
import config

load_dotenv(config.PROJECT_ROOT / ".env")


class SpliceExperiment:
    """Run the splice experiment."""
    
    def __init__(self, delta_threshold: float, num_samples: int, concurrency: int):
        self.delta_threshold = delta_threshold
        self.num_samples = num_samples
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # Initialize OpenRouter client
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )
        
        # Load data
        self.delta_df = pd.read_csv(config.DELTA_ANALYSIS_CSV)
        with open(config.PROFESSOR_HINTED_MMLU, 'r') as f:
            self.questions = json.load(f)
        
        # We need to map problem_number to question data
        # The delta CSV uses problem_number which seems to be an index
        # We'll need the original CoT data to get the actual hinted CoTs
        self.hinted_cots_by_problem = {}
        
        print(f"Loaded {len(self.delta_df)} sentence entries from delta analysis")
        print(f"Loaded {len(self.questions)} questions")
    
    def load_hinted_cots(self, cot_data_path: Path):
        """Load the original hinted CoTs from a previous run."""
        with open(cot_data_path, 'r') as f:
            cot_data = json.load(f)
        
        # Map by question_id (which should correspond to problem_number)
        for item in cot_data:
            # We'll use the first hinted sample as our reference
            if item['hinted_samples']:
                problem_num = item.get('question_id')  # or derive from index
                self.hinted_cots_by_problem[problem_num] = {
                    'cot': item['hinted_samples'][0]['cot'],
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'hinted_answer': item['hinted_answer']
                }
        
        print(f"Loaded {len(self.hinted_cots_by_problem)} hinted CoTs")
    
    def split_cot_into_sentences(self, cot_text: str) -> List[str]:
        """
        Split a CoT into sentences.
        
        This is a simple split by common sentence endings.
        For better accuracy, we'd need the exact splitting logic used in the analysis.
        """
        import re
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', cot_text)
        return [s.strip() for s in sentences if s.strip()]
    
    def find_high_delta_sentences(self, problem_numbers: List[int] = None) -> pd.DataFrame:
        """Find sentences with high delta_cue_p."""
        high_delta = self.delta_df[self.delta_df['delta_cue_p'] >= self.delta_threshold].copy()
        
        if problem_numbers:
            high_delta = high_delta[high_delta['problem_number'].isin(problem_numbers)]
        
        # Sort by delta_cue_p descending
        high_delta = high_delta.sort_values('delta_cue_p', ascending=False)
        
        print(f"\nFound {len(high_delta)} sentences with delta_cue_p >= {self.delta_threshold}")
        print("\nTop 10 high delta sentences:")
        for _, row in high_delta.head(10).iterrows():
            print(f"  Problem {row['problem_number']}, Sentence {row['sentence_number']}: "
                  f"delta={row['delta_cue_p']:.3f}")
        
        return high_delta
    
    async def generate_continuation(self, prefix: str, base_prompt: str) -> Dict[str, Any]:
        """
        Generate an unhinted continuation from a given prefix.
        
        Args:
            prefix: The hinted CoT prefix (up to but not including the high delta sentence)
            base_prompt: The original question without hint
        
        Returns:
            Dict with continuation text and extracted answer
        """
        # Construct prompt: give context of the question and partial reasoning
        full_prompt = f"""{base_prompt}

You've started reasoning about this question, and here's what you've thought so far:

{prefix}

Please continue your reasoning from where you left off and complete your analysis. Then provide your final answer as a single letter (A, B, C, or D)."""

        async with self.semaphore:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=config.MODEL_NAME,
                        max_tokens=4096,
                        temperature=config.TEMPERATURE,
                        messages=[{"role": "user", "content": full_prompt}]
                    ),
                    timeout=180.0
                )
                continuation = response.choices[0].message.content
            except asyncio.TimeoutError:
                print(f"  ⚠ API call timed out")
                return {"continuation": None, "answer": None, "error": "timeout"}
            except Exception as e:
                print(f"  ⚠ API error: {e}")
                return {"continuation": None, "answer": None, "error": str(e)}
        
        # Extract answer
        answer = self.extract_answer(continuation)
        
        return {
            "continuation": continuation,
            "full_spliced_cot": prefix + "\n\n" + continuation,
            "answer": answer
        }
    
    def extract_answer(self, text: str) -> str:
        """Extract the letter answer (A, B, C, or D) from response text."""
        if not text:
            return None
            
        text_upper = text.strip().upper()
        
        # Look for explicit answer patterns at the end
        lines = text.strip().split('\n')
        for line in reversed(lines[-5:]):
            line_clean = line.strip()
            if len(line_clean) == 1 and line_clean in ['A', 'B', 'C', 'D']:
                return line_clean
            for letter in ['A', 'B', 'C', 'D']:
                if f"answer: {letter.lower()}" in line_clean.lower():
                    return letter
                if f"answer is {letter.lower()}" in line_clean.lower():
                    return letter
        
        # Fallback: return last isolated A, B, C, or D found
        for char in reversed(text_upper):
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return None
    
    async def process_splice_point(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single splice point: sample continuations from unhinted model.
        
        Args:
            row: A row from the delta_df with high delta_cue_p
        
        Returns:
            Dict with splice results
        """
        problem_num = row['problem_number']
        sentence_num = row['sentence_number']
        delta_cue_p = row['delta_cue_p']
        splice_sentence = row['sentence']
        
        print(f"\n=== Processing Problem {problem_num}, Sentence {sentence_num} ===")
        print(f"Delta cue_p: {delta_cue_p:.3f}")
        print(f"Splice sentence: {splice_sentence[:80]}...")
        
        # Get the hinted CoT and split it
        if problem_num not in self.hinted_cots_by_problem:
            print(f"  ⚠ No hinted CoT found for problem {problem_num}")
            return None
        
        cot_data = self.hinted_cots_by_problem[problem_num]
        hinted_cot = cot_data['cot']
        
        # Find the splice point in the CoT
        # We need to extract text up to (but not including) the splice sentence
        sentences = self.split_cot_into_sentences(hinted_cot)
        
        if sentence_num >= len(sentences):
            print(f"  ⚠ Sentence {sentence_num} not found in CoT (only {len(sentences)} sentences)")
            return None
        
        # Prefix is everything before the splice sentence
        prefix_sentences = sentences[:sentence_num]
        prefix = " ".join(prefix_sentences)
        
        if not prefix.strip():
            print(f"  ⚠ Empty prefix for sentence {sentence_num}")
            return None
        
        print(f"  Prefix length: {len(prefix)} chars")
        print(f"  Prefix: {prefix[:100]}...")
        
        # Get the base prompt (without hint)
        question_data = next((q for q in self.questions if q['id'] == problem_num), None)
        if not question_data:
            print(f"  ⚠ Question data not found for problem {problem_num}")
            return None
        
        base_prompt = question_data['base_prompt']
        
        # Sample continuations
        print(f"  Sampling {self.num_samples} continuations...")
        tasks = [
            self.generate_continuation(prefix, base_prompt)
            for _ in range(self.num_samples)
        ]
        
        continuations = await asyncio.gather(*tasks)
        
        # Calculate answer distribution
        answers = [c['answer'] for c in continuations if c['answer']]
        answer_dist = {
            'A': answers.count('A') / len(answers) if answers else 0,
            'B': answers.count('B') / len(answers) if answers else 0,
            'C': answers.count('C') / len(answers) if answers else 0,
            'D': answers.count('D') / len(answers) if answers else 0,
        }
        
        print(f"  ✓ Answer distribution: {answer_dist}")
        
        # Compile results
        result = {
            'problem_number': int(problem_num),
            'sentence_number': int(sentence_num),
            'delta_cue_p': float(delta_cue_p),
            'splice_point_sentence': splice_sentence,
            'hinted_prefix': prefix,
            'question': cot_data['question'],
            'correct_answer': cot_data['correct_answer'],
            'hinted_answer': cot_data['hinted_answer'],
            'splice_samples': [
                {
                    'sample_num': i,
                    'continuation': c['continuation'],
                    'full_spliced_cot': c['full_spliced_cot'],
                    'answer': c['answer']
                }
                for i, c in enumerate(continuations)
            ],
            'answer_distribution': answer_dist,
            'hinted_answer_probability': answer_dist.get(cot_data['hinted_answer'], 0),
            'correct_answer_probability': answer_dist.get(cot_data['correct_answer'], 0)
        }
        
        return result
    
    async def run(self, problem_numbers: List[int] = None, cot_data_path: Path = None):
        """
        Run the full splice experiment.
        
        Args:
            problem_numbers: Optional list of specific problems to process
            cot_data_path: Path to the original CoT data
        """
        # Load hinted CoTs
        if cot_data_path:
            self.load_hinted_cots(cot_data_path)
        else:
            print("\n⚠ No CoT data path provided. You need to specify --cot-data")
            print("Example: --cot-data data/20251006_142931_deepseek_r1/cot_responses.json")
            return
        
        # Find high delta sentences
        high_delta = self.find_high_delta_sentences(problem_numbers)
        
        if len(high_delta) == 0:
            print("\nNo high delta sentences found. Try lowering the threshold.")
            return
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = config.RESULTS_DIR / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nResults will be saved to: {results_dir}")
        
        # Process each splice point
        results = []
        for idx, (_, row) in enumerate(high_delta.iterrows()):
            result = await self.process_splice_point(row)
            if result:
                results.append(result)
                
                # Save incrementally
                with open(results_dir / "splice_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"  Saved result {idx + 1}/{len(high_delta)}")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": config.MODEL_NAME,
            "delta_threshold": self.delta_threshold,
            "num_samples": self.num_samples,
            "num_splice_points": len(results),
            "problem_numbers_processed": sorted(list(set(r['problem_number'] for r in results)))
        }
        
        with open(results_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate summary
        self.generate_summary(results, results_dir)
        
        print(f"\n✓ Experiment complete! Results saved to {results_dir}")
    
    def generate_summary(self, results: List[Dict], results_dir: Path):
        """Generate a summary of the experiment results."""
        summary = {
            "total_splice_points": len(results),
            "problems_tested": len(set(r['problem_number'] for r in results)),
            "average_hinted_answer_probability": sum(r['hinted_answer_probability'] for r in results) / len(results),
            "average_correct_answer_probability": sum(r['correct_answer_probability'] for r in results) / len(results),
            "splice_points_by_delta": {}
        }
        
        # Group by delta ranges
        for result in results:
            delta = result['delta_cue_p']
            if delta >= 0.7:
                bucket = "0.7+"
            elif delta >= 0.5:
                bucket = "0.5-0.7"
            else:
                bucket = "<0.5"
            
            if bucket not in summary['splice_points_by_delta']:
                summary['splice_points_by_delta'][bucket] = {
                    'count': 0,
                    'avg_hinted_prob': 0
                }
            
            summary['splice_points_by_delta'][bucket]['count'] += 1
            summary['splice_points_by_delta'][bucket]['avg_hinted_prob'] += result['hinted_answer_probability']
        
        # Average the probabilities
        for bucket in summary['splice_points_by_delta']:
            count = summary['splice_points_by_delta'][bucket]['count']
            summary['splice_points_by_delta'][bucket]['avg_hinted_prob'] /= count
        
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n=== Summary ===")
        print(f"Total splice points tested: {summary['total_splice_points']}")
        print(f"Average hinted answer probability: {summary['average_hinted_answer_probability']:.3f}")
        print(f"Average correct answer probability: {summary['average_correct_answer_probability']:.3f}")


async def main():
    parser = argparse.ArgumentParser(description='Run splice experiment')
    parser.add_argument('--delta-threshold', type=float, default=config.DEFAULT_DELTA_THRESHOLD,
                       help='Minimum delta_cue_p to consider (default: 0.5)')
    parser.add_argument('--num-samples', type=int, default=config.DEFAULT_NUM_SAMPLES,
                       help='Number of continuations per splice point (default: 5)')
    parser.add_argument('--concurrency', type=int, default=config.DEFAULT_CONCURRENCY,
                       help='Max concurrent API calls (default: 5)')
    parser.add_argument('--problems', type=str, default=None,
                       help='Comma-separated list of problem numbers to test (e.g., "3,288,972")')
    parser.add_argument('--cot-data', type=str, required=True,
                       help='Path to the original CoT data JSON file')
    
    args = parser.parse_args()
    
    # Parse problem numbers
    problem_numbers = None
    if args.problems:
        problem_numbers = [int(p.strip()) for p in args.problems.split(',')]
        print(f"Will process specific problems: {problem_numbers}")
    
    # Create experiment
    experiment = SpliceExperiment(
        delta_threshold=args.delta_threshold,
        num_samples=args.num_samples,
        concurrency=args.concurrency
    )
    
    # Run experiment
    await experiment.run(
        problem_numbers=problem_numbers,
        cot_data_path=Path(args.cot_data)
    )


if __name__ == "__main__":
    asyncio.run(main())

