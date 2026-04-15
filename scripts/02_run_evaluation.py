#!/usr/bin/env python3
"""
Script 02: Run Evaluation
=========================

Evaluate all samples with 20 RAG evaluation frameworks.

LLM-as-Judge Frameworks (DeepSeek primary, Groq/Gemini fallback, temperature=0):
- RAGAS, DeepEval, G-Eval, FaithJudge

Other Frameworks:
- RAGChecker, TRACe, ARES, BERTScore, UniEval, QAFactEval
- ReDeEP, LettuceDetect, LRP4RAG, LUMINA, HALT-RAG, MetaRAG
- KG-RAG, GaRAGe, HSAD, SIRG

Usage:
    # Run with LLM-as-judge (DeepSeek primary, Groq/Gemini fallback)
    python scripts/02_run_evaluation.py --use-llm

    # Run with heuristics only (testing mode, no API costs)
    python scripts/02_run_evaluation.py --no-llm

Environment:
    Configure API keys in secrets/token.txt (see README)
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load tokens early so env vars (e.g. HF_TOKEN) are set before imports
def _load_env_from_tokens():
    token_path = os.path.join(os.path.dirname(__file__), "..", "secrets", "token.txt")
    if os.path.exists(token_path):
        with open(token_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    if k == "HF_TOKEN" and not os.environ.get("HF_TOKEN"):
                        os.environ["HF_TOKEN"] = v.strip()

_load_env_from_tokens()

# Import all framework evaluators - Traditional (2020-2024)
from src.frameworks.ragas_eval import RAGASEvaluator
from src.frameworks.deepeval_eval import DeepEvalEvaluator
from src.frameworks.ragchecker import RAGCheckerEvaluator
from src.frameworks.trace_eval import TRACeEvaluator
from src.frameworks.ares_eval import ARESEvaluator
from src.frameworks.geval import GEvalEvaluator
from src.frameworks.bertscore_eval import BERTScoreEvaluator
from src.frameworks.unieval import UniEvalEvaluator
from src.frameworks.qafacteval import QAFactEvalEvaluator

# Import 2025 frameworks
from src.frameworks.kgrag_eval import KGRAGEvaluator
from src.frameworks.faithjudge_eval import FaithJudgeEvaluator
from src.frameworks.garage_eval import GaRAGeEvaluator
from src.frameworks.hsad_eval import HSADEvaluator

# Import NEW 2025 frameworks (added Feb 2026)
from src.frameworks.redeep_eval import ReDeEPEvaluator
from src.frameworks.lettucedetect_eval import LettuceDetectEvaluator
from src.frameworks.lumina_eval import LUMINAEvaluator
from src.frameworks.metarag_eval import MetaRAGEvaluator

# Import additional 2025 frameworks
from src.frameworks.haltrag_eval import HALTRAGEvaluator
from src.frameworks.lrp4rag_eval import LRP4RAGEvaluator

# Import 2026 frameworks
from src.frameworks.sirg_eval import SIRGEvaluator


def get_output_dir():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Also create a 'latest' symlink (if possible on Windows)
    latest_link = 'results/latest'
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            import shutil
            shutil.rmtree(latest_link)
        os.symlink(f'run_{timestamp}', latest_link)
    except (OSError, NotImplementedError):
        pass

    return output_dir


def load_samples(path: str = 'data/samples_200.json'):
    """Load evaluation samples."""
    with open(path, 'r') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    return samples


def initialize_frameworks(use_llm: bool = True):
    """
    Initialize all 20 evaluation frameworks.

    When use_llm=True, builds a Groq client with Gemini fallback from
    secrets/token.txt. If Groq is rate-limited, calls fall back to Gemini
    automatically.
    """
    from src.llm_client import load_tokens, get_llm_client

    print("\nInitializing 20 evaluation frameworks...")

    llm_client = None
    if use_llm:
        tokens = load_tokens()
        has_keys = tokens.get("DEEPSEEK_API_KEY") or tokens.get("GROQ_API_KEY") or tokens.get("GEMINI_API_KEY")
        if not has_keys:
            print("  WARNING: No API keys in secrets/token.txt. Falling back to heuristics.")
            use_llm = False
        else:
            print("  Mode: LLM-as-judge (DeepSeek primary, Groq/Gemini fallback)")
            try:
                llm_client = get_llm_client(tokens=tokens)
            except Exception as e:
                print(f"  WARNING: Could not initialize LLM client: {e}")
                print("  Falling back to heuristics.")
                use_llm = False
    else:
        print("  Mode: Heuristic only (testing)")

    print("\n  Traditional frameworks (2020-2024):")

    # LLM-as-judge frameworks
    llm_frameworks = []
    if use_llm:
        llm_frameworks = [
            RAGASEvaluator(use_llm=True, llm_client=llm_client),
            DeepEvalEvaluator(use_llm=True, llm_client=llm_client),
            GEvalEvaluator(use_llm=True, llm_client=llm_client),
        ]
    else:
        llm_frameworks = [
            RAGASEvaluator(use_llm=False, noise_std=0.05),
            DeepEvalEvaluator(use_llm=False, noise_std=0.06),
            GEvalEvaluator(use_llm=False, noise_std=0.045),
        ]

    for fw in llm_frameworks:
        mode = "LLM" if use_llm else "heuristic"
        print(f"    [OK] {fw.name}: {len(fw.metrics)} metrics ({mode})")

    # Non-LLM traditional frameworks
    traditional = [
        RAGCheckerEvaluator(noise_std=0.055),
        TRACeEvaluator(noise_std=0.03),
        ARESEvaluator(noise_std=0.04),
        BERTScoreEvaluator(noise_std=0.04),
        UniEvalEvaluator(noise_std=0.05),
        QAFactEvalEvaluator(noise_std=0.045),
    ]

    for fw in traditional:
        print(f"    [OK] {fw.name}: {len(fw.metrics)} metrics")

    print("  2025 frameworks (Early - ICLR, Feb, May, Jun):")

    # FaithJudge with LLM support
    if use_llm:
        faithjudge = FaithJudgeEvaluator(use_llm=True, llm_client=llm_client)
        print(f"    [OK] {faithjudge.name}: {len(faithjudge.metrics)} metrics (LLM, {faithjudge.version})")
    else:
        faithjudge = FaithJudgeEvaluator(use_llm=False, noise_std=0.04)
        print(f"    [OK] {faithjudge.name}: {len(faithjudge.metrics)} metrics (heuristic, {faithjudge.version})")

    # Other 2025 Early frameworks
    early_2025 = [
        ReDeEPEvaluator(noise_std=0.04),
        LettuceDetectEvaluator(noise_std=0.04),
        LRP4RAGEvaluator(noise_std=0.04),
    ]

    for fw in early_2025:
        print(f"    [OK] {fw.name}: {len(fw.metrics)} metrics ({fw.version})")

    print("  2025 frameworks (Mid-Late - Sep, Oct):")

    late_2025 = [
        LUMINAEvaluator(noise_std=0.04),
        HALTRAGEvaluator(noise_std=0.04),
        MetaRAGEvaluator(noise_std=0.04),
        KGRAGEvaluator(noise_std=0.05),
        GaRAGeEvaluator(noise_std=0.05),
        HSADEvaluator(noise_std=0.04),
    ]

    for fw in late_2025:
        print(f"    [OK] {fw.name}: {len(fw.metrics)} metrics ({fw.version})")

    print("  2026 frameworks:")

    frameworks_2026 = [
        SIRGEvaluator(noise_std=0.04),
    ]

    for fw in frameworks_2026:
        print(f"    [OK] {fw.name}: {len(fw.metrics)} metrics ({fw.version})")

    # Combine all frameworks
    frameworks = llm_frameworks + traditional + [faithjudge] + early_2025 + late_2025 + frameworks_2026
    return frameworks


def run_evaluation(samples, frameworks, verbose: bool = False):
    """
    Evaluate all samples with all frameworks.

    Args:
        samples: List of sample dictionaries
        frameworks: List of evaluator instances
        verbose: Print detailed progress

    Returns:
        DataFrame with all evaluation results
    """
    print(f"\nEvaluating {len(samples)} samples with {len(frameworks)} frameworks...")
    print("This may take a few minutes (longer with LLM mode)...\n")

    results = []

    for sample in tqdm(samples, desc="Evaluating samples"):
        sample_results = {
            'id': sample['id'],
            'domain': sample['domain'],
            'subset': sample.get('subset', 'unknown'),
            'human_label': sample.get('human_label', None)
        }

        # Run each framework
        for framework in frameworks:
            try:
                scores = framework.evaluate(sample)

                # Add scores with framework prefix
                for metric, value in scores.items():
                    key = f"{framework.name}_{metric}"
                    sample_results[key] = value

            except Exception as e:
                if verbose:
                    print(f"Warning: {framework.name} failed on {sample['id']}: {e}")
                # Add NaN for failed evaluations
                for metric in framework.metrics:
                    key = f"{framework.name}_{metric}"
                    sample_results[key] = float('nan')

        results.append(sample_results)

    return pd.DataFrame(results)




def min_max_scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-metric min-max scaling to [0, 1]."""
    scaled_df = df.copy()
    metric_cols = [c for c in scaled_df.columns if c not in ["id", "domain", "subset", "human_label"]]

    for col in metric_cols:
        col_min = scaled_df[col].min(skipna=True)
        col_max = scaled_df[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        if col_max == col_min:
            scaled_df[col] = 0.5
        else:
            scaled_df[col] = (scaled_df[col] - col_min) / (col_max - col_min)

    return scaled_df


def print_summary(df):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    metric_cols = [c for c in df.columns if c not in ['id', 'domain', 'subset']]
    frameworks = set()
    for col in metric_cols:
        frameworks.add(col.split('_')[0])

    print(f"\nFrameworks evaluated: {len(frameworks)}")
    print(f"Total metrics: {len(metric_cols)}")
    print(f"Samples evaluated: {len(df)}")

    print("\nDomain distribution:")
    for domain, count in df['domain'].value_counts().items():
        print(f"  - {domain}: {count}")

    print("\nMean faithfulness scores by framework:")
    for fw in sorted(frameworks):
        faith_col = f"{fw}_faithfulness"
        if faith_col in df.columns:
            mean_score = df[faith_col].mean()
            print(f"  - {fw}: {mean_score:.3f}")
        else:
            fw_cols = [c for c in metric_cols if c.startswith(f"{fw}_")]
            if fw_cols:
                alt_col = fw_cols[0]
                print(f"  - {fw} ({alt_col.split('_')[1]}): {df[alt_col].mean():.3f}")


def main(use_llm: bool = True):
    print("="*60)
    print("RAG Meta-Analysis: Framework Evaluation")
    print("="*60)

    # Create output directory
    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load samples
    samples = load_samples()

    # Initialize frameworks
    frameworks = initialize_frameworks(use_llm=use_llm)

    # Run evaluation
    raw_results_df = run_evaluation(samples, frameworks)
    results_df = min_max_scale_scores(raw_results_df)

    # Save scaled and raw results
    raw_results_df.to_csv(f'{output_dir}/evaluation_scores_raw.csv', index=False)
    results_df.to_csv(f'{output_dir}/evaluation_scores.csv', index=False)
    print(f"\n[OK] Results saved to {output_dir}/evaluation_scores.csv")

    # Also save as JSON for flexibility
    results_df.to_json(f'{output_dir}/evaluation_scores.json', orient='records', indent=2)
    raw_results_df.to_json(f'{output_dir}/evaluation_scores_raw.json', orient='records', indent=2)

    # Save evaluation config
    from src.llm_client import load_tokens
    tokens = load_tokens()
    config = {
        'use_llm': use_llm,
        'n_samples': len(samples),
        'n_frameworks': len(frameworks),
        'frameworks': [fw.name for fw in frameworks],
        'timestamp': datetime.now().isoformat(),
        'llm_primary': f"deepseek/{tokens.get('DEEPSEEK_MODEL', 'deepseek-chat')}",
        'llm_fallback': f"groq/{tokens.get('GROQ_MODEL', 'llama-3.3-70b-versatile')} -> gemini/{tokens.get('GEMINI_MODEL', 'gemini-2.5-flash')}",
        'normalization': 'min-max per metric to [0,1]'
    }
    with open(f'{output_dir}/evaluation_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print_summary(results_df)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

    return results_df, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation with all frameworks")
    parser.add_argument('--use-llm', action='store_true', default=True,
                        help='Use LLM for LLM-as-judge frameworks (default)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Use heuristics only (no API calls)')
    args = parser.parse_args()

    use_llm = not args.no_llm
    results, output_dir = main(use_llm=use_llm)
