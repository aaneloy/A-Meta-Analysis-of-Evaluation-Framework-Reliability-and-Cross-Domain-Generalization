# Do RAG Metrics Measure What Matters?

## A Meta-Analysis of Evaluation Framework Reliability and Cross-Domain Generalization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for our meta-analysis comparing **20 RAG (Retrieval-Augmented Generation) evaluation frameworks** across **3 domains** (General Knowledge, Finance, Biomedicine). We evaluate inter-framework agreement, cross-domain generalization, and human correlation to answer critical questions about metric reliability.

**Paper**: Accepted at the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026).

**Authors**: Asif Ahmed Neloy (University of British Columbia) and MD Nazmul Islam (Keyano College).

---

## Table of Contents

- [Research Questions](#research-questions)
- [Key Findings](#key-findings)
- [Frameworks](#frameworks)
- [Dataset](#dataset)
- [Installation](#installation)
- [Reproducing Results](#reproducing-results)
- [Repository Structure](#repository-structure)
- [Evaluation Pipeline Details](#evaluation-pipeline-details)
- [Statistical Methods](#statistical-methods)
- [Practical Recommendations](#practical-recommendations)
- [Citation](#citation)
- [License](#license)

---

## Research Questions

1. **RQ1: Inter-Framework Agreement** -- How well do different RAG evaluation frameworks agree with each other when measuring the same quality dimensions (e.g., faithfulness)?

2. **RQ2: Cross-Domain Generalization** -- Do evaluation metrics generalize consistently across different domains (General Knowledge, Finance, Biomedicine)?

3. **RQ3: Metric Reliability** -- Which metrics show the highest reliability and consistency across frameworks?

4. **RQ4: Minimum Viable Evaluation** -- What is the minimum evaluation setup needed to reliably assess RAG system quality?

---

## Key Findings

- **Substantial heterogeneity.** Cochran's Q = 10,055.63 (p < 0.001, df = 19), I^2 = 99.81%. Virtually all variance reflects true differences between frameworks rather than sampling error.

- **Three framework clusters emerge.** Hierarchical clustering (Ward's method) identifies three methodological families:
  - *LLM-as-Judge* (4 frameworks: RAGAS, G-Eval, DeepEval, FaithJudge): within-cluster mean r = 0.55
  - *Mixed Methods* (11 frameworks: ARES, HSAD, KG-RAG, LRP4RAG, LUMINA, LettuceDetect, MetaRAG, ReDeEP, SIRG, TRACe, UniEval): within-cluster mean r = 0.63
  - *Outlier* (5 frameworks: BERTScore, GaRAGe, HALT-RAG, QAFactEval, RAGChecker): within-cluster mean r = 0.01
  - Between-cluster correlations average r = 0.10.
  - Cluster structure is perfectly stable across normalization schemes (ARI = 1.0) and bootstrap resampling (co-assignment probabilities at least 0.75 within the LLM-as-judge and mixed-methods clusters).

- **Domain effects are significant.** Finance samples score lower than General Knowledge for most frameworks (two-way ANOVA interaction F = 6.95, p < 0.001). Interpretability and token-level methods show the strongest domain sensitivity (KG-RAG delta = 0.267, MetaRAG delta = 0.242), while LLM-as-judge methods remain stable (FaithJudge delta = 0.030, RAGAS delta = 0.097).

- **Human correlation varies by cluster.** FaithJudge achieves the highest human correlation (r = 0.596), followed by G-Eval (r = 0.564). The LLM-as-judge cluster shows the strongest alignment with human judgments (cluster mean r = 0.51), while the outlier cluster averages r = 0.05.

- **Cross-cluster consensus is rare.** A consensus protocol labels 92.5% of samples as contested, indicating that the three clusters rarely converge on a shared verdict for the same sample.

---

## Frameworks

We evaluate 20 frameworks spanning four methodological categories and six years of development (2020--2026).

### Traditional Frameworks (2020--2024)

| Framework | Category | Year | Key Metrics | Reference |
|-----------|----------|------|-------------|-----------|
| BERTScore | Embedding | 2020 | Precision, Recall, F1 | [Zhang et al., ICLR 2020](https://arxiv.org/abs/1904.09675) |
| UniEval | QA-based | 2022 | Multi-dimension unified | [Zhong et al., 2022](https://arxiv.org/abs/2210.07197) |
| QAFactEval | QA-based | 2022 | Factual Consistency | [Fabbri et al., 2022](https://arxiv.org/abs/2112.08542) |
| G-Eval | LLM-judge | 2023 | Coherence, Consistency, Fluency, Relevance | [Liu et al., EMNLP 2023](https://arxiv.org/abs/2303.16634) |
| RAGAS | LLM-judge | 2024 | Faithfulness, Answer Relevancy, Context Precision/Recall | [Es et al., EACL 2024](https://arxiv.org/abs/2309.15217) |
| DeepEval | LLM-judge | 2024 | Faithfulness, Answer Relevancy, Hallucination | [Confident AI](https://github.com/confident-ai/deepeval) |
| RAGChecker | NLI | 2024 | Claim Recall, Faithfulness, Noise Sensitivity | [Ru et al., NeurIPS 2024](https://arxiv.org/abs/2408.08067) |
| TRACe | Token-level | 2024 | Utilization, Relevance, Adherence, Completeness | [Friel et al., 2024](https://arxiv.org/abs/2407.11005) |
| ARES | Classifier | 2024 | Context/Answer Relevance, Faithfulness | [Saad-Falcon et al., 2024](https://arxiv.org/abs/2311.09476) |

### 2025 Frameworks

| Framework | Category | Year | Key Metrics | Reference |
|-----------|----------|------|-------------|-----------|
| ReDeEP | Interpretability | 2025 | External Context Score, Parametric Knowledge Score | [Sun et al., ICLR 2025](https://arxiv.org/abs/2410.11414) |
| LettuceDetect | Token Detection | 2025 | Token Hallucination, Span Precision, F1 | [Kovacs & Recski, 2025](https://arxiv.org/abs/2502.17125) |
| FaithJudge | LLM-judge | 2025 | Faithfulness, Hallucination Rate, Grounding Score | [Tamber et al., EMNLP 2025](https://arxiv.org/abs/2505.04847) |
| LRP4RAG | Interpretability | 2025 | Context Relevance, Response Relevance, Classifier Score | [Hu et al., 2025](https://arxiv.org/abs/2408.15533v3) |
| LUMINA | Interpretability | 2025 | Context Utilization, Knowledge Utilization, LUMINA Score | [Yeh et al., NeurIPS 2025 Workshop](https://arxiv.org/abs/2509.21875) |
| HALT-RAG | NLI Ensemble | 2025 | NLI Ensemble Score, Calibrated Confidence, Abstention Score | [Kurra et al., 2025](https://arxiv.org/abs/2509.07475) |
| MetaRAG | Meta-evaluation | 2025 | Factoid Consistency, Synonym Stability, Antonym Detection | [Sok et al., ECAI 2025](https://arxiv.org/abs/2509.09360) |
| KG-RAG | Knowledge Graph | 2025 | KG Faithfulness, Multi-hop Score, Community Overlap | [Dong et al., 2025](https://arxiv.org/abs/2510.02549) |
| GaRAGe | LLM-judge | 2025 | Relevance, Grounding, Completeness | [Sorodoc et al., ACL 2025](https://arxiv.org/abs/2412.13834) |
| HSAD | Signal Analysis | 2025 | Spectral Density, Pattern Coherence, Entropy Score | [Li et al., 2025](https://arxiv.org/abs/2509.13154) |

### 2026 Frameworks

| Framework | Category | Year | Key Metrics | Reference |
|-----------|----------|------|-------------|-----------|
| SIRG | Reasoning Graph | 2026 | Semantic Grounding, Linking Ratio, Attribution Consistency | [Hu et al., 2026](https://arxiv.org/abs/2601.03052) |

---

## Dataset

We use **RAGBench** ([Friel et al., 2024](https://arxiv.org/abs/2407.11005)) with stratified sampling across 3 domains:

| Domain | Samples | Sources | Description |
|--------|---------|---------|-------------|
| General Knowledge | 67 | HotpotQA, MS MARCO, HAGRID | Factual questions about people, places, events |
| Finance | 67 | FinQA, DelucionQA | Numerical reasoning over financial documents |
| Biomedicine | 66 | CovidQA, PubMedQA | Medical research findings |

**Total: 200 samples** with question, context, answer, and binary human faithfulness annotations (faithful/unfaithful), balanced within each domain.

Data file: `data/samples_200.json`

Each sample contains:
```json
{
  "id": "hotpotqa_327",
  "domain": "General Knowledge",
  "subset": "hotpotqa",
  "question": "...",
  "answer": "...",
  "context": "...",
  "ground_truth": {},
  "metadata": {"source": "hotpotqa", "has_trace_labels": false},
  "human_label": 1
}
```

---

## Installation

### Prerequisites

- Python 3.9+
- (Optional) CUDA-compatible GPU for NLI model inference
- (Optional) OpenAI API key for LLM-as-judge evaluations

### Setup

```bash
# Clone the repository
git clone https://github.com/aaneloy/A-Meta-Analysis-of-Evaluation-Framework-Reliability-and-Cross-Domain-Generalization.git
cd A-Meta-Analysis-of-Evaluation-Framework-Reliability-and-Cross-Domain-Generalization

# Create virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core dependencies (`requirements.txt`):

| Category | Packages |
|----------|----------|
| Core | numpy, pandas, scipy, scikit-learn |
| Deep Learning | torch, transformers, sentence-transformers |
| NLI Model | transformers (loads `microsoft/deberta-v3-large-mnli`) |
| BERTScore | bert-score (loads `microsoft/deberta-xlarge-mnli`) |
| LLM API | openai (OpenAI-compatible client for Groq/Gemini/OpenAI) |
| Statistics | statsmodels, pingouin |
| Visualization | matplotlib, seaborn, plotly |
| Data | datasets, huggingface-hub |
| NLP | nltk, spacy |

### API Keys

API keys are stored in `secrets/token.txt` (git-ignored). The pipeline uses a 3-level fallback chain: DeepSeek (primary) -> Groq -> Gemini. If the primary is rate-limited, the next provider is used automatically.

```bash
# DeepSeek (primary)
DEEPSEEK_API_KEY=sk-your_deepseek_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Groq (fallback 1)
GROQ_API_KEY=gsk_your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Gemini (fallback 2)
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Hugging Face
HF_TOKEN=hf_your_hf_token
```

Get keys: [DeepSeek](https://platform.deepseek.com/api_keys) | [Groq](https://console.groq.com/keys) | [Gemini](https://aistudio.google.com/apikey)

---

## Reproducing Results

The pipeline is organised in two stages. Stage A (scripts 01 to 05) reproduces every number in the main body of the paper. Stage B (scripts 06 to 09) runs the robustness and validation analyses that were added during the camera-ready revision. The stages are independent: Stage B reads the outputs of Stage A and does not modify them.

### Stage A. Core pipeline (reproduces the main tables and figures)

```bash
# Step 1: Prepare the dataset (downloads from RAGBench, creates samples_200.json)
python scripts/01_prepare_dataset.py

# Step 2: Run all 20 frameworks on 200 samples
#   - LLM-as-judge frameworks use DeepSeek V3 (primary) / Groq / Gemini (fallback) with temperature=0
#   - NLI frameworks use DeBERTa-v3-large fine-tuned on MNLI
#   - BERTScore uses microsoft/deberta-xlarge-mnli
#   - Applies min-max normalization to [0,1] per metric
#   - Saves both raw and normalized scores
python scripts/02_run_evaluation.py --use-llm

# Step 3: Run statistical analysis
#   - Pairwise correlations (Pearson, Spearman, Kendall) with bootstrap CIs
#   - Cochran's Q and I^2 heterogeneity statistics (df=19 over 20 frameworks)
#   - Binary agreement at threshold tau=0.5, robustness across tau in {0.3,0.4,0.5,0.6,0.7}
#   - Hierarchical clustering (Ward's method) with within-cluster mean correlations
#   - Human label correlations (Pearson, Spearman vs binary annotations)
#   - Mixed-effects regression for domain effects
#   - One-way ANOVA with eta-squared effect sizes
python scripts/03_analyze_results.py

# Step 4: Generate publication-quality figures
python scripts/04_generate_figures.py

# Step 5: Failure case analysis (disagreement cases)
python scripts/05_failure_analysis.py
```

### Stage B. Robustness and human validation (camera-ready additions)

Stage B reproduces the revision-round analyses without re-running any framework. All four scripts read from the latest `results/run_<timestamp>/` directory by default and write new subdirectories inside it.

```bash
# Step 6: Normalization sensitivity analysis (Section 5, Appendix A.1)
#   - Rebuilds correlations, Cochran's Q, I^2, and Ward clusters under
#     four normalization schemes: raw, min-max, z-score, and rank.
#   - Reports the adjusted Rand index between every pair of schemes.
#   - Uses the unmodified evaluation_scores_raw.csv so no re-scoring is needed.
python scripts/06_normalization_sensitivity.py

# Step 7: Cluster stability validation (Section 5, Appendix A.2)
#   - Bootstrap co-assignment matrix over B resamples of the 200 samples.
#   - Silhouette width and gap statistic for k in {2..8}.
#   - Adjusted Rand index between Ward and average/complete/single linkage.
python scripts/07_cluster_stability.py --n-boot 1000

# Step 8a: Build the 50-sample human annotation template
#   - Stratified by domain (GK 17, Finance 17, Biomedicine 16).
#   - Within each domain, samples span five disagreement percentiles
#     (max-min range across the 20 primary scores).
#   - Deterministic under seed 20260409.
python scripts/08_human_validation.py --select

# At this point the annotator fills in data/human_validation_template.csv
# following docs/ANNOTATION_PROTOCOL.md and saves the result as
# data/human_validation_filled.csv. A 10-sample re-annotation pass
# (data/human_validation_reannot.csv) is optional and drives the
# intra-rater kappa estimate.

# Step 8b: Analyse the human annotations
#   - Per-framework Pearson and Spearman correlation with the human label,
#     with 95% bootstrap CIs.
#   - Per-cluster mean human correlation.
#   - Intra-rater Cohen's kappa on the re-annotation subset.
python scripts/08_human_validation.py --analyze \
    --filled data/human_validation_filled.csv \
    --reannot data/human_validation_reannot.csv

# Step 9: Cross-cluster consensus protocol (Section 6.3)
#   - Picks one representative framework per cluster (the member with the
#     highest mean within-cluster correlation).
#   - Labels each sample as faithful, unfaithful, or contested based on
#     whether the representatives agree at threshold tau.
#   - If data/human_validation_filled.csv exists, reports coverage and
#     accuracy of the non-contested verdicts against the human labels.
python scripts/09_consensus_protocol.py --tau 0.5
```

### Testing Mode (No API Costs)

To run without OpenAI API calls (uses heuristic approximations for LLM-as-judge frameworks):

```bash
python scripts/02_run_evaluation.py --no-llm
python scripts/03_analyze_results.py
```

### Using a Specific Results Directory

```bash
python scripts/03_analyze_results.py --results-dir results/run_20260213_101355
python scripts/04_generate_figures.py --results-dir results/run_20260213_101355
python scripts/05_failure_analysis.py --results-dir results/run_20260213_101355
```

### Step-by-Step Explanation

#### Step 1: Dataset Preparation (`01_prepare_dataset.py`)

Downloads RAGBench data and creates a stratified sample of 200 instances balanced across 3 domains and faithfulness labels (faithful vs. unfaithful). Output: `data/samples_200.json`.

#### Step 2: Evaluation (`02_run_evaluation.py`)

Runs all 20 frameworks on each sample. Frameworks are organized by methodology:

- **LLM-as-Judge** (RAGAS, DeepEval, G-Eval, FaithJudge): Uses DeepSeek V3 (`deepseek-chat`) as primary with Groq and Gemini as automatic fallbacks, temperature 0 via OpenAI-compatible API. Each framework sends structured prompts and parses JSON responses. Configured in `secrets/token.txt`.
- **NLI-based** (RAGChecker, HALT-RAG): Uses `microsoft/deberta-v3-large-mnli` for entailment/contradiction/neutral classification via the `src/nli.py` module.
- **BERTScore**: Uses the official `bert-score` library with `microsoft/deberta-xlarge-mnli` and baseline rescaling.
- **All others**: Heuristic evaluators using lexical features, n-gram overlap, and coverage metrics.

After evaluation, **min-max normalization** is applied per metric to scale all scores to [0, 1], preserving rank ordering. Both raw and normalized CSVs are saved.

Output: `results/run_<timestamp>/evaluation_scores.csv` (normalized), `evaluation_scores_raw.csv` (raw).

#### Step 3: Analysis (`03_analyze_results.py`)

Computes all statistical analyses reported in the paper:

| Analysis | Method | Output File |
|----------|--------|-------------|
| Heterogeneity | Cochran's Q, I^2 over 20 primary metrics | `heterogeneity.json` |
| Correlations | Pearson, Spearman, Kendall with bootstrap 95% CIs | `pairwise_correlations.csv`, `bootstrap_ci.csv` |
| Agreement | Cohen's kappa at tau=0.5 | `agreement_matrix.csv`, `kappa_matrix.csv` |
| Threshold robustness | Agreement at tau in {0.3, 0.4, 0.5, 0.6, 0.7} | `threshold_robustness.csv` |
| Clustering | Ward's method, d = sqrt(2(1-r)), 3 clusters | `clustering.json` |
| Human correlation | Pearson/Spearman vs binary human labels | `human_label_correlations.csv` |
| Domain effects | One-way ANOVA, eta-squared | `domain_anova.csv` |
| Mixed-effects model | score ~ domain (fixed) + (1\|sample) | `mixed_effects_model.csv` |
| Framework summary | Mean, std, min, max per framework | `framework_summary.csv` |

#### Step 4: Figures (`04_generate_figures.py`)

Generates publication-quality PDF figures:
- Correlation heatmap
- Domain comparison plots
- Score distributions
- Scatter matrices
- Agreement heatmaps
- Forest plots

#### Step 5: Failure Analysis (`05_failure_analysis.py`)

Identifies samples where frameworks disagree most, generates qualitative analysis of disagreement patterns for the paper's discussion section.

#### Step 6: Normalization Sensitivity (`06_normalization_sensitivity.py`)

Reads `evaluation_scores_raw.csv`, applies four normalization schemes (`raw`, `minmax`, `zscore`, `rank`) defined in `src/normalization.py`, and recomputes Cochran's Q, I^2, the mean pairwise Pearson correlation, and a Ward clustering at k=3 under each scheme. The output tables sit in `results/<run>/normalization_sensitivity/`:

| File | Content |
|------|---------|
| `summary.csv` | Per-scheme Q, I^2, mean r, within- and between-cluster r |
| `correlation_<scheme>.csv` | 20 x 20 Pearson correlation matrix under each scheme |
| `cluster_assignments.csv` | Cluster label of each framework under each scheme |
| `ari_matrix.csv` | Adjusted Rand index between every pair of schemes |

#### Step 7: Cluster Stability (`07_cluster_stability.py`)

Runs three validations on the three-cluster Ward solution, implemented in `src/cluster_stability.py`. Outputs in `results/<run>/cluster_stability/`:

| File | Content |
|------|---------|
| `coassignment_matrix.csv` | Bootstrap probability that each pair of frameworks ends up in the same cluster (default B = 1000 resamples of the 200 samples) |
| `k_selection.csv` | Silhouette width and Tibshirani et al. (2001) gap statistic for k in {2..8} |
| `linkage_comparison.json` | Adjusted Rand index between Ward and average, complete, and single linkage at k=3 |

#### Step 8: Human Validation (`08_human_validation.py`)

Two entry points. `--select` reads `data/samples_200.json` and the latest `evaluation_scores.csv`, selects 50 samples stratified by domain and disagreement percentile, and writes `data/human_validation_template.csv` together with a manifest. The annotator fills in `human_label`, `confidence`, and `notes` following `docs/ANNOTATION_PROTOCOL.md`.

`--analyze` merges the filled CSV with the per-sample framework scores and writes `results/<run>/human_validation/`:

| File | Content |
|------|---------|
| `framework_human_correlation.csv` | Per-framework Pearson and Spearman correlation with the human label, with 95% bootstrap CIs |
| `cluster_alignment.csv` | Mean framework-human correlation per Ward cluster |
| `intra_rater_reliability.json` | Cohen's kappa and percent agreement between the first and second annotation passes (only if `--reannot` is supplied) |

#### Step 9: Cross-Cluster Consensus (`09_consensus_protocol.py`)

Applies the decision rule defined in `src/consensus.py`. For each cluster, the most centrally correlated framework is chosen as the representative. A sample is labelled `faithful` when all representatives vote 1 at threshold `tau`, `unfaithful` when they all vote 0, and `contested` otherwise. Outputs in `results/<run>/consensus/`:

| File | Content |
|------|---------|
| `representatives.json` | Representative framework chosen for each cluster |
| `verdicts.csv` | Per-sample verdict and per-representative binary votes |
| `coverage.json` | Percent of samples labelled faithful, unfaithful, or contested |
| `human_agreement.json` | Coverage and accuracy of non-contested verdicts against the human labels, written only when `data/human_validation_filled.csv` exists |

---

## End-to-End Verification Protocol

The following protocol reproduces every number reported in the paper from a clean checkout. It is the same sequence used to validate the code before camera-ready submission.

### Prerequisites

- Python 3.9 or newer, with `pip install -r requirements.txt`.
- Optional: API keys in `secrets/token.txt` (DeepSeek, Groq, and/or Gemini). Without them, all LLM-as-judge frameworks fall back to the heuristic path (`--no-llm`), which is sufficient for verifying pipeline correctness but not for reproducing the production scores.
- Optional: a CUDA device. NLI inference runs on CPU but is slow.

### Step-by-step verification

1. **Clean state check.** Confirm that `results/` contains no stale `run_*` directory that you intend to overwrite.
2. **Stage A run.** Execute scripts 01 to 05 in order. Expected wall time is dominated by Step 2 (under 10 minutes on CPU in `--no-llm` mode, longer with the OpenAI API).
3. **Artefact check (Stage A).** The latest `results/run_<timestamp>/` directory must contain:
   - `evaluation_scores.csv` (200 rows, 82 columns).
   - `evaluation_scores_raw.csv` (same shape, pre-normalization).
   - `heterogeneity.json`, `clustering.json`, `pairwise_correlations.csv`, `bootstrap_ci.csv`, `agreement_matrix.csv`, `kappa_matrix.csv`, `threshold_robustness.csv`, `domain_anova.csv`, `domain_statistics.csv`, `framework_summary.csv`, `latex_tables.tex`, `correlation_summary_for_paper.txt`.
   - A `figures/` subdirectory populated with PDFs.
4. **Main-table reproduction.** Open `correlation_summary_for_paper.txt` and `heterogeneity.json`. Cochran's Q, I^2, and the three-cluster within/between correlations must match Table 1, Table 4, and the headline numbers in Section 5.1 of the paper.
5. **Stage B run (robustness).** Execute scripts 06 and 07. Expected wall time under five minutes.
6. **Artefact check (robustness).**
   - `results/<run>/normalization_sensitivity/ari_matrix.csv` reports the adjusted Rand index between normalization schemes. The paper claims cluster structure is normalization-invariant; all off-diagonal entries should be 1.0 on the reference run.
   - `results/<run>/cluster_stability/k_selection.csv` reports the gap statistic. The paper claims k=3 is the optimal choice; the maximum gap should fall at k=3.
   - `results/<run>/cluster_stability/linkage_comparison.json` reports the adjusted Rand index between Ward and the three alternative linkages.
7. **Stage B run (human validation).** Execute `python scripts/08_human_validation.py --select`. The script must write exactly 50 rows to `data/human_validation_template.csv` with the domain breakdown 17 / 17 / 16.
8. **Annotation.** Following `docs/ANNOTATION_PROTOCOL.md`, label every row of the template. Save the result as `data/human_validation_filled.csv`. For intra-rater reliability, copy 10 randomly chosen rows into a second file, clear their labels, re-annotate at least 24 hours later, and save as `data/human_validation_reannot.csv`.
9. **Stage B run (consensus and analysis).**
   ```bash
   python scripts/08_human_validation.py --analyze \
       --filled data/human_validation_filled.csv \
       --reannot data/human_validation_reannot.csv
   python scripts/09_consensus_protocol.py --tau 0.5
   ```
10. **Artefact check (human validation).** `results/<run>/human_validation/framework_human_correlation.csv` must contain one row per primary metric with finite Pearson and Spearman values and non-empty bootstrap CIs. `results/<run>/human_validation/cluster_alignment.csv` must contain one row per Ward cluster. `results/<run>/consensus/coverage.json` reports faithful / unfaithful / contested percentages, and `human_agreement.json` reports accuracy and coverage against the human labels.
11. **Cross-check.** The numbers in the revised Section 5.4 (normalization sensitivity), Section 5.5 (cluster stability), Section 5.6 (human validation), and Section 6.3 (consensus protocol) of the camera-ready must match the contents of the files listed in the previous step. Any mismatch indicates either a stale run directory or a modified seed.

### Reproducibility notes

- All Stage B scripts are deterministic under the default seed (`20260409`). Changing `--n-boot`, `--n-reference`, or the seed will change the stability numbers but not the cluster assignments.
- Stage B never writes back into files produced by Stage A. It is safe to re-run Stage B repeatedly without re-running evaluation.
- Stage A uses DeepSeek V3 (`deepseek-chat`) as primary LLM judge with Groq and Gemini as fallbacks. Different LLM judge models will produce different scores and therefore different clusters. This is discussed in the limitations section of the paper.

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── main.tex                     # Paper source (Canadian AI 2026, accepted)
├── references.bib               # BibTeX references
├── fig/                         # Publication figures (PDF, used by main.tex)
│
├── data/
│   ├── samples_200.json                 # 200 evaluation samples
│   ├── human_validation_template.csv    # 50-sample annotation template (Step 8a)
│   ├── human_validation_manifest.json   # IDs and seed for the selected subset
│   ├── human_validation_filled.csv      # Filled by annotator (not in repo)
│   ├── human_validation_reannot.csv     # Re-annotation subset (not in repo)
│   └── raw/                             # Raw dataset cache (from RAGBench)
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── TAXONOMY.md
│   ├── LIMITATIONS.md
│   └── ANNOTATION_PROTOCOL.md           # Rules for the human validation subset
│
├── src/
│   ├── __init__.py
│   ├── nli.py                   # DeBERTa-v3-large MNLI scorer (safe_nli_score)
│   ├── llm_client.py            # OpenAI-compatible LLM client (Groq/Gemini/OpenAI)
│   ├── metrics.py               # Core statistical analysis (correlations, heterogeneity,
│   │                            #   clustering, agreement, mixed-effects, human corr)
│   ├── normalization.py         # raw / minmax / zscore / rank schemes (Step 6)
│   ├── cluster_stability.py     # bootstrap co-assignment, gap statistic, linkage ARI (Step 7)
│   ├── human_validation.py      # stratified selection, IAA, human correlation (Step 8)
│   ├── consensus.py             # cross-cluster consensus protocol (Step 9)
│   ├── visualization.py         # Publication-quality plotting
│   ├── failure_analysis.py      # Disagreement case analysis
│   │
│   └── frameworks/              # 20 evaluation framework implementations
│       ├── __init__.py          # BaseEvaluator, HeuristicEvaluator base classes
│       ├── ragas_eval.py        # RAGAS (LLM + heuristic modes)
│       ├── deepeval_eval.py     # DeepEval (LLM + heuristic modes)
│       ├── geval.py             # G-Eval (LLM + heuristic modes)
│       ├── faithjudge_eval.py   # FaithJudge (LLM + heuristic modes)
│       ├── ragchecker.py        # RAGChecker (NLI-based)
│       ├── haltrag_eval.py      # HALT-RAG (dual NLI ensemble)
│       ├── bertscore_eval.py    # BERTScore (official bert-score library)
│       ├── trace_eval.py        # TRACe
│       ├── ares_eval.py         # ARES
│       ├── unieval.py           # UniEval
│       ├── qafacteval.py        # QAFactEval
│       ├── redeep_eval.py       # ReDeEP
│       ├── lettucedetect_eval.py # LettuceDetect
│       ├── lrp4rag_eval.py      # LRP4RAG
│       ├── lumina_eval.py       # LUMINA
│       ├── metarag_eval.py      # MetaRAG
│       ├── kgrag_eval.py        # KG-RAG
│       ├── garage_eval.py       # GaRAGe
│       ├── hsad_eval.py         # HSAD
│       └── sirg_eval.py         # SIRG
│
├── scripts/
│   ├── 01_prepare_dataset.py             # Download and sample from RAGBench
│   ├── 02_run_evaluation.py              # Run all 20 frameworks + min-max normalize
│   ├── 03_analyze_results.py             # Full statistical analysis
│   ├── 04_generate_figures.py            # Publication-quality figures
│   ├── 05_failure_analysis.py            # Disagreement case analysis
│   ├── 06_normalization_sensitivity.py   # Sensitivity across 4 normalization schemes
│   ├── 07_cluster_stability.py           # Bootstrap, gap statistic, linkage ARI
│   ├── 08_human_validation.py            # --select template and --analyze labels
│   └── 09_consensus_protocol.py          # Cross-cluster consensus verdicts
│
├── results/                              # Output directory (auto-created per run)
│   └── run_<timestamp>/
│       ├── evaluation_scores.csv         # Normalized scores (200 x 20 frameworks)
│       ├── evaluation_scores_raw.csv     # Raw scores before normalization
│       ├── evaluation_config.json        # Run configuration
│       ├── heterogeneity.json            # Q, I^2 statistics
│       ├── clustering.json               # Ward's method cluster assignments
│       ├── pairwise_correlations.csv     # All pairwise correlations
│       ├── bootstrap_ci.csv              # 95% bootstrap CIs
│       ├── agreement_matrix.csv          # Binary agreement at tau=0.5
│       ├── kappa_matrix.csv              # Cohen's kappa
│       ├── threshold_robustness.csv      # Agreement across tau thresholds
│       ├── human_label_correlations.csv  # Correlations vs human annotations
│       ├── domain_statistics.csv         # Per-domain descriptive statistics
│       ├── domain_anova.csv              # ANOVA results
│       ├── mixed_effects_model.csv       # Mixed-effects regression
│       ├── framework_summary.csv         # Per-framework summary
│       ├── latex_tables.tex              # Auto-generated LaTeX tables
│       ├── correlation_summary_for_paper.txt
│       ├── normalization_sensitivity/    # Step 6 outputs
│       │   ├── summary.csv
│       │   ├── cluster_assignments.csv
│       │   ├── ari_matrix.csv
│       │   └── correlation_<scheme>.csv
│       ├── cluster_stability/            # Step 7 outputs
│       │   ├── coassignment_matrix.csv
│       │   ├── k_selection.csv
│       │   └── linkage_comparison.json
│       ├── human_validation/             # Step 8 outputs
│       │   ├── framework_human_correlation.csv
│       │   ├── cluster_alignment.csv
│       │   └── intra_rater_reliability.json
│       └── consensus/                    # Step 9 outputs
│           ├── representatives.json
│           ├── verdicts.csv
│           ├── coverage.json
│           └── human_agreement.json
│
└── figures/                              # Generated figures (PDF)
```

---

## Evaluation Pipeline Details

### Models Used

| Purpose | Model | Source |
|---------|-------|--------|
| LLM-as-Judge (primary) | `deepseek-chat` (DeepSeek V3) | DeepSeek API |
| LLM-as-Judge (fallback 1) | `llama-3.3-70b-versatile` (Llama 3.3 70B) | Groq API |
| LLM-as-Judge (fallback 2) | `gemini-2.5-flash` (Gemini 2.5 Flash) | Google Gemini API |
| NLI Entailment | `microsoft/deberta-v3-large-mnli` | HuggingFace Transformers |
| BERTScore | `microsoft/deberta-xlarge-mnli` | bert-score library |

### Normalization

All framework scores are min-max normalized per metric to [0, 1]:

```
s_normalized = (s - min(s)) / (max(s) - min(s))
```

This preserves rank ordering within each framework while enabling cross-framework comparison.

### Dual Evaluation Modes

Each LLM-as-judge framework supports two modes:

- **`--use-llm` (production)**: Sends structured prompts via DeepSeek V3 (primary) with Groq/Gemini fallback, parses JSON responses. Deterministic (temperature=0). Requires API keys in `secrets/token.txt`.
- **`--no-llm` (testing)**: Uses heuristic approximations (word overlap, n-gram matching, coverage scores) with small Gaussian noise. No API costs.

NLI-based frameworks (RAGChecker, HALT-RAG) always attempt to load DeBERTa-v3-large locally and fall back to heuristics if unavailable.

---

## Statistical Methods

### Heterogeneity (Meta-Analysis)

Cochran's Q tests whether all 20 frameworks measure the same quantity. One primary metric is selected per framework (faithfulness > factual_consistency > consistency > ...) yielding a 200 x 20 score matrix. The I^2 statistic estimates the proportion of variance due to true between-framework differences.

### Hierarchical Clustering

Ward's method is applied to the 20 x 20 correlation matrix using distance d_jk = sqrt(2(1 - r_jk)). The number of clusters is set to three based on the gap statistic (Tibshirani et al., 2001) and confirmed by silhouette analysis and 1,000-iteration bootstrap co-assignment probabilities. Within-cluster and between-cluster mean correlations are reported.

### Threshold Robustness

Binary agreement is computed at five thresholds (tau in {0.3, 0.4, 0.5, 0.6, 0.7}). Spearman rank correlation between agreement vectors at each tau vs. the tau=0.3 baseline measures stability.

### Human Correlation

Framework scores are correlated (Pearson, Spearman) against binary human faithfulness labels from RAGBench. Only samples with valid human labels are used.

### Mixed-Effects Model

A linear mixed-effects regression models score as a function of domain (fixed effect) with random intercepts per sample:

```
score_ij = B0 + B1(domain_i) + u_j + e_ij
```

This accounts for sample-level variation when estimating domain effects. ICC (intraclass correlation coefficient) is reported.

---

## Practical Recommendations

Based on our analysis, we provide framework selection guidance:

| Use Case | Recommended Framework(s) | Rationale |
|----------|--------------------------|-----------|
| General-purpose scoring | G-Eval or FaithJudge | LLM-as-judge cluster (r = 0.55), lowest domain sensitivity (eta^2 <= 0.039) |
| Claim-level diagnostics | RAGChecker or HALT-RAG | Per-claim verdicts for localizing failures, deterministic outputs |
| Process-level analysis | ReDeEP or LRP4RAG | Interpretability methods reveal how the generator uses retrieved context |
| Multi-perspective reporting | 1 from each cluster | Cross-cluster agreement (r = 0.10) is a stronger signal than within-cluster agreement |

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{neloy2026ragmetrics,
  title={A Meta-Analysis of Evaluation Framework Reliability and Cross-Domain Generalization},
  author={Neloy, Asif Ahmed and Islam, MD Nazmul},
  booktitle={Proceedings of the 39th Canadian Conference on Artificial Intelligence},
  year={2026},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RAGBench](https://arxiv.org/abs/2407.11005) dataset from Galileo AI
- Framework authors for open-source implementations
- DeepSeek for V3 API, Groq for Llama 3.3 70B inference, Google for Gemini API
- HuggingFace for hosting DeBERTa-v3-large-MNLI
