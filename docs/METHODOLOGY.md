# Methodology

## Research Design

This study employs a systematic meta-analysis approach to compare RAG evaluation frameworks. We evaluate **9 state-of-the-art frameworks** on **200 samples** from **3 domains**.

## Evaluation Frameworks

### 1. RAGAS (Es et al., EACL 2024)
**Type:** LLM-as-Judge

RAGAS uses GPT-4 to evaluate RAG outputs through structured prompts:
- **Faithfulness:** Decomposes answers into claims, verifies each against context
- **Answer Relevancy:** Generates synthetic questions from answer, compares to original
- **Context Precision:** Checks if retrieved passages help answer the question
- **Context Recall:** Verifies context covers ground truth information

### 2. DeepEval (Confident AI)
**Type:** LLM-as-Judge

DeepEval provides enterprise-grade evaluation with:
- Stricter faithfulness checking with detailed rubrics
- Explicit hallucination detection (inverse of faithfulness)
- Chain-of-thought reasoning for scoring

### 3. RAGChecker (Ru et al., NeurIPS 2024)
**Type:** Claim-Level NLI

RAGChecker differs by using claim-level entailment:
- Decomposes answers into atomic claims
- Checks each claim via NLI model against context
- Provides fine-grained diagnosis of hallucinations

### 4. TRACe (Friel et al., 2024)
**Type:** Token-Level Annotation

TRACe provides the most fine-grained evaluation:
- Token-level labels for utilization, relevance, adherence
- Comprehensive annotations from RAGBench
- Used as pseudo ground truth in our analysis

### 5. ARES (Saad-Falcon et al., NAACL 2024)
**Type:** Fine-tuned Classifier

ARES trains DeBERTa-v3 classifiers:
- Binary classifiers for context relevance, answer faithfulness
- Prediction-Powered Inference for confidence intervals
- Does not require LLM API calls

### 6. G-Eval (Liu et al., EMNLP 2023)
**Type:** GPT-4 with CoT

G-Eval adapts GPT-4 for NLG evaluation:
- Chain-of-thought prompting for reasoning
- Form-filling paradigm for structured scores
- Better human alignment than traditional metrics

### 7. BERTScore (Zhang et al., ICLR 2020)
**Type:** Embedding Similarity

BERTScore uses contextual embeddings:
- Computes token-level similarity using BERT
- Handles synonyms and paraphrases
- Provides precision, recall, F1

### 8. UniEval (Zhong et al., EMNLP 2022)
**Type:** Boolean QA

UniEval unifies multiple dimensions:
- Formulates evaluation as Boolean questions
- Uses T5-based model to answer yes/no
- Score = probability of "Yes"

### 9. QAFactEval (Fabbri et al., NAACL 2022)
**Type:** QA-based Verification

QAFactEval uses question generation and answering:
- Generates questions from summary
- Answers from both summary and source
- Compares answers for consistency

## Dataset: RAGBench

We sample from RAGBench (Friel et al., 2024), which provides:
- 200 total samples (stratified by domain)
- Token-level TRACe annotations
- Question, context, answer, ground truth

### Domain Distribution
| Domain | Subsets | Samples |
|--------|---------|---------|
| General Knowledge | HotpotQA, MS MARCO, HAGRID | 67 |
| Finance | FinQA, DelucionQA | 67 |
| Biomedicine | CovidQA, PubMedQA | 66 |

## Statistical Analysis

### Correlation Analysis
- **Pearson correlation:** Linear relationship between scores
- **Spearman correlation:** Rank-order relationship
- **Kendall's tau:** Ordinal association
- **Bootstrap CIs:** 1000 iterations for 95% confidence intervals

### Agreement Metrics
- **Percent agreement:** Fraction of matching binary classifications
- **Cohen's Kappa:** Agreement corrected for chance

### Heterogeneity Statistics
- **Cochran's Q:** Test for heterogeneity across frameworks
- **I² statistic:** Percentage of variability due to heterogeneity

### Domain Analysis
- **One-way ANOVA:** Test for domain effect
- **Effect size (η²):** Proportion of variance explained
- **Pairwise t-tests:** Compare specific domains
- **Cohen's d:** Effect size for domain comparisons

## Implementation Notes

### Heuristic Approximations

For frameworks requiring LLM API calls, we provide heuristic approximations:
- Word/n-gram overlap as proxy for semantic similarity
- Coverage scores for faithfulness
- Claim extraction via sentence splitting

These approximations are suitable for:
1. Understanding framework behavior patterns
2. Comparing relative rankings
3. Testing methodology before API deployment

For production use, deploy with actual framework libraries and LLM backends.

### Noise Simulation

Each framework adds small Gaussian noise (σ = 0.03-0.06) to simulate:
- LLM response variability
- Framework-specific scoring tendencies
- Real-world evaluation noise

## Reproducibility

```bash
# Set random seed
RANDOM_SEED=42

# Run full pipeline
python scripts/01_prepare_dataset.py
python scripts/02_run_evaluation.py
python scripts/03_analyze_results.py
python scripts/04_generate_figures.py
python scripts/05_failure_analysis.py
```

All random operations use the configured seed for reproducibility.
