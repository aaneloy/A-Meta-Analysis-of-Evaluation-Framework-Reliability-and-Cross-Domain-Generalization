# Framework Taxonomy

## Classification of RAG Evaluation Frameworks

This document provides a detailed taxonomy of the 9 evaluation frameworks compared in our meta-analysis.

## 1. LLM-as-Judge Frameworks

### Characteristics
- Use large language models to score outputs
- Flexible and human-aligned
- Expensive (API costs) and non-deterministic
- Can evaluate subjective qualities

### Frameworks

#### RAGAS (Es et al., EACL 2024)
- **Mechanism**: Decomposes evaluation into structured sub-tasks
- **Faithfulness**: Extracts claims from answer, verifies each against context
- **Answer Relevancy**: Generates synthetic questions, compares embeddings
- **Context Metrics**: Checks precision/recall of retrieved passages
- **Strengths**: Comprehensive, well-documented
- **Weaknesses**: Multiple API calls per sample

#### DeepEval (Confident AI)
- **Mechanism**: Similar to RAGAS with stricter rubrics
- **Faithfulness**: More stringent grounding requirements
- **Hallucination**: Explicit inverse of faithfulness
- **Strengths**: Enterprise-ready, good documentation
- **Weaknesses**: Proprietary aspects

#### G-Eval (Liu et al., EMNLP 2023)
- **Mechanism**: Chain-of-thought prompting with form-filling
- **Process**: LLM reasons through evaluation criteria step-by-step
- **Output**: Probability-weighted scores
- **Strengths**: Better human alignment than traditional metrics
- **Weaknesses**: Requires powerful LLM (GPT-4)

---

## 2. NLI-based Frameworks

### Characteristics
- Use Natural Language Inference models
- Check entailment relationships
- Fast and interpretable
- Limited to factual consistency

### Frameworks

#### RAGChecker (Ru et al., NeurIPS 2024)
- **Mechanism**: Claim-level entailment verification
- **Process**: 
  1. Decompose answer into atomic claims
  2. Check each claim against context via NLI
  3. Aggregate claim-level scores
- **Metrics**: Claim recall, faithfulness, noise sensitivity
- **Strengths**: Fine-grained diagnosis
- **Weaknesses**: Claim extraction can be noisy

#### QAFactEval (Fabbri et al., NAACL 2022)
- **Mechanism**: Question generation + answering
- **Process**:
  1. Generate questions from summary
  2. Answer from both summary and source
  3. Compare answers for consistency
- **Strengths**: Intuitive interpretation
- **Weaknesses**: Depends on QG/QA model quality

---

## 3. Embedding-based Frameworks

### Characteristics
- Use neural embeddings for similarity
- No API calls required
- Fast inference
- May miss logical/factual errors

### Frameworks

#### BERTScore (Zhang et al., ICLR 2020)
- **Mechanism**: Token-level embedding similarity
- **Process**:
  1. Get contextual embeddings for all tokens
  2. Compute pairwise cosine similarity
  3. Greedy matching for P/R/F1
- **Strengths**: Handles synonyms, paraphrases
- **Weaknesses**: Doesn't verify factual accuracy

---

## 4. Classifier-based Frameworks

### Characteristics
- Use fine-tuned discriminative models
- Deterministic and fast
- Requires training data
- Domain-specific performance

### Frameworks

#### ARES (Saad-Falcon et al., NAACL 2024)
- **Mechanism**: Fine-tuned DeBERTa-v3 classifiers
- **Training**: Binary classification on labeled examples
- **Inference**: Prediction-Powered Inference for CIs
- **Metrics**: Context relevance, answer faithfulness/relevance
- **Strengths**: No LLM API needed, statistically rigorous
- **Weaknesses**: Needs labeled data for new domains

---

## 5. Hybrid Frameworks

### Characteristics
- Combine multiple signal types
- More comprehensive coverage
- Complex implementation
- Harder to debug failures

### Frameworks

#### UniEval (Zhong et al., EMNLP 2022)
- **Mechanism**: Boolean QA formulation
- **Process**: "Is this text [dimension]?" → P(Yes)
- **Dimensions**: Coherence, consistency, fluency, relevance
- **Strengths**: Unified framework for multiple dimensions
- **Weaknesses**: May oversimplify nuanced qualities

#### TRACe (Friel et al., 2024)
- **Mechanism**: Token-level annotation
- **Annotations**: Utilization, Relevance, Adherence, Completeness
- **Granularity**: Most fine-grained of all frameworks
- **Strengths**: Detailed diagnostic information
- **Weaknesses**: Annotation is expensive

---

## Taxonomy Summary Table

| Category | Framework | Granularity | API Required | Speed | Best For |
|----------|-----------|-------------|--------------|-------|----------|
| LLM-as-Judge | RAGAS | Claim | Yes | Slow | General evaluation |
| LLM-as-Judge | DeepEval | Claim | Yes | Slow | Enterprise use |
| LLM-as-Judge | G-Eval | Document | Yes | Slow | Human-aligned scores |
| NLI-based | RAGChecker | Claim | Optional | Medium | Hallucination diagnosis |
| NLI-based | QAFactEval | Fact | Optional | Medium | Factual consistency |
| Embedding | BERTScore | Token | No | Fast | Semantic similarity |
| Classifier | ARES | Document | No | Fast | Production deployment |
| Hybrid | UniEval | Document | No | Medium | Multi-dimensional |
| Hybrid | TRACe | Token | No | Fast | Detailed diagnosis |

---

## Implications for Practitioners

### When to use LLM-as-Judge
- Prototype evaluation
- Subjective quality assessment
- When budget allows API costs

### When to use NLI-based
- Factual consistency is primary concern
- Need interpretable claim-level results
- Moderate computational budget

### When to use Embedding-based
- High throughput requirements
- Semantic similarity sufficient
- No API budget

### When to use Classifier-based
- Production deployment
- Deterministic results required
- Have domain-specific training data

### When to use Hybrid
- Comprehensive evaluation needed
- Debugging system failures
- Research benchmarking
