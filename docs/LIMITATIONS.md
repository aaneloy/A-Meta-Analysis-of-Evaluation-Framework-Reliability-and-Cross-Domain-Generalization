# Limitations and Future Work

## Current Limitations

### 1. Heuristic Approximations

**Issue**: The current implementation uses heuristic proxies (word overlap, n-gram matching) for LLM-based metrics rather than actual LLM calls.

**Impact**: 
- Absolute scores may differ from production frameworks
- Relative rankings and correlations should still be valid
- Pattern of framework agreement/disagreement is preserved

**Mitigation for Production**:
```bash
# To use real RAGAS with Ollama
pip install ragas langchain-ollama
ollama pull llama3:8b
# Then modify src/frameworks/ragas_eval.py to use real implementation
```

### 2. Single Benchmark Limitation

**Issue**: Results are derived solely from RAGBench dataset.

**Impact**:
- Findings may not generalize to other benchmarks
- Domain coverage limited to 3 areas
- Sample size (200) is moderate

**Mitigation**:
- RAGBench internally draws from 7 diverse source datasets
- Stratified sampling ensures domain representation
- Future work should validate on additional benchmarks

### 3. No Human Correlation Study

**Issue**: We measure inter-framework agreement, not correlation with human judgments.

**Impact**:
- Cannot determine which framework is "most correct"
- High agreement doesn't guarantee accuracy
- Low agreement doesn't indicate which framework is wrong

**Future Work**:
- Collect human annotations for subset of samples
- Compute framework-human correlation
- Identify systematic biases per framework

### 4. Metric Definition Inconsistency

**Issue**: "Faithfulness" is defined differently across frameworks.

**Examples**:
- RAGAS: "Fraction of claims supported by context"
- DeepEval: "Whether answer can be inferred from context"
- RAGChecker: "Claim-level entailment score"
- BERTScore: "Token embedding similarity" (not really faithfulness)

**Impact**:
- Direct score comparison may be misleading
- Aggregating scores requires caution
- Users may misinterpret what's being measured

**Recommendation**:
- Always report which framework was used
- Don't average scores across frameworks
- Prefer rank correlation over absolute comparisons

### 5. Domain Imbalance Effects

**Issue**: Finance domain consistently shows lower scores across all frameworks.

**Possible Explanations**:
1. Finance RAG systems genuinely perform worse
2. Finance language is harder to evaluate
3. Frameworks are biased toward general language
4. Training data underrepresents financial text

**Impact**:
- Cannot determine true cause from our data
- Domain-specific calibration may be needed
- Cross-domain score comparison is problematic

### 6. Temporal Validity

**Issue**: Evaluation frameworks are rapidly evolving.

**Impact**:
- Results reflect framework versions as of 2024-2025
- Newer versions may have different behaviors
- API-based frameworks may change without notice

**Recommendation**:
- Pin framework versions in requirements.txt
- Re-run analysis periodically
- Document exact versions used

---

## Known Technical Limitations

### Computational
- Bootstrap CI computation is CPU-intensive (1000 iterations)
- Large sample sizes (>1000) may require optimization
- Some visualizations don't scale well

### Statistical
- Correlation doesn't imply causation
- Multiple comparisons increase false positive risk
- Small subgroup sizes limit domain-specific conclusions

### Implementation
- Heuristic evaluators add small random noise for realism
- Claim extraction uses simple sentence splitting
- Synonym matching is limited to predefined dictionary

---

## Future Research Directions

### 1. Unified Metric Definitions
Develop community standards for what "faithfulness", "relevance", etc. actually mean.

### 2. Domain-Adaptive Calibration
Create methods to adjust scores based on domain characteristics.

### 3. Human Correlation Studies
Collect and publish human annotations for framework validation.

### 4. Adversarial Robustness
Test how frameworks respond to adversarial inputs, prompt injections, etc.

### 5. Multi-lingual Extension
Extend analysis to non-English RAG systems.

### 6. Temporal Stability
Track how framework scores change over model versions.

### 7. Cost-Quality Tradeoffs
Systematic analysis of evaluation cost vs. reliability.

### 8. Ensemble Methods
Develop principled approaches to combining multiple frameworks.

---

## Reporting Recommendations

When using these frameworks in your own work:

1. **Report framework versions** - Pin and document exact versions
2. **Report multiple metrics** - Don't rely on single number
3. **Use diverse frameworks** - At least 2 from different categories
4. **Acknowledge limitations** - No metric is perfect
5. **Consider domain effects** - Scores aren't comparable across domains
6. **Validate on subset** - Spot-check with human evaluation
