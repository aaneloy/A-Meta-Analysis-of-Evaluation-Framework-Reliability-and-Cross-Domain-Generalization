# Human Annotation Protocol for RAG Faithfulness Validation

This document defines the annotation protocol used to produce fresh human faithfulness
labels for a subset of the 200 RAGBench samples evaluated in the meta-analysis. The
resulting labels support the human-validation analysis that addresses reviewer comments
on Section 5.6 of the paper.

## 1. Purpose

The goal is to obtain a small but trustworthy set of human faithfulness judgements that
can be used to:

1. Correlate each of the 20 frameworks against real human decisions.
2. Compare within-cluster and between-cluster alignment with humans.
3. Provide an empirical anchor for the cross-cluster consensus protocol.

This is not intended to replace RAGBench labels at scale. Fifty samples are sufficient
for stable point estimates of framework-human correlation given the binary outcome and
the sample-level bootstrap procedure used in the paper.

## 2. Sample selection

Samples are drawn stratified by domain from `data/samples_200.json`:

- 17 samples from General Knowledge
- 17 samples from Finance
- 16 samples from Biomedicine

Within each stratum, samples are selected to cover the full range of framework
disagreement. Concretely, we rank the 200 samples by the score range
`max - min` across the 20 primary framework scores and pick evenly spaced percentiles
(10, 30, 50, 70, 90) within each domain, then fill the remainder uniformly at random.
This produces a subset that spans both low- and high-disagreement regions, which is
more informative than a uniform random sample of this size.

The selection is deterministic (seed 20260409) so the exact 50 IDs are reproducible
from `scripts/08_human_validation.py --select`.

## 3. Annotation task definition

For each sample the annotator sees:

- The question `q`.
- The retrieved context `X` (a single concatenated passage in RAGBench).
- The generated answer `a`.

The annotator assigns one binary label:

- `1` = **faithful**. Every factual claim in the answer is directly supported by the
  retrieved context. Paraphrasing and compression are allowed. Claims that depend on
  widely known background knowledge but do not contradict the context are allowed
  only if they are not necessary to the answer.
- `0` = **unfaithful**. At least one factual claim in the answer is not supported by
  the retrieved context, contradicts it, or introduces information that cannot be
  verified from the context alone.

Optional fields:

- `confidence` in `{low, medium, high}`.
- `notes` free text, used only for edge cases.

## 4. Decision rules

1. **Answer contains no factual claims** (e.g., pure refusal, question restatement):
   mark faithful (`1`) if it does not introduce unsupported facts.
2. **Partial support**: if any single factual claim in the answer is unsupported,
   label unfaithful (`0`). Faithfulness is not an average.
3. **Numerical precision**: in Finance, numbers must match the context to the
   precision stated. Rounding within the context's stated precision is allowed.
4. **Temporal claims**: dates and ordering must be directly inferable from the
   context.
5. **Biomedical claims**: mechanism, dosage, and effect direction must be stated
   in the context. General medical background not stated in the context counts as
   unsupported.
6. **Abstention / "I don't know"**: label faithful (`1`) if the context genuinely
   does not contain the answer. Label unfaithful (`0`) if the context does contain
   the answer and the system failed to use it.

## 5. Workflow

1. Run `python scripts/08_human_validation.py --select` to produce
   `data/human_validation_template.csv`.
2. Open the CSV in a spreadsheet. Do not edit any column except `human_label`,
   `confidence`, and `notes`.
3. For each row, read the question, context, and answer shown in the adjacent
   columns, apply the rules in Section 4, and enter the label.
4. Save the file as `data/human_validation_filled.csv`.
5. Run `python scripts/08_human_validation.py --analyze` to compute per-framework
   correlations with human labels and per-cluster alignment statistics.

## 6. Quality control

Because the annotator is the paper's sole author, the protocol uses the following
safeguards in place of a second independent rater:

- **Temporal re-annotation.** A random 10-sample subset is re-labelled at least
  24 hours after the first pass, blind to the original labels. Agreement between
  the two passes is reported as intra-rater Cohen's kappa.
- **Pre-registered decision rules.** The rules in Section 4 are fixed before
  annotation begins. Any rule change during annotation requires restarting the
  affected stratum.
- **Edge-case log.** Any sample where the annotator's confidence is `low` is
  logged with a free-text justification in the `notes` field and is analysed
  separately.

These measures are not a substitute for a second rater, and the paper reports
this explicitly in the limitations.

## 7. Reporting

The paper reports:

- Total labelled sample count by domain.
- Intra-rater kappa on the re-annotation subset.
- Low-confidence sample count.
- Per-framework Pearson and Spearman correlation with the human label, with
  95% bootstrap confidence intervals.
- Per-cluster mean human correlation.
- Accuracy of the cross-cluster consensus protocol against the human labels.
