# Annotator Guide: Human Faithfulness Labels for RAG Outputs

This document is written to provide guidance and provide exact details about what you will receive, what you need to
do for each item, how to decide in edge cases, and what you should send back when you
are finished. You do not need to read any research paper before starting. Everything
you need is in this guide.

If after reading this document you still have a question, write it down in the
`notes` column of the CSV for that specific row and continue. Do not guess at
protocol rules that are not written here.

---

## 1. What this task is about, in plain language

A Retrieval-Augmented Generation (RAG) system is a pipeline in which a question is
asked, a search step fetches some reference text (the "context"), and a language
model writes an answer using that context. The question the researchers are asking
is simple: **does the answer actually stick to the context, or does it make things
up?**

Your job is to read 50 such triples and mark each one as either faithful or
unfaithful. You are the ground truth. Automated evaluation tools will be compared
against your labels.

You do not need to judge whether the answer is well written, polite, complete, or
interesting. You only need to judge whether its factual content is supported by the
context it was given.

---

## 2. Core definition: what counts as "faithful"

Use this single rule as your compass:

> **An answer is faithful if and only if every factual claim it makes is directly
> supported by the context. Everything else is unfaithful.**

"Directly supported" means a careful reader can find each claim in the context,
possibly rephrased or compressed, without needing to bring in outside knowledge.

The answer is still faithful if it:

- Paraphrases the context in different words.
- Compresses several sentences of context into one.
- Skips information that is in the context but not asked for.
- Says "I do not know" or "the context does not say" when the context genuinely does
  not contain the answer.
- Uses everyday background words like "a company", "the patient", "in the year",
  which are not themselves facts.

The answer is unfaithful if it:

- States any specific fact, number, date, name, quantity, relationship, cause,
  effect, or event that is not in the context.
- Contradicts the context.
- Invents citations, studies, authors, or sources.
- Adds plausible-sounding medical, financial, or general-knowledge details that are
  not in the context, even if they happen to be true in the real world.
- Answers confidently when the context does not actually contain the answer.

A single unsupported claim is enough to make the whole answer unfaithful. You do not
average, and you do not grade on a curve. Faithfulness is a yes or no property.

---

## 3. What you will receive

You will receive one file:

```
data/human_validation_template.csv
```

It is a plain CSV with 50 rows. You can open it in Excel, Google Sheets, LibreOffice
Calc, or any spreadsheet tool. The columns are:

| Column | Filled already? | Your job |
|---|---|---|
| `id` | Yes | Do not edit. This is the sample identifier. |
| `domain` | Yes | Do not edit. Tells you whether the sample is General Knowledge, Finance, or Biomedicine. |
| `subset` | Yes | Do not edit. Tells you which benchmark the sample comes from. |
| `question` | Yes | Read this. It is the question the RAG system was asked. |
| `context` | Yes | Read this carefully. This is the reference text the system was given. |
| `answer` | Yes | Read this. This is what the system produced. |
| `human_label` | **Empty** | **You fill in `1` for faithful or `0` for unfaithful.** |
| `confidence` | **Empty** | **You fill in `high`, `medium`, or `low`.** |
| `notes` | **Empty** | Optional. Free text. Use it for anything that influenced your decision. |

The 50 samples are balanced across three domains:

- 17 General Knowledge samples (from HotpotQA, MS MARCO, or HAGRID)
- 17 Finance samples (from FinQA or DelucionQA)
- 16 Biomedicine samples (from CovidQA or PubMedQA)

You do not need to know anything about these benchmarks. Every sample is
self-contained. The `context` field gives you all the text you are allowed to use.

---

## 4. How to annotate one sample, step by step

For each row, do the following in order:

1. **Read the question.** Ask yourself what is really being asked.
2. **Read the context.** Read it carefully. Reread any part that is technical. You
   are allowed to take as long as you need.
3. **Read the answer.** Pause between sentences. Do not read it as prose. Read it as
   a list of claims.
4. **Extract the claims.** Mentally list every factual claim the answer makes. A
   claim is anything of the form "X is Y", "X did Y", "X equals Y", "X causes Y",
   "X happened at time T", or "X is located in Y".
5. **Check each claim against the context.** For every claim, find the sentence or
   number in the context that supports it. If you cannot find support for even one
   claim, the answer is unfaithful. Do not use external knowledge. Do not open
   Google. Do not open Wikipedia. Do not open any reference book. If it is not in
   the context, it is unsupported.
6. **Decide and record.** Write `1` in `human_label` if every claim is supported,
   `0` otherwise.
7. **Record your confidence.** Write `high` if the decision was obvious, `medium` if
   you had to think, `low` if you felt unsure even after careful reading.
8. **Optional notes.** If the sample had anything unusual (ambiguous wording, a
   claim that was technically a matter of interpretation, a context that seemed to
   contradict itself), briefly describe it in `notes`. This helps the researchers
   analyse edge cases separately.

Save the file often. At the end, save a copy as
`data/human_validation_filled.csv` and return it.

---

## 5. Decision rules for tricky cases

The rules below are fixed before annotation begins and will not change during the
task. Apply them in order.

### 5.1 The answer contains no factual content

Some answers refuse to answer, restate the question, or say things like "the context
does not provide enough information". If the answer introduces no unsupported facts,
mark it faithful (`1`). A refusal or an admission of ignorance is faithful by
definition.

### 5.2 The answer is partially correct

If one claim is supported and another is not, the answer is unfaithful (`0`).
Faithfulness is not an average.

### 5.3 Numbers and precision

In Finance samples especially, numbers must match the context up to the precision
the context itself gives. If the context says "revenue grew by 12 percent" and the
answer says "revenue grew by 12 percent", that is faithful. If the answer says
"revenue grew by 12.3 percent", that is unfaithful because the extra precision was
invented. Rounding in the other direction, for example from "12.3 percent" in the
context to "about 12 percent" in the answer, is faithful.

### 5.4 Dates, times, and ordering

If the context gives a date, the answer must use that date or a compatible
description (for example, "in the late 1990s" is compatible with "in 1998" only if
the context actually says 1998). If the context does not state the date, the answer
cannot invent one.

### 5.5 Widely known background facts

Even if a fact is common knowledge, the answer must not rely on it unless the
context states it. For example, if the context does not mention that water boils at
100 degrees Celsius, the answer cannot use this fact to draw a new conclusion.
Faithfulness here is strictly about grounding in the provided context.

This rule has one small exception. Common function words, ordinary synonyms, and
standard phrasing do not count as facts. For example, turning "the CEO" in the
context into "the chief executive officer" in the answer is allowed. Turning "the
CEO" into "John Smith" is not allowed unless the context actually names John Smith.

### 5.6 Biomedicine: mechanism, dose, effect direction

Medical claims must be grounded precisely. If the context says "drug X reduced
mortality in patients with condition Y", the answer may say that. The answer may
not say "drug X is a standard treatment for condition Y" unless that phrasing is in
the context. Dose, frequency, direction of effect (increase or decrease), and
patient population must all come from the context.

### 5.7 Finance: claims, entities, and time windows

Claims about company performance, revenue, growth, loss, ratios, debts, or
transactions must match the context. Entity names must match (do not merge or
rename subsidiaries). Time windows must match (a quarterly number is not an annual
number).

### 5.8 General knowledge: named entities and relations

People, places, organisations, dates, and relationships between them must come from
the context. If the context says "A is the capital of B", the answer may repeat
that. The answer may not add that "B borders C" unless the context says so.

### 5.9 The context contradicts itself

Sometimes the context has two sentences that disagree. If the answer picks one of
the two, mark it faithful (`1`) and add a note explaining which sentence it used.
Do not try to resolve the contradiction yourself.

### 5.10 The context is irrelevant to the question

Sometimes the retrieved context does not actually address the question at all. If
the answer says so honestly ("the context does not contain information about X"),
mark it faithful (`1`). If the answer pretends to answer anyway and invents details,
mark it unfaithful (`0`).

### 5.11 When you genuinely cannot decide

Mark your best guess in `human_label`, set `confidence` to `low`, and write in
`notes` what made you uncertain. Do not leave the row blank. The researchers will
analyse low-confidence rows separately.

---

## 6. Worked examples

These examples are illustrative. They are not real samples from the file you will
receive. They are here to show you how to apply the rules.

### Example A: faithful (1)

- **Question.** What causes the greenhouse effect?
- **Context.** "The greenhouse effect occurs when certain gases in the Earth's
  atmosphere, including carbon dioxide and methane, trap heat from the sun. This
  trapped heat raises the planet's surface temperature."
- **Answer.** "The greenhouse effect is caused by gases such as carbon dioxide and
  methane that trap heat from the sun in the atmosphere, which raises surface
  temperature."
- **Label.** `1`. Every claim in the answer is supported by the context.
  Confidence: `high`.

### Example B: unfaithful, invented detail (0)

- **Question.** What causes the greenhouse effect?
- **Context.** "The greenhouse effect occurs when certain gases in the Earth's
  atmosphere trap heat from the sun."
- **Answer.** "The greenhouse effect is caused by gases such as carbon dioxide,
  methane, and water vapour that trap heat from the sun, raising the global
  temperature by roughly 1.1 degrees Celsius since 1900."
- **Label.** `0`. The context does not name carbon dioxide, methane, or water
  vapour, and does not mention the 1.1 degree figure. Even though these facts happen
  to be true in the real world, they are not in the context. The answer invented
  them. Confidence: `high`. Note: "answer introduces gas names and a 1.1 degree
  figure not in the context".

### Example C: unfaithful, numerical drift (0)

- **Question.** What was Company X's revenue growth?
- **Context.** "Company X reported revenue growth of 12 percent year over year in
  fiscal 2023."
- **Answer.** "Company X's revenue grew by 12.4 percent in 2023."
- **Label.** `0`. The answer introduces precision (`.4`) that is not in the context.
  Confidence: `high`.

### Example D: faithful refusal (1)

- **Question.** What is the dosage of drug X for children under 5?
- **Context.** "Drug X has been approved for adults only. Pediatric use has not
  been studied."
- **Answer.** "The context does not give a pediatric dosage for drug X. It only
  states that drug X is approved for adults and that pediatric use has not been
  studied."
- **Label.** `1`. The answer honestly reports that the context does not contain the
  answer, and it does not invent anything. Confidence: `high`.

### Example E: unfaithful, answered anyway (0)

- **Question.** What is the dosage of drug X for children under 5?
- **Context.** "Drug X has been approved for adults only. Pediatric use has not
  been studied."
- **Answer.** "Drug X may be given to children under 5 at a dose of 5 mg per
  kilogram body weight once daily."
- **Label.** `0`. The dose and frequency are invented. The context explicitly says
  pediatric use has not been studied. Confidence: `high`. Note: "answer fabricates
  pediatric dose that is absent and implicitly contradicted".

### Example F: borderline, low confidence

- **Question.** When did the treaty take effect?
- **Context.** "The treaty was signed on 3 June 1998 and came into force after
  ratification by all signatories."
- **Answer.** "The treaty came into effect in 1998."
- **Label.** `1`, confidence `low`. The context says the treaty was signed in June
  1998 but does not explicitly say it came into force in 1998, only that it came
  into force after ratification. A strict reader could argue the answer is
  unfaithful because the year of effect is not stated. A lenient reader could argue
  it is faithful because ratification in the same calendar year is implied by the
  context. Mark your best judgement, set confidence to `low`, and put the reasoning
  in `notes`. The researchers will analyse these separately.

---

## 7. Confidence field

Use the confidence field honestly. The researchers need to know which labels you
were certain about and which were judgement calls. The three levels are:

- `high`. You read the answer and the context once and the decision was obvious.
- `medium`. You had to reread parts of the context or think about it for a moment,
  but you arrived at a clear decision.
- `low`. Even after careful reading, the decision was not obvious. Mark your best
  guess and explain in `notes`.

Do not use `high` for every row just because you want to feel decisive. It is more
useful to the researchers if your confidence levels reflect real uncertainty.

---

## 8. Quality control: the re-annotation subset

Because you are the only annotator, the researchers use a simple form of quality
control called intra-rater agreement. They will ask you to re-annotate 10 of the 50
samples, blind to your original labels, at least 24 hours after you finish the main
pass. This gives an estimate of how consistent you are with yourself over time.

The re-annotation workflow is:

1. Finish and save `data/human_validation_filled.csv` as described in Section 4.
2. Wait at least 24 hours.
3. Open `data/human_validation_reannot.csv` (the researchers will provide this
   file). It contains the same 10 rows as a random subset of your first pass, but
   with `human_label`, `confidence`, and `notes` cleared.
4. **Do not look at your first-pass file during this step.** Relabel the 10 rows
   from scratch using the same rules as before.
5. Save and return both files.

The researchers will compute Cohen's kappa between your two passes. They will not
show you which rows you changed your mind on.

---

## 9. What you send back

When you are done, you should send the following files:

1. `data/human_validation_filled.csv`. This is the 50-row file with all three
   columns filled in (`human_label`, `confidence`, and `notes` where applicable).
2. `data/human_validation_reannot.csv`. This is the 10-row re-annotation file.

Both files are plain CSV. Please do not convert them to Excel format. Please do not
reorder the rows. Please do not add, rename, or delete columns. If your spreadsheet
tool asks whether to keep the CSV format on save, say yes.

If you have a Git account, you can open a pull request. Otherwise, email the two
files to the researcher.

---

## 10. Timeline and effort

- There are 50 samples in the main pass and 10 samples in the re-annotation pass.
- Expect roughly two to five minutes per sample on average, depending on the length
  of the context. Some biomedical and finance samples have longer contexts and may
  take longer.
- Total effort for the main pass is typically between two and four hours. The
  re-annotation pass is typically under one hour.
- Please do not rush. A thoughtful label is more valuable than a fast one. If you
  feel tired, take a break. Fatigued annotation is a known source of noise.

---

## 11. What you should not do

- Do not use any source other than the `context` field. No web searches, no
  textbooks, no prior knowledge about the topic, no other samples in the file.
- Do not discuss specific samples with anyone else before you finish. If you want
  to discuss the protocol itself (not any specific row), send the question in
  writing to the researcher and wait for an answer.
- Do not edit the `id`, `domain`, `subset`, `question`, `context`, or `answer`
  columns. If you spot a typo in the data, note it in `notes` but do not change the
  text.
- Do not leave any `human_label` cell blank. Every row must be labelled `0` or `1`.
- Do not label based on how well written the answer is. A badly written but
  factually supported answer is faithful. A beautifully written but unsupported
  answer is unfaithful.

---

## 12. End result

When this task is finished and returned, the researchers will have:

- 50 binary faithfulness labels produced by a single careful human reader.
- 50 confidence levels.
- A small number of free-text notes on edge cases.
- 10 re-annotation labels used to estimate how consistent you were with yourself.

These labels will be used to compute, for each of the 20 automated evaluation
frameworks compared in the paper, a Pearson and Spearman correlation against your
labels. The paper will report which families of frameworks align best with human
judgement and will use your labels to validate a decision rule for combining
frameworks. Your contribution is the single empirical anchor that moves the paper's
human-validation section from reusing prior numbers to reporting fresh evidence.

---

## 13. Quick checklist before you send the files back

- [ ] All 50 rows have a value of `0` or `1` in `human_label`.
- [ ] All 50 rows have `high`, `medium`, or `low` in `confidence`.
- [ ] Any low-confidence or edge-case row has an explanation in `notes`.
- [ ] The `id`, `domain`, `subset`, `question`, `context`, and `answer` columns are
      unchanged.
- [ ] The file is saved as CSV (not XLSX) at `data/human_validation_filled.csv`.
- [ ] The re-annotation file at `data/human_validation_reannot.csv` is saved and
      was produced at least 24 hours after the main pass, without looking at the
      main pass.

Thank you again. Your careful judgement is the ground truth that the rest of the
analysis depends on.
