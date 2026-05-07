# paper/CHANGES.md

Diff log for two consistency passes on `paper/aerograph.tex`. The
paper's structure and arguments are unchanged. Only factual drift
between the paper text and the demo-bundle artifacts, plus
proofreading-level fixes, are addressed.

## Proof pass (2026-05-06)

Goal of this pass: resolve the unresolved $\tau$ contradiction left by
the gate-6 footnote, do a typo / broken-citation sweep, and verify
that the table numbers still match the published JSONs.

### `paper/aerograph.tex`

| Line(s) | Change | Reason |
|---|---|---|
| 543--552 | Replaced the "$\tau = -0.07$ ($p = 1.00$)" claim plus its long footnote with a body sentence that reports $\tau = -0.138$ ($p = 0.702$) directly, and a parenthetical methods note that earlier internal numbers gave $\tau \approx -0.07$ under rank-aggregation choices we could not reproduce. The qualitative reading is preserved as "system rankings show no statistically significant correlation across corpora" rather than the now-incorrect "$p = 1.00$". | The re-derivation in `demo/eval/swap_corpus.py` is the ground truth: it reads the published `paper/results/eval_results_*.json` files and computes $\tau$ over the six-system P@10 vector with `scipy.stats.kendalltau`. Carrying both numbers, one in the body and one in a footnote, advertised an internal contradiction we could not close. The paper now carries the reproducible scalar in the body and acknowledges the older number as a methods caveat. |
| 641 (figure caption for `fig:crosscorpus`) | "Kendall $\tau = -0.07$" with reference to the footnote replaced by "Kendall $\tau = -0.138$, $p = 0.702$". | Caption matches the body. |
| 582 | `$++0.090$` typo replaced with `$+0.090$`. | Was a stray plus sign. |
| 169 | Citation key `\citep{icao2018doc9859}` replaced with `\citep{icao2016doc9859}` to match the entry in `references.bib` (which is named `icao2016doc9859` despite the title's 2018 4th-edition year). | Broken citation, would have rendered as `[?]`. |

### `paper/references.bib`

Added three entries that `paper/aerograph.tex` cites but were missing
from the bibliography file:

- `reimers2019sbert` -- Sentence-BERT embedding model paper. Used at line 291 in the dense-vector section.
- `page1999pagerank` -- Original PageRank tech report. Used at line 304 in the PPR section.
- `robertson2009bm25` -- BM25 probabilistic relevance framework. Used at line 297 in the BM25 section.

Without these, `bibtex` would have left the three citations as `[?]`
in the rendered PDF.

### Numbers verified against `paper/results/`

| Table | Numbers checked | Source |
|---|---|---|
| Table 1 (`tab:retrieval`) | P@10 / nDCG@10 for all six systems | `eval_results_claude_judge.json` `metrics` block. All six P@10 values match to three decimal places (`baseline 0.114`, `bm25 0.106`, `graph_only 0.058`, `graphrag 0.152`, `ppr_only 0.200`, `hybrid_4way 0.210`). |
| Table 2 (`tab:answer-quality`) | Ollama faithfulness column | Same JSON. `hybrid_4way` rounds to `0.685`, `baseline` to `0.608`, etc. Matches. |
| Table 5 (`tab:cross-corpus`) | Held-out P@10 and faithfulness | `eval_results_heldout.json`. All six systems match to three decimals. |
| Kendall $\tau$ scalar | Re-derived | `python -c "from scipy.stats import kendalltau; print(kendalltau([0.114,0.106,0.058,0.152,0.200,0.210], [0.028,0.028,0.138,0.102,0.084,0.070]))"` returns $\tau = -0.1380$, $p = 0.7021$. Matches the new body number. |

### Numbers deliberately left as-is

The abstract reports the raw graph as 29,244 entities and 43,505
relations, which corresponds to `data/graphs/aerograph.pkl.bak` (a
specific extraction snapshot). The current `data/graphs/aerograph.pkl`
is 30,237 / 54,980. The author's instruction was to leave the abstract
number unchanged; it documents a specific run, and Section 3.5 line
271 ("Raw extraction produces 29,244 entities and 43,505 relations")
agrees with the abstract. The canonicalized counts (23,948 / 48,479)
and chunk count (4,710) reproduce throughout the paper.

The abstract's "$\tau \approx 0$" framing is consistent with the new
$-0.138$ scalar and was not edited.

## Gate-6 pass (earlier)

## What was changed

### `paper/abstract.md`

| Field | Before | After | Reason |
|---|---|---|---|
| Ollama judge model | `qwen2.5:7b` | `llama3.1:8b` | The actual `_call_judge()` in `src/aerograph/eval.py:179` calls `llama3.1:8b`, and `paper/aerograph.tex` already documents this in three places (lines 119, 380, 459). The abstract was the lone holdout. |
| Raw graph counts | `The raw extracted graph contains 29,244 entities and 43,505 relations; after taxonomy-based entity resolution and cleanup the evaluation graph contains 23,948 canonicalized entities and 48,479 relations.` | `After taxonomy-based entity resolution and cleanup the evaluation graph contains 23,948 canonicalized entities and 48,479 relations.` | Per gate-6 user instruction: every public-facing surface should report the same canonical 23,948 / 48,479. The raw-vs-cleaned framing distracted from the consistent number that downstream readers actually care about. The .bak file (29,244 / 43,505) still exists on disk if anyone wants to reproduce the pre-cleanup snapshot, but the abstract no longer cites those numbers. |

### `paper/aerograph.tex`

Added a footnote anchored to the τ = -0.07 claim around line 545
documenting the τ discrepancy. Added a parenthetical to the figure
caption at line 632 referencing the same finding. Verbatim:

> An independent re-derivation in `demo/eval/swap_corpus.py` that
> reads the published `paper/results/eval_results_*.json` files and
> computes Kendall's τ over the same six-system P@10 vector returns
> τ = -0.138 (p = 0.702). The per-system P@10 values reproduce
> exactly to three decimal places, so the qualitative claim is
> unchanged, but the scalar disagrees. We log the full probe trail
> (alternate metrics, system-set ablations, tie-handling variants)
> in `demo/eval/swap_corpus_diagnosis.md` and treat the discrepancy
> as an open methodological question.

This is the only numerical change to the .tex file. The paper still
claims τ = -0.07 in its main text; the footnote signals that the
re-derived value is -0.138 and that the qualitative reading is
unaffected.

## What was deliberately not changed

### Note on the raw graph count decision

An earlier draft of this CHANGES.md left the abstract's raw-extraction
counts (29,244 / 43,505) in place, on the rationale that they
reproduce from `data/graphs/aerograph.pkl.bak` and document a specific
run. At gate-6 final review the user reversed that decision: every
public-facing surface should report the same canonical 23,948 /
48,479. The abstract was edited accordingly (see the table at the top
of this file). The .bak file remains on disk for anyone who wants to
reproduce the pre-cleanup snapshot, but the abstract no longer cites
those numbers.

### Test count

The paper does not claim a test count, so there is nothing to align.
For the record, the live test suite passes 139/139 in 16.30 s
(`pytest tests/ -q`); the deprecated count "135" appeared in
`app_full.py`'s About tab and was already flagged in
`demo/STATUS.md` (not edited in this gate to keep `src/` untouched
unless a deliverable required it).

### Chunk count

`paper/aerograph.tex:292` already says 4,710 chunks. Matches live
`data/chroma_db` collection `aerograph_chunks` exactly. No change.

### `paper/methods.md`, `paper/related_work.md`, `paper/results_table.md`

Read; no numerical drift identified relative to the demo bundle.
`results_table.md` line 21 gives Graph-only P@10 = 0.058, which
matches both `paper/results/eval_results_claude_judge.json` and
`demo/eval/swap_results.json`. No change.

### Two τ ≈ 0 references at the end of the related/conclusion sections

`paper/aerograph.tex:59` says `Kendall $\tau \approx 0$` without a
specific number. The "≈ 0" framing covers both -0.07 and -0.138, so
the qualitative claim still stands. No change. If the paper goes
through another revision, swapping that "≈ 0" for "the rank
correlation is small and not significantly different from zero
under any of the metric variants we probed" would be more durable;
out of scope for this consistency pass.

## Verification

```bash
# Re-run the swap-corpus reproduction; both numbers will print.
python demo/eval/swap_corpus.py
# τ (Kendall): -0.1380  (paper -0.07, |Δ| 0.0680)
# Per-system table reproduces exactly to 3 decimal places.

# Confirm the abstract change took.
grep "llama3.1:8b\|qwen2.5:7b" paper/abstract.md
# llama3.1:8b expected; qwen2.5:7b should not appear.

# Confirm the footnote landed.
grep -A2 "demo/eval/swap_corpus.py" paper/aerograph.tex
```

## Files touched

- `paper/abstract.md` (one-word change)
- `paper/aerograph.tex` (footnote + caption parenthetical)

## Files not touched but flagged for the next paper revision

- The "≈ 0" framing in `paper/aerograph.tex:59`. Stronger to anchor
  the claim with a metric label.
- The "62% pairwise blind comparisons" claim in `paper/abstract.md`
  was not re-verified against the published comparative JSONs. If
  that becomes a load-bearing claim in a future writeup, recompute
  from `paper/results/eval_results_claude_judge.json` first.
