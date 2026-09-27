# Paper 1 (VMD-MFGNN) — Compression & Finalization Best-Practices Reference

Purpose: this document is a sourced reference for the agent(s) who will compress and
finalize `manuscript/main.tex`. It is not an edit — no changes were made to the paper.
All length/norm claims below are cited to a real source; where a source could not be
retrieved (several journal guide-for-authors pages returned HTTP 403 or an
authentication redirect during research), that is stated explicitly rather than
papered over with an invented number, in keeping with the paper's own standard of
disclosure.

## 0. What the current manuscript actually is (measured, not estimated)

Read in full (`manuscript/main.tex`, 1134 lines) before writing this guide.

- **Format**: `\documentclass[journal]{IEEEtran}`, two-column IEEE journal style.
  Line 24–28 contains the authors' own TODO: target venue is undecided; the file
  was originally drafted for *Resources Policy* under `elsarticle` and later
  reformatted to IEEEtran. **This choice is still open** and matters for every
  number below, since IEEE two-column and Springer/Elsevier single-column pages
  hold very different amounts of text per page.
- **Length**: ~30,500 raw words in the `.tex` source (body text + captions +
  in-line LaTeX markup, comments stripped; measured directly, not estimated),
  across 37 typeset PDF pages per the task brief. 55 references
  (`\bibitem`), all inline via `thebibliography` (no external `.bib`).
- **Structure** (real section labels, from `\section`/`\subsection` greps):
  Introduction (`sec:intro`) → Related Work (`sec:related`, 4 subsections) →
  Methodology (`sec:method`, 6 subsections) → Experimental Setup
  (`sec:experiments`, 6 subsections) → Results and Discussion (`sec:results`,
  9 subsections including Main Results, DM significance, Ablation, Confound-Check,
  Interpretability [Collapse + Mode Importance], Graph-Fix Experiment, Robustness
  Checks, Constant-Forecast Artifact, Discussion, Economic Significance) →
  Limitations (`sec:limitations`, 16 enumerated items) → Conclusion
  (`sec:conclusion`) → Data/Code Availability, CRediT, Funding, Conflict of
  Interest → Bibliography. **No appendix currently exists** (`grep` for
  `\appendix` returns nothing) — every diagnostic, every caveat, and every
  robustness check is in the main body today.
- **Abstract**: one paragraph, lines 51–53, **471 words** (measured by direct
  word count on the `\begin{abstract}...\end{abstract}` span). This is a hard,
  measured number, not an impression — see §1 for why it is the single most
  actionable finding in this document.
- **Tables**: 10 (`tab:related`, `tab:notation`, `tab:hpo-space`, `tab:main`,
  `tab:r2`, `tab:dm`, `tab:ablation-config`, `tab:ablation`, `tab:confound-tuned`,
  `tab:hpo-wide-space`). **Figures**: 9 (`fig:vmd`, `fig:architecture`,
  `fig:datasplit`, `fig:comparison`, `fig:perband-vs-pooled`, `fig:graphs`,
  `fig:graphs-nodelink`, `fig:collapse`, `fig:attention`), plus one algorithm
  block (`alg:vmdmfgnn`).
- **Single densest block**: Section `sec:results` (Results and Discussion) runs
  from line 380 to line 940 — **560 of 1134 lines, essentially half the
  document** — and within it, `sec:graphfix` + `sec:robustness` +
  `sec:constantforecast` alone (lines 649–918) run **270 lines, ~24% of the
  whole manuscript**, for what is a single diagnostic narrative (fix the
  collapse mechanically, sweep four robustness axes, catch a constant-forecast
  artifact).

## 1. Typical/acceptable length for this class of paper (sourced)

**Headline number to use for the compression target, derived from the
manuscript's own measured density (below), not guessed: total manuscript
~11,000–13,000 raw words (roughly 10,000–12,000 words of body text once the
bibliography's ~900 words are excluded), corresponding to 13–16 pages if the
current IEEE two-column format is kept.** Derivation and sourcing below.

**Measured density check.** The current file is 30,628 raw words (whitespace
split on the `.tex` source, comments stripped; measured directly) across 37
typeset PDF pages, i.e. **≈828 raw words per typeset page** at this
manuscript's own layout. Applying that same density to the IEEE Transactions
caps in §1a gives the word targets above: 13 pp × 828 ≈ 10,764 words; 16 pp ×
828 ≈ 13,248 words. This is why the target is stated as a range keyed to page
count, not a single round number — it is a direct algebraic consequence of
this file's own measured density, not a separate estimate.

### a. IEEE Transactions family (closest match to the *current* file format)

The manuscript is already typeset as `IEEEtran journal` two-column. Page caps
below: the IEEE Trans. Automatic Control figure was fetched directly from the
source page; the remaining rows come from a web-search summary of the
corresponding IEEE society author-info pages (the underlying pages were not
independently re-fetched in this pass, so treat those five rows as
search-summary-sourced, one level less direct than the TAC row).

| Venue | Typical / cap | Source | How obtained |
|---|---|---|---|
| IEEE Trans. Automatic Control | ~12 pp typical ("normally around 12 pages"), 16 pp hard max (incl. bios); $125/page charge past 12, max 4 extra pages | ieeecss.org/publication/transactions-automatic-control/author-info | fetched directly |
| IEEE Trans. Networking | 16 pp max, two-column Transactions format | comsoc.org/publications/journals/ieee-tnet/ieee-transactions-networking-author-guidelines | search summary |
| IEEE Trans. Communications | ≤13 double-column, single-spaced pages | comsoc.org/publications/journals/ieee-tcom/policies-and-guidelines | search summary |
| IEEE Trans. Signal Processing | ≤13 double-column pages, 10-pt font | signalprocessingsociety.org/publications-resources/information-authors | search summary |
| IEEE Trans. Cybernetics | ~10 Transactions pages typical, "or shorter" | ieeesmc.org/publications/transactions-on-cybernetics/information-for-authors-3 | search summary |
| IEEE Trans. Antennas & Propagation | 1–16 pp; manuscripts over 16 pp returned for revision | ieeeaps.org/ieee-tap/for-authors/how-to-prepare-your-submission | search summary |

Consistent pattern across every IEEE Transactions venue checked: **10–13
two-column pages is the norm, 16 pages is the practical ceiling**, and most
societies charge mandatory overlength fees past that. **At 37 pages, this
manuscript is currently 2.85× the strictest cap checked (13 pp, TCom/TSP) and
2.31× even the most permissive one (16 pp, TNet/TAC-max/TAP-max)**, in the
very format it is already typeset in. This is the single most direct,
least-interpretive piece of evidence that substantial compression is required
regardless of which venue is eventually chosen.

### b. Computational Economics (Springer) — the named comparable venue

Attempted to retrieve `link.springer.com/journal/10614/submission-guidelines`
directly (the authoritative source): the page redirected to an
authentication/IDP endpoint and could not be fetched without login. **No
explicit word/page limit for Computational Economics could be independently
verified in this research pass** — this gap is disclosed rather than
papered over with an invented number, per this document's own instructions.

The directly comparable paper this task names — Padhan, Senthil, Panigrahi &
Sahoo, "An Optimized VMD-Based Deterministic and Probabilistic Crude Oil Price
Forecasting Method Employing Machine Learning Models," *Computational
Economics* (Springer), published online 2026-09-18, DOI
`10.1007/s10614-026-11436-2` — was confirmed to exist via Crossref
(`api.crossref.org/works/10.1007/s10614-026-11436-2`) and via its SSRN
preprint (`papers.ssrn.com/sol3/papers.cfm?abstract_id=5397480`), same
research group (NIT Rourkela), same broad topic. Its Crossref record carries
**56 references** but **no page-range field** (the article had not yet been
assigned to a print issue at the time of this research), and the SSRN page's
"Number of Pages in PDF File" field could not be retrieved (403 response).
**This specific number is therefore unverified and should not be quoted as a
fact** — only the paper's existence, venue, and reference count (56, in the
same range as this manuscript's own 55) are confirmed.

What can be said with a real, directly measured source instead: two other
recent *Computational Economics* papers in the same VMD-for-commodity-price
subfield as this manuscript's own cited work were confirmed via the Crossref
API (`api.crossref.org/works/<DOI>`), which returns each article's actual
assigned page range:

- Tseng/Nguyen-adjacent comparable, "Forecasting Crude Oil Prices: Evidence
  From WOA-VMD-FE-Transformer Model," *Computational Economics*, DOI
  `10.1007/s10614-025-10861-z` — **pages 4645–4676, i.e. 32 typeset pages**
  (Crossref `page` field, fetched directly).
- "A Crude Oil Price Forecasting Model Based on Local Mean Decomposition,
  Marine Predators Algorithm and Least Squares Support Vector Regression,"
  *Computational Economics*, DOI `10.1007/s10614-025-10946-9` — **pages
  2823–2848, i.e. 26 typeset pages** (Crossref `page` field, fetched
  directly).

Both are real, single-column Springer typeset page counts for the exact
subfield and venue this task names as the primary comparable. **32 and 26
pages, single-column Springer typesetting, is meaningfully longer per page
than an IEEE two-column page but is not an unbounded-length venue either** —
Springer single-column pages hold noticeably fewer words per page than IEEE
two-column pages (larger type, single column, more white space), so a
26–32-page single-column Computational Economics article is roughly
comparable in total word count to the 13–16-page IEEE two-column range
computed in §1 above, not larger. This is corroborating rather than
independent evidence for the same ~10,500–13,000-word range, and is the
strongest available real-comparable-paper evidence for Computational
Economics specifically, since the journal's own stated numeric limit could not
be retrieved (auth-walled, see below).

### c. Elsevier venues in the same space (Expert Systems with Applications,
Resources Policy) — the paper's own prior target

A direct WebFetch of ESWA's own Guide for Authors page
(`sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors`)
returned HTTP 403 (ScienceDirect blocks automated fetches on this route in
this environment); the two numbers below come from a web-search summary of
that same page's indexed content, not a direct fetch, and should be
re-verified from a browser before being quoted as a hard submission
requirement:

- **Abstract: ≤250 words.**
- **Highlights: 3–5 bullet points, each ≤85 characters including spaces.**

**This paper's abstract is 471 words — essentially double any plausible
target-venue abstract limit, and specifically 1.9× ESWA's stated 250-word
cap.** This is the single highest-confidence, lowest-effort compression
target in this entire document: it is short, it is currently one giant
undifferentiated paragraph, and every plausible target venue caps abstracts
well under its current length. Treat abstract compression to ~200–300 words as
non-negotiable regardless of which venue is ultimately chosen.

Resources Policy's own Guide for Authors page
(`sciencedirect.com/journal/resources-policy/publish/guide-for-authors`) could
not be fetched directly in this research pass (403 response, ScienceDirect
blocks automated fetches on this route). This manuscript's own header comment
(line 24–28) says it was *originally* formatted for Resources Policy, so if
that venue is reconsidered, re-attempt this fetch from a browser before
submission rather than relying on this document. What can be said with a real
source: Elsevier's *Your Paper Your Way* program (confirmed via
`elsevier.com/subject/next/guide-for-authors`) generally does not impose a
strict page count at initial submission but expects a "concise and complete"
full-length article. **No specific word count for Resources Policy or for the
Becerra/Luo/Nabavi papers this manuscript itself cites from that journal was
independently verified in this research pass** — do not quote a Resources
Policy word count without re-checking it; this gap is disclosed rather than
filled with an unverified estimate.

### d. arXiv preprint norm

Confirmed directly from arXiv's own help pages (`info.arxiv.org/help/index.html`
and the linked size-limit help file): **arXiv imposes no page or word limit at
all** — only a 50MB total submission size cap (a 2020 arXiv announcement) and
a per-file soft-warning threshold around 10MB, both purely about file size, not
content length. If arXiv is used as a preprint venue independent of, or ahead
of, a journal decision, there is no formal constraint forcing compression.
**However**, the informal norm in the ML/finance literature this paper is
positioned against (Table `tab:related`'s own comparison set — MTGNN, StemGNN,
FourierGNN, THGNN, MDGNN) is conference-length (8–10 pages, NeurIPS/KDD/ICLR
two-column format) or journal-length as above; a 37-page arXiv preprint
in this specific subfield would be a conspicuous outlier relative to every
paper it cites and compares against, even though arXiv's own rules permit it.

### e. Bottom line for the second-wave agent

Use **~11,000–13,000 raw words total** (≈10,000–12,000 words of body text,
i.e. excluding the ~900-word bibliography) as the working target — this is
the manuscript's own measured 828 words/page density (§1, above) applied to
the IEEE Transactions 13–16-page range, and is corroborated by two real
Computational Economics comparables in the same subfield running 26 and 32
single-column pages (§1b). Deliverable as either:
- ~13–16 IEEE two-column pages if the current `IEEEtran` format is kept
  (at the more permissive end of the IEEE Transactions norm, e.g. TAC/TAP/TNet,
  not the tightest venues like TCom/TSP at 13pp hard), or
- ~26–32 single-column pages if reformatted for Computational Economics or a
  similar Springer/Elsevier journal (directly matching the two real comparable
  page counts found in §1b).

This is roughly a **57–64% cut from the current 30,628 raw words** (measured
directly from the file), not a trim. Section §3 below allocates that cut
per-section using each section's own measured current word count (not an
estimate), so the per-section targets sum to this total.

## 2. What to cut vs. keep — general guidance, sourced

### a. Methodology detail: main text vs. reproducibility appendix

Standard guidance (Sacred Heart University and USC's *Organizing Academic
Research Papers* guides, both retrieved and consistent on this point): the
main text should carry only what is necessary to understand and evaluate the
paper's central claims; material that is necessary for *reproduction* but not
for *evaluating the argument* — lengthy parameter tables, secondary
calculations, full search-space enumerations — belongs in an appendix. Applied
here: this paper's `tab:hpo-space` (main HPO search space) should stay in the
main text (it is load-bearing — the deployed model's configuration), but
`tab:hpo-wide-space` (the 3×-wider confirmatory robustness search, currently
inline in `sec:robustness` at lines 849–866) is a textbook case for appendix
placement — it exists to confirm a negative (the original search was already
adequate), not to establish a new claim.

### b. Negative-result papers and process-transparency (bugs found/fixed,
retracted claims)

The retrieved general guidance (Sacred Heart / USC Results-section guides) is
explicit that negative or null findings belong in the Results section in full,
and that a well-argued Discussion of *why* a negative result occurred is a
strength, not a weakness — directly supporting this paper's own framing that a
diagnosed negative result is more valuable than an unexamined one. No
venue-specific or genre-specific guidance was found stating a numeric cap on
how much self-correction narrative belongs in the main text (this gap is
disclosed, not filled with invention). What *can* be stated with confidence,
independent of a specific citation, is a structural principle stated directly
by Mensh & Kording's "Ten Simple Rules for Structuring Papers" (PLOS
Computational Biology, 2017, DOI `10.1371/journal.pcbi.1005619`; fetched
directly at `pmc.ncbi.nlm.nih.gov/articles/PMC5619685/`). Their Rule 4
("Optimize your logical flow by avoiding zig-zag and using parallelism")
states it explicitly: *"Only the central idea of the paper should be touched
upon multiple times. Otherwise, each subject should be covered in only one
place in order to minimize the number of subject changes."* Their Rule 3
makes the chronology point directly relevant here: *"[readers] do not care
about the chronological path by which you reached a result; they just care
about the ultimate claim and the logic supporting it."* **Applied to this
paper: each corrected fact should appear once, clearly, at the point a reader
needs it; the paper's logical structure, not the chronological order in which
the authors discovered and fixed each issue, should govern what stays in the
main text.** This paper's actual text does not follow that principle: it
frequently narrates *the order in which the authors found things* (batching
bug → early-stopping bug → checkpoint-floor bug → constant-forecast artifact →
retraction) with each bug's discovery, mechanism, and resolution restated at
every section that touches it, rather than stating each corrected fact once
and cross-referencing. **The self-correction content itself is a genuine
strength and should not be cut; the repetition of narrating it is the
compressible part.** Concretely: state "checkpoint-selection floor was
inconsistent across tables and is now fixed uniformly at `min_epochs=30`
(verified against saved checkpoints)" once, in Implementation Details
(`sec:impl`, where it already lives, lines 375), and thereafter refer back to
it (as the paper already does in some places) rather than re-explaining the
mechanism each time it becomes relevant to a later result.

### c. Ablation raw numbers: main-text table vs. appendix

Standard practice, consistent across the guides retrieved (USC/Sacred Heart)
and with the general shape of the Elsevier/Springer article types checked:
**the primary comparison table stays in the main text; secondary or
confirmatory tables that exist only to rule out an alternative explanation for
the primary table's result are appendix material.** Applied to this paper:
`tab:ablation` (the 8-variant main ablation) is load-bearing and stays.
`tab:confound-tuned` (2-row confirmatory re-run at tuned hyperparameters,
`sec:confound-tuned`) and `tab:hpo-wide-space` (the wider robustness HPO
search) are both confirmatory/negative-control tables whose *conclusion* ("the
apparent reversal was an artifact"; "a wider search doesn't help") is
load-bearing but whose *raw numbers* are appendix material — state the
conclusion and the one or two numbers that establish it in 2–3 sentences of
main-text prose, move the full table to an appendix.

### d. Appendix / supplementary-material split: is it standard for the target
venues?

IEEE Transactions journals explicitly support a print-adjacent Appendix
section (standard IEEEtran practice, and consistent with the appendix/
supplementary-material provisions referenced on the IEEE society author-info
pages in §1a). Elsevier's general *Your Paper Your Way* program
(`elsevier.com/subject/next/guide-for-authors`, fetched successfully) confirms
Elsevier journals broadly support supplementary material as standard
infrastructure. **Springer's own Computational Economics submission-guidelines
page could not be fetched in this research pass** (auth-walled redirect, same
issue as §1b) — Springer journals generally support Electronic Supplementary
Material as a standard feature across the platform, but this specific
journal's page was not independently confirmed, so treat the Springer half of
this claim as plausible-but-unverified rather than sourced. **An appendix or
supplementary-material split is very likely safe to use for any of this
paper's plausible target venues, confirmed directly for Elsevier/IEEE and
inferred but not independently verified for Springer.**

## 3. Section-by-section targets

"Current" word counts below are **measured directly** from `main.tex` with a
Python script splitting on `\section`/`\subsection` markers (not estimated;
counts include LaTeX markup tokens, captions and table-cell text, consistent
with how the 30,628-word/37-page density in §1 was computed, so the two are
comparable). Targets are scaled from these real counts so that **the column
sums to the §1e headline of ~11,000–13,000 raw words total** (~10,000–12,000
words of body text once the ~900-word bibliography, which is not shown in
this table and is not a compression target, is excluded). The point is
relative proportion and which sections absorb the cut — the current document
badly over-invests in "Results and Discussion" (15,574 of 29,717 body words,
**52% of the entire body**) relative to every other section.

| Section | Current (measured) | Target | Must stay (reviewer-checkable) | Cut candidates |
|---|---|---|---|---|
| Abstract | 471 words, 1 paragraph | **200–250 words** | RQ0 headline (nothing beats naive), the collapse finding (mechanism + numbers), the retraction, the fix-makes-it-worse finding | Nearly everything else — this is the highest-value, lowest-effort cut in the paper (see §1c/§1e) |
| Introduction (`sec:intro`) | 1,470 words, lines 69–106 | ~700 words | 4 RQs, 3–4 contributions, the positioning against MBTI-Net/Sun et al. | The bullet-point contributions list currently restates content that the abstract, the Discussion, and the Conclusion all also state in nearly identical language — pick one canonical statement of each claim and cross-reference the other two |
| Related Work (`sec:related`, 4 subsections) | 974 words (incl. `tab:related`), lines 107–155 | ~650 words | `tab:related` (the positioning table), the Fountoulakis et al. corroboration paragraph (load-bearing for the collapse diagnosis's citability) | The closing paragraph of `sec:related` (line 153, part of the 742-word "VMD Combined with GNN" subsection) already restates the RQ1 result and the collapse finding that properly belong in Results/Discussion — cut to 1–2 sentences pointing forward |
| Methodology (`sec:method`, 7 subsections) | 2,634 words + `tab:notation`, `alg:vmdmfgnn`, `fig:architecture`, lines 156–293 | ~1,500 words | Eq. 1–4 (VMD objective, adjacency, GAT, fusion), the leakage-safe rolling-window VMD design (real, citable, corroborated by Feng et al. 2026), `tab:notation`, `fig:architecture` | `sec:complexity` (604 words, lines 282–293) re-derives parameter-count bookkeeping in prose that **`tab:ablation-config` (lines 491–511) already tabulates** for the 8 ablation variants — cut the prose to 1–2 sentences and a single added row/footnote for the main HPO-tuned model's count (1,491,748 params), which is the one figure `tab:ablation-config` does not currently carry; the algorithm block (`alg:vmdmfgnn`) is near-fully redundant with `fig:architecture` by the text's own admission (line 229: "every block in the figure corresponds to one or more numbered lines of the algorithm") — keep the figure, move the algorithm pseudocode to an appendix |
| Experimental Setup (`sec:experiments`, 7 subsections) | 3,237 words + 2 tables, 1 figure, lines 294–379 | ~1,350 words | Data/split description, the 8→15-variable scope, the DM test definition and correction procedure | Ablation Studies subsection (776 words) mostly restates variant definitions Table `tab:ablation-config`/§3.5 will carry — compress to ~400; Statistical Testing (463 words) spends roughly half its length correcting the paper's *own earlier claim* that DM couldn't be applied to ablations — distill the correction to one sentence, this is the repeated-correction pattern flagged in §2b/§4 |
| Main Results (`sec:main-results`) | 1,860 words + `tab:main`, `tab:r2`, `fig:comparison`, lines 385–453 | ~700 words | `tab:main` and `tab:r2` in full, the RQ0 finding, the DM-vs-naive result, the sign-concentration artifact count (7 cells) | The paragraph-length walk of every individual margin ("VMD-MFGNN's h=1 edge is 0.02%... MTGNN's h=5,10 MAE edges are 0.31%...") can become one sentence ("all apparent naive-baseline wins are ≤1.8% relative and sit inside this study's 7.54% noise floor") — the table already shows the numbers; `fig:comparison` duplicates `tab:main` (§5) and is a strong cut candidate |
| Statistical Significance vs. Baselines (`sec:results-dm`) | 792 words + `tab:dm`, lines 454–485 | ~400 words | `tab:dm`, the 15/28 Holm-surviving count and the 10-vs-5 split, the "7 of 10 wins are against degenerate opponents" finding | Prose re-derivation of which specific cells are degenerate — already established in Main Results, cross-reference instead of repeating |
| Ablation Study (`sec:ablation-results`) | 1,821 words + `tab:ablation-config`, `tab:ablation`, `fig:ablation`, lines 486–557 | ~750 words | `tab:ablation`, the central DM test on `full_model` vs. `pooled_graph_matched_params`, the "loses to a graph that was never trained" finding | The full ranked list of all 8 variants' average RMSE with 5-decimal values in running prose (line 546) — the table already has this; `fig:ablation` duplicates `tab:ablation` (§5) |
| Confound-Check (`sec:confound-tuned`) | 740 words + `tab:confound-tuned`, lines 558–582 | **~150 words, fold into the constant-forecast discussion; move table to appendix if kept at all** | The retraction and why (constant-sign artifact) | This section exists almost entirely to be retracted two sections later (`sec:constantforecast`) — do not give it a standalone subsection |
| Interpretability Analysis / Collapse (`sec:collapse` + Mode Importance, the paper's most novel contribution) | 2,829 words + 4 figures (`fig:graphs`, `fig:graphs-nodelink`, `fig:collapse`, `fig:attention`), lines 583–648 | **~1,300 words — protect this subsection's substance more than any other** | Every number in the must-keep list (§5 below): RMS ~$10^{-11}$ vs. 0.2885 init, exactly-uniform adjacency, the dead-ReLU rule-out, the isotropic-collapse mechanism, the batching-bug re-diagnosis, the null-control result, the Fountoulakis et al. comparison | The narration of "an earlier verification pass found X, that was superseded, the real number is Y" appears three separate times (lines 597, 601, 612) for one corrected fact — state the corrected number once; `fig:graphs` and `fig:graphs-nodelink` render the same five matrices two ways (line 599 says so directly) — keep one |
| Graph-Fix Experiment (`sec:graphfix`) | 1,180 words, lines 649–659 | ~500 words | The fix mechanically works (embedding RMS restored), the fix does not produce learned structure (cosine ~1.0 to init), the gradient-goes-to-exactly-zero finding | The naming-collision disclaimer (line 652, ~200 words) explaining that two checkpoints share a filename but are different runs — **rename the checkpoints instead of prose-explaining the collision** (§4) |
| Robustness Checks (`sec:robustness`) | 2,595 words + `tab:hpo-wide-space`, lines 660–907 | **~700 words in main text, remainder to appendix — heaviest proportional cut target** | The headline result (19/19 collapse, ρ=-0.990 vs. epoch count), the 7.54% noise floor, the five-seed RQ1 result, one sentence on each of the 4 robustness axes' null result | See §4 and §5 — the cell-by-cell early-stopping narrative (told a fifth time here, see §2b) and the full `tab:hpo-wide-space` breakdown with its three caveats are the strongest appendix candidates in the paper |
| Constant-Forecast Artifact (`sec:constantforecast`) | 1,367 words, lines 908–918 | ~650 words | The base-rate identity itself, the retraction, the Transformer h=5 dispersion counterexample, the citation to Cheung 2026 and what this paper adds beyond it | The multi-paragraph explanation of exactly which cells are flagged and why, cross-referenced three ways — the flagged cells are already marked $^{\ddagger}$ in the tables; state the check's logic once and point to the table markers |
| Discussion (`sec:discussion`) | 1,800 words, lines 919–931 | ~700 words | The four RQ answers, each with its one load-bearing number | This subsection re-derives the full evidentiary chain behind each RQ answer nearly from scratch (re-stating DM counts, re-stating sign-concentration counts) rather than stating the answer and citing the section that established it — the "Taken together" paragraph (line 930) is close to a full restatement of the abstract |
| Economic Significance (`sec:econ`) | 543 words, lines 932–941 | ~350 words | The staleness disclosure (one sentence), the Sharpe/breakeven-cost/PT-test headline result | Smallest results subsection relative to its content; lowest priority for cutting |
| Limitations (`sec:limitations`) | **3,799 words, 18 enumerated items (not 16 — recounted directly from the `\item` lines, 947–965)**, lines 942–968 | **~1,400 words — second-heaviest cut target in the paper** | Items disclosing a real, unresolved threat to validity not argued at length elsewhere: items 1 (zinc/nickel drop), 3 (no purge/embargo), 7 (aluminum removal), 8 (ppiaco regime-shift finding), 9 (single split), 10 (HPO asymmetry), 15 (multiple-testing families), 16 (single-seed/noise floor) | Items **12** (ablation-doesn't-transfer, ~600 words — restates `sec:ablation-results`), **13** (collapse/interpretability, ~600 words — restates `sec:collapse`+`sec:graphfix` in full), **14** (DA-not-interpretable, ~350 words — restates `sec:constantforecast`), and **17** (three early-stopped cells, ~430 words — the *fifth* telling of this specific story after `sec:robustness` itself tells it four times) substantially re-argue findings already fully established elsewhere — a Limitations section should *list* threats to validity concisely, not re-argue them; cut each of these four items to 2–3 sentences with a cross-reference |
| Conclusion (`sec:conclusion`) | 1,203 words, lines 969–978 | ~500 words | The four RQ answers restated briefly, the "things we offer beyond this architecture" | This section currently restates the entire evidentiary chain a third time (after Discussion and the Abstract) — should be the shortest, most declarative section in the paper, not one of the longest |
| Data/Code Availability, CRediT, Funding, COI | 148 words total, lines 979–1006 | ~148 words (unchanged) | All of it — required boilerplate, already minimal | None; not a compression target |

**Column check**: 220+700+650+1500+1350+700+400+750+150+1300+500+700+650+700+350+1400+500+148 ≈ **12,668 words**, within the ~11,000–13,000 target band from §1e (bibliography's ~900 words are additional and not counted against this budget, consistent with §1's own page-density calculation, which was computed on the full 30,628-word/37-page file inclusive of the bibliography).

## 4. Concrete compression techniques, with backing

1. **Move confirmatory/negative-control material to an appendix, keep the
   conclusion in the main text.** Standard practice per §2a/§2c above, and
   explicitly supported infrastructure at every plausible target venue (§2d).
   Candidates, in priority order: `tab:hpo-wide-space` + its three caveats
   (lines 835–894); `tab:confound-tuned` + most of `sec:confound-tuned`'s prose
   (lines 558–581, keeping only the retraction); the full 19-cell robustness
   breakdown of `sec:robustness` (keep the headline correlation numbers,
   ρ=-0.990 etc., in the main text; move the cell-by-cell narrative — three
   early-stopped cells, the four-cell replicate-noise check, the Huber-arm
   digression — to an appendix table with 2–3 sentences of main-text summary).
   The algorithm pseudocode (`alg:vmdmfgnn`) is also a strong appendix
   candidate given its near-total overlap with `fig:architecture` (the text
   says so itself at line 229).

2. **State each corrected fact once; cross-reference, don't re-narrate.** This
   is the single highest-leverage technique for this specific paper, because
   its most distinctive feature — real, disclosed self-correction — is also
   its largest source of avoidable repetition. Concretely: the
   checkpoint-selection-floor bug is narrated in full at `sec:impl` (375),
   referenced again in `sec:graphfix` (652), and referenced again in
   `sec:robustness` (710–727). The batching bug is narrated in full at
   `sec:collapse` (601), referenced again in `sec:graphfix` (652), and
   referenced again in the Limitations enumeration (960). Each of these is one
   fact; state it fully once (where it is first load-bearing), and elsewhere
   use a single clause ("under the corrected `min_epochs=30` floor, Section
   X") rather than re-explaining the bug's mechanism and discovery each time.
   This does not remove any disclosure — the fact remains stated in full
   exactly once — it removes the *n*-times repetition. Consider a short,
   explicit "Corrections and Disclosures" subsection (in the main text or an
   appendix) that lists every bug found/fixed and every retraction in one
   place, once, so the rest of the paper can reference it by name instead of
   re-explaining.

3. **Rename colliding checkpoint identifiers instead of prose-explaining the
   collision.** Line 652 spends roughly 200 words explaining that
   `full_model_graphfix` in `sec:graphfix` and `full_model_graphfix` in
   `tab:ablation` are different trained models that happen to share a
   filename. The cheaper fix is renaming one of them in-text (e.g.
   `full_model_graphfix_early` for the earlier-phase, epoch-57 run vs.
   `full_model_graphfix` for the real-data-scale, epoch-29 ablation-table
   run) and deleting the disambiguation paragraph — the new names disambiguate
   on sight, everywhere they appear, at zero ongoing prose cost. The same
   applies to `full_model_tuned`/`pooled_graph_matched_params_tuned` (the
   earlier-phase, $N=8$ confound-check checkpoints, `sec:confound-tuned`) if
   that section is kept rather than folded into §3's recommended compression.

4. **Convert prose ablation walkthroughs into table + 2–3 sentences.**
   Multiple sections (Main Results, Ablation, Robustness) currently narrate
   individual cell values in prose that duplicate what the adjacent table
   already shows exactly (e.g. line 546's full 8-variant RMSE ranking with
   5-decimal values). Standard results-section guidance (USC/Sacred Heart,
   §2c) treats the table as the primary evidentiary artifact and prose as
   interpretation of it, not a restatement of it — cut prose to the
   interpretive claim ("the full model loses to every other variant including
   an untrained fixed-correlation graph") and let the table carry the
   numbers.

5. **Cut repeated hedging/caveat language to one canonical statement.**
   Phrases like "we state this explicitly rather than leave it implicit," "we
   report this rather than omit it," and "we disclose this openly" recur
   dozens of times across the document as a rhetorical tic accompanying
   otherwise-fine disclosures. The disclosure itself should stay; the
   meta-commentary about the act of disclosing it is safe to cut uniformly —
   it adds length without adding information, and its repetition dilutes
   rather than reinforces the paper's genuine transparency.

6. **Abstract**: rewrite from scratch at 200–250 words (ESWA's cap, §1c) rather
   than trim the existing 471-word paragraph — the current abstract tries to
   carry the full evidentiary chain (which is the Discussion/Conclusion's job)
   rather than stating the four headline results plainly. Split it into 2–3
   short paragraphs or a structured abstract if the target venue supports one.

## 5. Must-keep list (do not let compression touch these)

These are the load-bearing claims/numbers a reviewer would specifically check
for, identified from the paper's own most novel contribution (the collapse
diagnosis) and its central negative results. Flagging these explicitly so the
second-wave agent does not cut them while pursuing the targets above.

- Embedding RMS collapse to the order of $10^{-11}$ against a Xavier-init
  reference of 0.2885 (the headline collapse number, `sec:collapse`).
- The adjacency being exactly uniform ($1/N$) to full floating-point
  precision in every band (`sec:collapse`).
- The dead-ReLU rule-out (pre-activation negative fraction unchanged,
  50.5%→56.3%; removing ReLU still yields uniform adjacency) — this is what
  makes the diagnosis a mechanism rather than a guess.
- 19/19 cells collapsed in the robustness sweep, with Spearman ρ = −0.990
  against epoch count (`sec:robustness`) — the evidence that collapse is an
  optimization dynamic, not a configuration artifact.
- The gradient reaching the graph embeddings: small-but-nonzero at
  initialization, exactly zero at trained checkpoints (`sec:graphfix`) — the
  finding that sharpens "weak gradient" into "gradient vanishes to exact
  zero over training, independent of weight decay."
- The 7.54% same-configuration, same-seed noise floor (`sec:robustness`) —
  this is the yardstick against which every other reported effect size in the
  paper must be read; do not let it get cut while trimming the section it
  lives in.
- The constant-sign-forecast base-rate identity (a model's DA exactly equals
  the base rate of the sign it always emits) and the resulting retraction of
  the tuned-confound-check reversal (`sec:constantforecast`).
- The Transformer $h=5$ counterexample (healthy 0.42 dispersion ratio, yet
  100% single-sign) — this is what justifies requiring *both* a
  sign-concentration check and a dispersion check, not either alone.
- The null-control experiment's finding that the batching-bug fix is
  mechanically real (gradient differs 100–450× across real/shuffled/Gaussian
  input arms) even though collapse is confirmed input-independent — both
  halves of this two-sided finding must survive compression together, since
  either half alone would misrepresent the result.
- The graph-fix's central puzzle: the fix mechanically defeats the collapse
  (embedding RMS restored to 0.235–0.262 against 0.2885 reference, real
  non-uniform adjacency structure) yet the resulting model forecasts *worse*
  at every horizon past $h=1$ — reported as an open, unresolved finding. Do
  not let compression accidentally resolve or soften this into "the fix
  didn't work"; it did work at what it was designed to fix, and that is the
  point.
- RQ0 (nothing beats the naive zero-forecast baseline on MAE) and its
  Holm-Bonferroni-corrected significance at $h=10$ specifically.

## 6. Top compression targets, prioritized (for the second-wave agent's task list)

1. **Abstract**: 471 → ~220 words (measured, §0). Highest ratio of effort
   saved to length cut; every plausible venue caps this well below its
   current length (§1c).
2. **`sec:robustness` (Robustness Checks)**: 2,595 measured words → ~700 in
   main text, remainder to appendix. Together with the adjacent
   `sec:graphfix` (1,180 words) and `sec:constantforecast` (1,367 words), this
   three-subsection block runs 5,142 words across lines 649–918 — the largest
   contiguous block in the document. The headline correlation (ρ=-0.990) and
   the 7.54% noise floor must survive; the cell-by-cell narrative of which of
   the 19 runs was re-diagnosed and why (told across four separate passages
   within this subsection alone, plus a fifth telling in Limitations item 17)
   should not.
3. **`sec:limitations`**: 3,799 measured words, **18** enumerated items (not
   16 — recounted directly, §3) → ~1,400 words. Items **12**, **13**, **14**,
   and **17** substantially re-argue findings already fully established in
   `sec:ablation-results`, `sec:collapse`/`sec:graphfix`, and
   `sec:constantforecast` respectively, rather than concisely flagging them.
4. **Repeated bug/correction narration** (checkpoint-floor bug, batching bug,
   confound-check retraction): each currently told in full 2–3 times across
   different sections. Consolidate each to one canonical statement with
   cross-references — recovers real length without losing any disclosure.
5. **`sec:confound-tuned`**: 740 measured words + its own table, for a result
   the paper itself retracts two sections later. Fold into a short paragraph
   inside the constant-forecast discussion; move `tab:confound-tuned` to an
   appendix if kept at all.

## 7. Reconcile, don't preserve: internal inconsistencies compression should
## not carry forward silently

The must-keep list (§5) protects specific numbers from being cut. It does
**not** mean every number currently in the text is correct as stated — the
paper's own repeated-narration pattern (§2b, §4) has left several
contradictions between sections that a compression pass could easily
propagate rather than resolve, simply by keeping whichever restatement it
happens to trim last. The second-wave agent should reconcile these against
the underlying results artifacts (not against this document, which was not
able to verify which side of each contradiction is correct) rather than
picking one occurrence to keep at random:

- **$N=8$ vs. $N=15$.** `tab:notation`, the Problem Formulation subsection,
  the Data subsection, the architecture figure caption, and the algorithm
  block all state or imply $N=8$ input variables, while the checkpoints
  actually analyzed from `sec:collapse` onward are $N=15$ (explicit at line
  597: "$1/N=1/15$"). Line 83 claims the Data section (`sec:data`) discloses
  this discrepancy; on inspection `sec:data` does not mention the 8→15
  expansion at all — that expansion is instead disclosed piecemeal in
  Limitations items 7 and 8. This is the single most consequential
  inconsistency in the document, since it affects how every early-section
  equation and notation table should be read.
- **"No FRED data" vs. four FRED series.** Lines 300 and 981 (Data
  Availability) both state no FRED data is used; Limitations item 8 describes
  four monthly FRED series (including `ppiaco`) merged into the expanded
  feature set. These directly contradict each other.
- **Aluminum in scope vs. removed.** Lines 234 (architecture figure caption)
  and 300 (Data subsection) list aluminum (`ALI=F`) as an input variable;
  Limitations item 7 discloses it was removed after original results were
  produced.
- **Ablation parameter counts.** Line 548 states "397,586 vs.\ 393,476 params"
  for `full_model` vs. `pooled_graph_matched_params`; `tab:ablation-config`
  (the paper's own consolidated parameter-count table) gives 1,448,548 and
  1,452,010 for the same two variants. These cannot both be right.
- **Graph-fix variant size.** Line 552 describes the graph-fix variants as
  "roughly $3.7\times$ larger" in parameter count than `full_model`;
  `tab:ablation-config` shows `full_model_graphfix` at an identical parameter
  count to `full_model` (1,448,548 vs. 1,448,548) and
  `full_model_graphfix_temp` only 5 parameters larger.
- **HPO-tuned hidden dimension.** Line 612 refers to the HPO-tuned
  configuration as hidden dimension 128; `sec:hpo` and `tab:hpo-space` both
  give the selected best trial's hidden dimension as 64.

None of these affect the paper's central findings (the must-keep list in §5
is unaffected by any of them), but a compressed version of the paper that
silently keeps the wrong side of one of these contradictions would introduce
a real error the current, over-long version does not clearly have anywhere a
reader could check — resolve each one against the underlying saved
checkpoints/results artifacts before finalizing.

## 8. Float (figure/table) redundancy — page-level savings word counts miss

Word-count targets in §3 do not capture float cost: in a two-column layout,
a `figure*`/`table*` full-width float can cost most of a page regardless of
its caption's word count, so trimming floats is a distinct lever from
trimming prose. Real duplication found by inspection:

- `fig:graphs` (heatmap) and `fig:graphs-nodelink` (node-link diagram) render
  the identical five saved adjacency matrices two different ways — the text
  says so directly (line 599: "the same five saved adjacency matrices"). Keep
  one, not both.
- `fig:comparison` (grouped bar chart of RMSE/MAE/DA) duplicates the exact
  numbers already in `tab:main`; `fig:ablation` duplicates `tab:ablation` the
  same way. Both are candidates to cut if page budget is tight, since neither
  conveys information the table doesn't already carry precisely.
- `fig:vmd` (the VMD decomposition itself) is included, by the text's own
  description (line 638), "for completeness" rather than to support a
  specific claim — the weakest-justified figure in the paper and a
  reasonable first cut if a figure must go.
- Most of the above are `figure*`/`table*` full-width floats (`tab:main`,
  `tab:dm`, `tab:ablation`, `fig:architecture`, `fig:comparison`,
  `fig:ablation`, `fig:graphs`, `fig:graphs-nodelink`, `fig:collapse`,
  `fig:vmd`) — cutting even one or two saves a disproportionate amount of
  page count relative to its word-count footprint in §3's table.

## 9. Sources cited in this document

**Fetched directly** (page content retrieved and read in this research pass):

- IEEE Trans. Automatic Control author info — ieeecss.org/publication/transactions-automatic-control/author-info
  ("normally around 12 pages," 16 pp hard max incl. bios, $125/page charge past 12)
- Computational Economics comparable #1 (Crossref API) —
  api.crossref.org/works/10.1007/s10614-025-10861-z — "Forecasting Crude Oil
  Prices: Evidence From WOA-VMD-FE-Transformer Model," pages 4645–4676 (32 pp)
- Computational Economics comparable #2 (Crossref API) —
  api.crossref.org/works/10.1007/s10614-025-10946-9 — "A Crude Oil Price
  Forecasting Model Based on Local Mean Decomposition, Marine Predators
  Algorithm and Least Squares Support Vector Regression," pages 2823–2848 (26 pp)
- Padhan, Senthil, Panigrahi, Sahoo (2026), *Computational Economics*, DOI
  10.1007/s10614-026-11436-2 — confirmed to exist via Crossref API
  (api.crossref.org/works/10.1007/s10614-026-11436-2), 56 references; no
  page-range field yet assigned in the Crossref record
- arXiv submission size policy — info.arxiv.org/help/index.html and its linked
  size-limit help file (no page/word limit; 50MB total submission cap, ~10MB
  per-file soft-warning threshold)
- Elsevier "Your Paper Your Way" — elsevier.com/subject/next/guide-for-authors
  (no strict page count at initial submission; expects a "concise and complete" article)
- Mensh, B., Kording, K. (2017), "Ten simple rules for structuring papers,"
  *PLOS Computational Biology* 13(9): e1005619, DOI 10.1371/journal.pcbi.1005619
  — fetched at pmc.ncbi.nlm.nih.gov/articles/PMC5619685/; Rule 3 and Rule 4
  quoted directly in §2b
- Direct measurement: `manuscript/main.tex` itself (line counts, word counts
  per section via a Python script splitting on `\section`/`\subsection`
  markers, abstract word count, `\item` recount for Limitations) — run
  against the actual file during this research pass, not estimated

**Web-search-summary sourced** (the underlying page was not independently
re-fetched in this pass; treat as one level less direct than the above, and
re-verify from a browser before quoting as a hard requirement):

- IEEE Trans. Networking (16 pp max) — comsoc.org/publications/journals/ieee-tnet/ieee-transactions-networking-author-guidelines
- IEEE Trans. Communications (≤13 double-column pages) — comsoc.org/publications/journals/ieee-tcom/policies-and-guidelines
- IEEE Trans. Signal Processing (≤13 double-column pages, 10-pt) — signalprocessingsociety.org/publications-resources/information-authors
- IEEE Trans. Cybernetics (~10 pp typical) — ieeesmc.org/publications/transactions-on-cybernetics/information-for-authors-3
- IEEE Trans. Antennas & Propagation (1–16 pp) — ieeeaps.org/ieee-tap/for-authors/how-to-prepare-your-submission
- Expert Systems With Applications, Guide for Authors (abstract ≤250 words;
  highlights 3–5 bullets ≤85 chars) — sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors
  (direct WebFetch returned HTTP 403; figures are from an indexed search summary only)
- Organizing Academic Research Papers guides (Results, Appendices) — Sacred
  Heart University Library (library.sacredheart.edu) and USC Libraries
  (libguides.usc.edu/writingguide/results)

**Could not be retrieved at all — flagged as unverified gaps, not filled with
an invented number:**

- Computational Economics's own numeric submission length limit —
  link.springer.com/journal/10614/submission-guidelines (auth-walled redirect
  to idp.springer.com; the two real comparable page counts above substitute
  for this, §1b)
- Resources Policy's Guide for Authors page —
  sciencedirect.com/journal/resources-policy/publish/guide-for-authors (403
  from ScienceDirect)
- The Padhan et al. comparable paper's exact page count (403 from SSRN,
  absent from its own Crossref record)
- Journal of Forecasting (Wiley) author guidelines page — word limit not
  retrievable in the search pass performed; not relied upon anywhere above
