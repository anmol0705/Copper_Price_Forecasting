# Bibliography Index — Literature Review for Paper 1
## "Does Per-Frequency-Band Graph Learning Help Copper Price Forecasting?"

Compiled 2026-09-03. This index covers papers found via arXiv (search_papers / get_abstract /
download_paper tools) that are genuinely new relative to `paper1/main.tex`'s existing 34
`\bibitem` entries (checked against the full bib-key list: Ahmad2024, Ahmad2025, Bao2025,
Becerra2022, Cao2020, ChengTsai2026, Cheng2022, Dai2025, DieboldMariano1995, Dragomiretskiy2014,
Garcia2019, IEA2024, Hu2020, Hu2023, Li2023, LiLiu2026, Liu2020, LiuLiu2022, Luo2022, Nabavi2024,
Pei2022, Qian2024, Sun2025, Tseng2025, Velickovic2018, Wu2020mtgnn, Xiang2023, Yi2023, Yuan2025,
Zhai2024, Zhang2025a, ZhangKe2025, Zhao2025, ZhengZhufu2025). All 17 papers below were located
via real arXiv search results and downloaded as PDFs into this directory. No fabricated entries.

Every entry below is confirmed **not** a duplicate of an existing citation (see individual notes;
one near-duplicate was found and explicitly *excluded*, see "Papers checked and excluded" at the
end).

---

## Topic Area 3 (checked first): Base-rate / constant-forecast / directional-accuracy artifacts

### Cheung, T. (2026). "When Directional Accuracy Lies: A Base-Rate-Honest Benchmark for LoRA-Adapted TimesFM on Equity Forecasting." arXiv:2607.12248.
**File:** `../shared/2607.12248_cheung_when_directional_accuracy_lies.pdf` (deduped; shared with paper2)
**Status: CONFIRMED FOUND AND DOWNLOADED** — this is the exact paper named in the task brief.
**Relevance:** This is the single closest methodological precedent for this paper's Section
6 (constant-forecast diagnosis). Cheung shows an ~80% directional accuracy figure for a
LoRA-adapted TimesFM model on equities was actually a base rate (~70% "always-up" during a
rising market) that the fine-tuned model scored *below* once corrected — nearly the identical
failure mode this paper documents for the Transformer baseline's h=5 forecast (100%
positive-sign predictions coincidentally matching the test period's up-day frequency). Cheung's
protocol (honest baselines including "always-up", paired significance tests with FDR control,
frozen-data benchmark) is a template this paper's Section 6 diagnostic implicitly follows
independently; citing it strengthens the claim that this is a *known, recurring, general* failure
mode in financial ML rather than an idiosyncratic bug, and gives the paper's "directional
accuracy is uninterpretable without checking prediction dispersion" claim a contemporaneous
peer precedent (both papers appeared/were worked on in 2026).

### Zhang, F., Li, Z., Peng, S., Chen, Y. (2026). "When Alpha Disappears: A One-Switch Benchmark for Decision-Time Leakage in Financial Backtests." arXiv:2605.23959.
**File:** `2605.23959_Zhang_WhenAlphaDisappearsLeakageBacktest.pdf`
**Relevance:** Companion methodological finding to Cheung (2026): rather than base-rate
artifacts, this paper isolates "decision-time leakage" in financial backtests by toggling one
evaluation convention at a time and measuring the resulting alpha inflation. Directly relevant
to this paper's leakage-safe-protocol framing (Section 3, VMD rolling-window refit; the paper's
own disclosed history of an earlier expanding-window decomposer that leaked information). Useful
as a second, independent 2026 paper making the general point that small, easy-to-miss protocol
choices in financial ML backtests silently manufacture apparent skill — reinforcing this paper's
framing that rigor/evaluation-protocol is itself the contribution.

---

## Topic Area 1: GNNs for financial/commodity forecasting — learned adjacency, GAT, failure modes at small N

### Roth, A., Liebig, T. (2023). "Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks." arXiv:2308.16800.
**File:** `2308.16800_Roth_RankCollapseOverSmoothing.pdf`
**Relevance:** Directly relevant to Section 5 (graph-embedding collapse diagnosis). This paper
provides a general theoretical account of *rank collapse* in GNN representations — node/feature
representations becoming dominated by a low-dimensional subspace as depth increases — which is
mechanistically adjacent to (though not identical to) the isotropic-norm-decay collapse this
paper diagnoses in VMD-MFGNN's per-band adjacency embeddings. It gives the paper's collapse
diagnosis a citable theoretical vocabulary ("rank collapse," "over-correlation") and prior art
for the general phenomenon of GNN representations degenerating toward a low-information solution
during training, distinct from the classic over-smoothing-via-depth story (this paper's model
uses only 1-2 GAT layers, so depth-driven over-smoothing is not the mechanism — a point this
citation helps make precisely: the paper's collapse is closer to Roth & Liebig's rank-collapse
framing than to classical over-smoothing).

### Roth, A., Bause, F., Kriege, N. M., Liebig, T. (2024). "Preventing Representational Rank Collapse in MPNNs by Splitting the Computational Graph." arXiv:2409.11504.
**File:** `2409.11504_Roth_PreventingRankCollapseMPNNs.pdf`
**Relevance:** Follow-up work by the same group proposing an architectural fix for rank collapse
in message-passing GNNs. Useful contrast to this paper's own fix (Section on graphfix: unit-
normalizing embeddings, excluding them from weight decay) — both papers target the same failure
class from different angles (architectural graph-splitting vs. this paper's normalization/
weight-decay-exclusion approach), and both find that resolving the mechanical collapse symptom
does not by itself guarantee the network learns task-relevant structure, echoing this paper's
own finding that the corrected embeddings still moved only 0.6-2.0% from random initialization.

### Fountoulakis, K., Levi, A., Yang, S., Baranwal, A., Jagannath, A. (2022). "Graph Attention Retrospective." arXiv:2202.13060.
**File:** `2202.13060_Fountoulakis_GraphAttentionRetrospective.pdf`
**Relevance:** A theoretical analysis of when and why graph attention (the GAT mechanism this
paper's FrequencyBandModule uses) succeeds or fails to recover useful structure, framed via a
contextual stochastic block model. Relevant background for the paper's claim that its per-band
GAT layers never learned differentiated attention/adjacency structure: this paper formalizes
conditions (signal-to-noise, node/feature regimes) under which graph attention provably fails to
separate informative from uninformative edges, which is the theoretical backdrop for a small-N
(N=8 nodes), noisy-financial-signal setting like this paper's being exactly the kind of regime
where such failure is theoretically expected rather than anomalous.

### Cini, A., Marisca, I., Zambon, D., Alippi, C. (2023). "Graph Deep Learning for Time Series Forecasting." arXiv:2310.15978.
**File:** `2310.15978_Cini_GraphDeepLearningTimeSeries.pdf`
**Relevance:** A comprehensive, recent survey of graph-based deep learning for time series
forecasting (successor-generation survey to the `gnn_literature_review.md`'s MTGNN/StemGNN/Graph
WaveNet-era coverage). Useful as an up-to-date, single citable reference for the state of the
field circa 2023, including explicit discussion of when learned/adaptive graphs help versus
merely add capacity without adding structure — directly relevant to this paper's RQ2 (does
end-to-end learned adjacency add measurable value over fixed correlation-based graphs), and a
good general Related Work anchor citation that is more current than the paper's existing
MTGNN/StemGNN/FourierGNN citations taken individually.

### Hong, Y., Klabjan, D. (2026). "Hierarchical Graph Learning for Calendar Spread Strategies in Commodity Futures Markets." arXiv:2606.25811.
**File:** `2606.25811_Hong_HierarchicalGraphLearningCommodityFutures.pdf`
**Relevance:** A 2026 application of learned graph structure specifically to commodity futures
markets (calendar spreads across underlying assets and individual contracts), not a copper paper
but the closest very-recent match for "graph learning + commodity futures" outside of the
paper's own MBTI-Net/Sun et al. (2025) MVGT lineage. Strengthens the Related Work section's claim
about how sparse commodity-specific graph-GNN literature is: this is one of very few genuinely
new 2025-2026 entries in that specific niche, and is architecturally distinct from VMD-MFGNN
(hierarchical contract-vs-underlying graph, no frequency decomposition), so it is complementary
rather than duplicative of Sun2025 or LiLiu2026.

---

## Topic Area 2: VMD / decomposition methods combined with deep learning for financial/commodity forecasting

### Boadi, E. (2025). "Bitcoin Price Forecasting Based on Hybrid Variational Mode Decomposition and Long Short Term Memory Network." arXiv:2510.15900.
**File:** `2510.15900_Boadi_BitcoinVMDLSTM.pdf`
**Relevance:** A direct, recent (2025) extension of the VMD-LSTM family (the paper's own
Liu2020 lineage) applied to a different volatile financial asset (Bitcoin instead of copper/
non-ferrous metals). Useful as evidence that the VMD-LSTM template remains actively applied
without methodological advances (no cross-asset graph structure, no leakage-safety discussion),
reinforcing this paper's framing that most of the decomposition-ensemble literature "treats
variables independently, discarding cross-variable information" (main.tex Section 2, Related
Work) — this is a clean, recent instance of exactly that limitation.

### Feng, W., Tao, R., Cartlidge, J., Zheng, J. (2025). "VMDNet: Temporal Leakage-Free Variational Mode Decomposition for Electricity Demand Forecasting." arXiv:2509.15394.
**File:** `2509.15394_Feng_VMDNetLeakageFreeVMD.pdf`
**Relevance:** The single most methodologically relevant VMD paper found in this search. VMDNet
explicitly targets and names the same VMD-leakage problem this paper's Section 3 (VMD
methodology) resolves via a rolling-window, refit-every-day design, and which this paper
explicitly reports having previously gotten wrong (the disclosed earlier expanding-window/21-day
refit decomposer that leaked information, replaced before any reported results). Citing VMDNet
gives this paper's leakage-safety discussion a directly on-point 2025 precedent showing the VMD
temporal-leakage problem is recognized as a real, general methodological issue outside finance
too (electricity demand), not an idiosyncratic concern specific to this project. This is a
strong candidate addition to Section 3's VMD discussion or Section 6 (limitations/leakage
framing).

### Putra, H. R. K., Yudistira, N., Fatyanosa, T. N. (2024). "Variational Mode Decomposition and Linear Embeddings are What You Need For Time-Series Forecasting." arXiv:2408.16122.
**File:** `2408.16122_Putra_VMDLinearEmbeddings.pdf`
**Relevance:** A 2024 VMD-based forecasting architecture using linear embeddings rather than
LSTM/GNN, relevant as an alternative point in the VMD-plus-downstream-model design space, and
useful evidence for the claim (Section 2) that VMD-combined approaches are an actively developing
area with many recent variants, most still univariate/single-series and not incorporating
cross-asset graph structure.

### Xue, X., Li, S., Wang, X. (2024). "Enhanced forecasting of stock prices based on variational mode decomposition, PatchTST, and adaptive scale-weighted layer." arXiv:2408.16707.
**File:** `2408.16707_Xue_VMDPatchTSTStockForecasting.pdf`
**Relevance:** Combines VMD with a Transformer-family model (PatchTST) rather than LSTM/GNN,
representing the "VMD + modern sequence architecture" branch of the decomposition-ensemble
literature. Useful for Related Work as one more up-to-date (2024) VMD-hybrid data point showing
the field's continued churn of VMD-plus-{LSTM, Transformer, GNN} recombinations without addressing
cross-variable relational structure or reporting a naive/zero-forecast baseline comparison — a
pattern this paper explicitly critiques.

---

## Topic Area 4: Copper and industrial metal price forecasting specifically

### Wang, Z., Lu, X. (2024). "COMEX Copper Futures Volatility Forecasting: Econometric Models and Deep Learning." arXiv:2409.08356.
**File:** `../shared/2409.08356_wang_lu_comex_copper_volatility_forecasting.pdf` (deduped; shared with paper2)
**Relevance:** A copper-specific forecasting paper not in the existing bibliography, comparing
GARCH/HAR econometric volatility models against deep-learning RNNs on high-frequency COMEX
copper futures. Directly on-topic for Section 2.1 (Copper Price Forecasting, currently citing
Becerra2022/ARIMA, Garcia2019/GARCH, Dai2025/VAR-VECM) — this is a genuinely new, more recent
(2024) econometric-vs-deep-learning copper comparison that belongs alongside those citations,
and is notable for targeting *volatility* rather than *level/return* forecasting, a
complementary target variable worth flagging in Discussion/Limitations as an alternative
forecasting target this paper did not pursue.

### Wang, Z., Li, X. (2024). "On the macroeconomic fundamentals of long-term volatilities and dynamic correlations in COMEX copper futures." arXiv:2409.08355.
**File:** `../shared/2409.08355_wang_li_macro_fundamentals_comex_copper.pdf` (deduped; shared with paper2)
**Relevance:** Companion paper (same first author/group as 2409.08356) using GARCH-MIDAS/
DCC-MIDAS to link low-frequency macroeconomic variables to copper futures' high-frequency
returns and dynamic correlation with the S&P 500 — one of this paper's own N=8 input variables.
Relevant to the paper's claim (Section 1, Introduction) that "copper-dollar correlation at low
frequencies... may differ structurally from... high frequencies" — this is an econometric paper
making essentially the same multi-frequency-dependency argument using GARCH-MIDAS rather than
VMD+GNN, and is a good citation for motivating why a frequency-aware architecture is a reasonable
premise to test in the first place, independent of whether VMD-MFGNN itself succeeded.

---

## Topic Area 5: Weak-form market efficiency tests on commodities/metals

**Note on this topic area:** arXiv coverage of weak-form efficiency / variance-ratio testing
specifically on LME/COMEX copper or base metals is thin — this literature lives predominantly in
finance/econometrics journals (Resources Policy, Journal of Commodity Markets, Applied
Economics) that are not arXiv-indexed. Multiple targeted searches (variance-ratio + copper/LME,
Hurst exponent + metals, random-walk + futures 2021-2026) did not surface a paper testing
variance-ratio/random-walk hypotheses on LME or COMEX copper specifically newer than what the
paper already implicitly draws on. The two closest arXiv matches, both testing weak-form
efficiency via different (non-variance-ratio) statistical machinery on assets that include
metals, are included below as the best available substitutes; neither is a copper-specific
variance-ratio study, so this should be flagged as a genuine gap rather than assumed covered.

### Brouty, X., Garcin, M. (2023). "Fractal properties, information theory, and market efficiency." arXiv:2306.13371.
**File:** `2306.13371_Brouty_FractalInfoTheoryMarketEfficiency.pdf`
**Relevance:** Establishes a theoretical link between the Hurst exponent (fractal/long-memory
measure) and entropy-based market-information measures, both used as weak-form efficiency
diagnostics. Relevant as a general, rigorous (2023) alternative-methodology paper for testing
market efficiency, useful if the paper's Discussion/Limitations wants to note variance-ratio
testing is one of several available weak-form efficiency diagnostics, with Hurst/entropy-based
approaches as a natural robustness check not pursued here.

### Takaishi, T. (2026). "The Impact of Trump-Era Tariffs on Financial Market Efficiency." arXiv:2602.00548.
**File:** `2602.00548_Takaishi_TariffsMarketEfficiency.pdf`
**Relevance:** Applies multifractal detrended fluctuation analysis (MF-DFA) to test time-varying
market efficiency across six assets including Gold — a metal closely related to copper and one
of this paper's own N=8 input variables — around a specific macro shock (tariff policy), a period
overlapping this paper's 2022-2025 test window. Useful precedent for the idea that market
efficiency in metals is regime-dependent and shock-sensitive, relevant to Section 4 (Discussion)
framing of why a single chronological split spanning multiple regimes (COVID, rate hikes, tariff
shocks) is a demanding test environment. Not copper-specific and not a variance-ratio test, so it
only partially fills the topic-5 gap noted above.

---

## Topic Area 6: Reproducibility and rigor in financial machine learning

### Kapoor, S., Narayanan, A. (2022). "Leakage and the Reproducibility Crisis in ML-based Science." arXiv:2207.07048.
**File:** `2207.07048_Kapoor_LeakageReproducibilityCrisisML.pdf`
**Relevance:** A seminal, widely-cited (Nature-published) paper systematically documenting data
leakage as the dominant cause of irreproducible ML-based science findings across multiple fields.
This is the single strongest candidate citation for this paper's overall framing as a rigor-
focused negative result: the paper's own narrative (an earlier VMD decomposer leaked information
via a 21-day-block refit and was replaced before any reported result; a re-run initially thought
to reverse the central finding turned out to be a constant-forecast artifact) is a textbook
instance of exactly the failure-and-correction pattern Kapoor & Narayanan document at a
field-wide level. Strongly recommended for the Introduction or Limitations section as the
paper's central rigor/reproducibility citation — arguably a more load-bearing citation than
several of this topic area's other finds.

### Wasserbacher, H., Spindler, M. (2021). "Machine Learning for Financial Forecasting, Planning and Analysis: Recent Developments and Pitfalls." arXiv:2107.04851.
**File:** `2107.04851_Wasserbacher_MLFinancialForecastingPitfalls.pdf`
**Relevance:** A practitioner-oriented survey of ML pitfalls specifically in financial
forecasting/planning/analysis (not general ML), covering common methodological mistakes
(overfitting, inadequate baselines, backtesting flaws) that this literature routinely makes.
Complements Kapoor & Narayanan's general ML framing with a finance-specific pitfalls catalogue,
directly supporting the paper's claim that omitting a naive zero-forecast baseline and reporting
raw directional accuracy without dispersion checks are common, named failure modes in this
specific sub-literature, not the paper's own idiosyncratic concerns.

---

## Papers checked and excluded (near-duplicates / not downloaded)

- **arXiv:2509.00703** ("Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded
  Variational Mode Decomposition," Ahmad, Wesemann, Waschkowski, Khalid, 2025) — checked via
  `get_abstract` and confirmed this is the same "Mode Adaptive Graph Network (MAGN)" /
  deep-unfolded VMD paper already cited in `main.tex` as `\bibitem{Ahmad2025}`. **Not
  downloaded** — genuine duplicate, not new material.
- **arXiv:2010.13152** ("A Simple Spectral Failure Mode for Graph Convolutional Networks,"
  Priebe et al.) — theoretically relevant to GNN failure modes, but published 2020-10-25,
  outside the task's 2021-2026 window. Excluded on date grounds, flagged here in case the
  window constraint is later relaxed — it would otherwise be a strong candidate for Topic 1.
- A large number of generic "ML for stock/crypto/exchange-rate price prediction" papers were
  surfaced across every topic-area search (see raw search results) and were judged topically
  adjacent but not plausibly citable in this paper's related-work/discussion sections; these
  were deliberately excluded per the task's "genuine relevance over raw volume" instruction.

---

## Summary counts

| Topic area | Papers downloaded |
|---|---|
| 1. GNN for financial/commodity forecasting, learned-adjacency failure modes | 5 |
| 2. VMD/decomposition + deep learning | 4 |
| 3. Base-rate/constant-forecast/directional-accuracy artifacts | 2 |
| 4. Copper/industrial metal forecasting specifically | 2 |
| 5. Weak-form market efficiency on commodities/metals | 2 |
| 6. Reproducibility/rigor in financial ML | 2 |
| **Total** | **17** |


---

See also literature_review/shared/ for papers relevant to both paper1 and paper2 (arXiv 2409.08355, 2409.08356, 2607.12248 were downloaded independently by both literature passes and have been deduped into a single shared copy).
