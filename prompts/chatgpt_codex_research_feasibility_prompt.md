# Research & Feasibility Deep-Dive Prompt (for ChatGPT / Codex with web browsing)

> **How to use this file:** Paste everything below the line into ChatGPT (a model/agent with live web access — e.g. a Deep Research mode, Codex with browsing, or GPT-5 with browsing tools enabled). Do not summarize or paraphrase it before pasting — the context is intentionally exhaustive so the model does not need to guess at anything about the project. If the tool supports file upload, you may also attach this file directly instead of pasting.

---

## PROMPT START

You are acting as a **senior quantitative-finance research analyst and ML literature reviewer**, conducting due diligence for a B.Tech Capstone research project. You have live web/browsing access, including arXiv, SSRN, Google Scholar, journal publisher sites (Elsevier ScienceDirect, Wiley, Springer, MDPI), FRED, and general web search. Use it exhaustively — this is a real research task with a real deadline, not a hypothetical. Do not rely on memorized/prior knowledge alone; every non-trivial factual claim in your report must be backed by a source you actually found and can link to, with the date you found it.

### 1. Project context (read fully before starting)

This project originally built a hybrid deep-learning architecture called **VMD-MFGNN** for copper price forecasting: Variational Mode Decomposition (VMD) splits the daily copper price series into frequency-decomposed "modes," and a per-frequency-band Graph Neural Network (GNN) learns cross-band relationships via a learned adjacency matrix (bilinear dot-product attention over learned node embeddings — the same mechanism family as Graph WaveNet (2019) and MTGNN (2020)).

That project is **finished and will not change** — do not spend time re-investigating or second-guessing it. Its outcome: the learned graph adjacency was diagnosed to collapse into a uniform, uninformative structure (an isotropic embedding-norm collapse driven by weight decay overpowering a very weak task gradient — task gradient at those parameters measured on the order of 1e-5 per element). A targeted fix (removing weight decay from the graph embeddings, L2-normalizing them) mechanically stopped the collapse but the embeddings still stayed within ~2% (L2 distance) of random initialization — i.e., the graph never learned anything, fix or no fix. Separately, a constant-forecast/base-rate-identity diagnostic was built and found that several of the project's own "high directional accuracy" results were artifacts — models predicting a constant sign that happened to match the test period's majority class, not real skill. All of this was written into a full, peer-reviewed, submission-ready negative-result paper (already complete, do not touch).

**The live, active project — the one this research task is for — is a second, new paper**, code-named internally **"CuBench."** Its premise, decided after extensive prior literature research (summarized below so you don't have to re-derive it, though you should independently verify and extend it): abandon the graph-neural-network approach entirely, and instead build a **gradient-boosted tree model (LightGBM primary, with XGBoost/CatBoost/Random Forest as comparison points)** whose *primary* prediction target is **volatility and return-distribution forecasting** (not point-direction prediction), using four feature blocks:

- **Block A — Trend/technical**: momentum at multiple lookback windows (5/21/63/126/252 days), EWMA crossovers, return skew/kurtosis, distance-from-moving-average.
- **Block B — Carry/curve**: copper futures term-structure slope (front-month vs. deferred-month contract spread — i.e., contango/backwardation), motivated by the commodity factor literature (Bakshi, Gao & Rossi, *Management Science*, 2019; Yang 2013) which finds carry/roll-yield and momentum are validated, peer-reviewed, out-of-sample commodity risk factors, reportedly *strongest specifically for industrial metals* among commodity sectors.
- **Block C — Macro**: a deliberately small set — USD Index (DXY), 10-Year TIPS real yield (FRED series `DFII10`), ICE BofA High Yield OAS credit spread (FRED series `BAMLH0A0HYM2`), and China Manufacturing PMI (Caixin and/or official NBS).
- **Block D — Volatility/regime**: GARCH(1,1) and GJR-GARCH conditional volatility (expanding-window, monthly refit), HAR-RV (Heterogeneous Autoregressive Realized Volatility) components, realized-volatility term structure, VIX level/changes, and a discretized volatility-regime label.

Prediction targets, in priority order: **(T1, primary) realized volatility at h-day horizons; (T2) return quantiles via quantile regression/LightGBM's native quantile objective; (T3, secondary/honest-reporting) return direction/sign**, at horizons h ∈ {1, 5, 22} trading days.

**Why this pivot, briefly** (for your context, not something to re-litigate): six parallel literature-research tracks converged independently on this direction. Key supporting findings already gathered:
- A large-scale study (~18M daily observations, 10,000+ US securities, expanding-window 2001–2023 backtest) found gradient-boosted trees (CatBoost) decisively beat zero-shot and fine-tuned time-series foundation models (Chronos, TimesFM) on daily financial return prediction.
- A formal London Metal Exchange market-efficiency study (Box-Pierce, Lo-MacKinlay variance ratio, Wright's rank/sign VR, Kim's wild bootstrap VR tests) **fails to reject the random-walk hypothesis for copper specifically** (and all base metals except lead) — meaning daily copper direction is close to weak-form efficient, a real ceiling on point-direction forecasting.
- A controlled 918-experiment study (9 models × 3 asset categories × 2 horizons) found mean directional accuracy of 50.08% — statistically indistinguishable from chance.
- The practitioner/trading-desk literature is unambiguous that volatility is far more forecastable than direction, and that real trading desks build "trend + carry + regime" signal ensembles rather than one complex end-to-end direction predictor.
- A credible, defensible target range for a paper in this space (per publication-strategy research into ICAIF-accepted papers and similar venues) is: directional accuracy in the low-to-mid 50s%, out-of-sample R² of roughly 0.5–3% on returns, Information Coefficient of 0.03–0.08 — **not** 60%+ DA or double-digit R², which would itself read as a red flag (likely leakage) to an informed reviewer.

The evaluation protocol already planned: expanding-window walk-forward validation with annual refits across ~10 non-overlapping out-of-sample years (2015–2024), embargo periods at fold boundaries, Diebold-Mariano and Pesaran-Timmermann significance tests with Holm-Bonferroni multiple-testing correction, Deflated Sharpe Ratio and Probability of Backtest Overfitting (Bailey & López de Prado), transaction-cost-adjusted backtesting, regime-segmented analysis (volatility terciles + macro-calendar regimes: pre-2020 / COVID 2020-21 / 2022 shock / 2023-25), block-wise feature ablations, and a formal base-rate/constant-forecast artifact check applied to every reported result cell.

Data: daily copper price data (Yahoo Finance / COMEX HG futures), 2010–2025, ~15 years. Compute: this whole pipeline is designed to be lightweight — LightGBM on ~4,000 rows × ~50-60 features trains in under a second; the full experimental grid is estimated at 20-60 minutes on a laptop CPU (no GPU required).

Target venue: ICAIF (International Conference on AI in Finance) or an arXiv preprint; academic supervisor is at NIT Rourkela, India; this is a B.Tech Capstone Project (CS4095) with a 4-month execution window (August–November 2026).

### 2. Your task

Conduct **exhaustive, source-verified research** across arXiv, SSRN, Google Scholar, and relevant journal sites to produce a full **feasibility and rigor-strengthening report** for this plan. Specifically:

#### A. Validate and stress-test the methodology
1. Search arXiv (categories q-fin.ST, q-fin.CP, q-fin.TR, cs.LG, stat.ML) and SSRN for the **most recent (2024–2026)** papers on: gradient boosting for commodity/financial return and volatility forecasting; LightGBM/XGBoost/CatBoost applied specifically to copper or industrial metals; HAR-RV and GARCH-family volatility forecasting benchmarks; commodity carry/momentum factor models (extend beyond Bakshi-Gao-Rossi/Yang — find anything more recent that specifically revisits or extends this factor set); quantile/distributional forecasting for commodity prices.
2. For each paper found, extract: exact methodology, exact dataset/period used, exact reported metrics (do not round or approximate — quote the paper's own numbers), and whether their evaluation protocol has any of the known weaknesses this project is trying to avoid (single train/test split, no multiple-testing correction, no transaction-cost adjustment, no leakage check, claims of implausibly high accuracy).
3. Explicitly flag: does anything found suggest this exact combination (LightGBM + carry/trend/macro/vol features + volatility-primary target + copper specifically) has **already been done** by someone else? If so, cite it precisely, describe how it differs from this project's plan (if at all), and assess whether this project's angle is still a genuine, non-duplicative contribution, or needs to be repositioned.
4. Search specifically for **any published critique or replication-failure analysis** of the Bakshi-Gao-Rossi / Yang commodity factor models, or of GARCH/HAR-RV as applied to industrial metals — i.e., look for reasons this methodology might NOT work as expected, not just supporting evidence. Actively look for disconfirming evidence.

#### B. Data feasibility — verify, don't assume
5. Investigate whether **individual COMEX copper futures contract-month data** (needed for the carry/term-structure feature, Block B) is actually obtainable for free (e.g., via `yfinance` tickers like `HG=F`, `HGZ25.CMX`, or similar contract-month-specific tickers) with enough historical depth (ideally back to 2010) and enough contracts (front month + at least one deferred month) to compute a genuine roll-yield/carry signal. If yfinance does not reliably serve this, identify what free or low-cost alternative sources exist (e.g., Investing.com, Quandl/Nasdaq Data Link free tiers, CME's own free data offerings, Barchart free tier) and their actual historical coverage and access mechanics (API vs. manual download, rate limits, registration requirements).
6. Verify current, actual access mechanics (not assumed) for: FRED series `DFII10` and `BAMLH0A0HYM2` (confirm these series still exist under these exact codes, are still updated, and are freely downloadable via the FRED API or CSV endpoint); a free, reliably-updated China Manufacturing PMI series (Caixin and/or official NBS) — identify the actual best free source as of today, since PMI is often paywalled or delayed on free tiers.
7. Investigate what free daily VIX data source is most reliable for a multi-year (2010-2025) pull (FRED `VIXCLS` vs. Yahoo Finance `^VIX` vs. CBOE's own historical data page) and note any known data-quality caveats (missing days, revisions).

#### C. Publication/venue feasibility
8. Search for the **current, actual ICAIF submission deadline, format requirements, and page limits** for the next upcoming edition (if ICAIF 2026 or 2027's call for papers is not yet public, find the most recent past edition's CFP as a proxy and note the typical annual timing pattern). Note: do not assume last year's dates carry over — verify.
9. Search for **2-3 alternative venues** appropriate for a rigorous-evaluation commodity/financial-ML paper if ICAIF's timeline doesn't fit a 4-month student project window — e.g., workshop tracks at NeurIPS/ICML/KDD focused on time series or finance, *Journal of Forecasting*, *Journal of Financial Data Science*, *Journal of Commodity Markets*, *Finance Research Letters* — and note their typical review timelines, since a B.Tech Capstone evaluation in November 2026 needs at least a submission (not necessarily acceptance) by then.
10. Verify current arXiv endorsement requirements for a first-time submitter in category q-fin.ST or q-fin.CP — is an endorsement still required, what is the actual current process and typical latency, and is there a way to shortcut it (e.g., submitting to a category that doesn't require endorsement first, or using an SSRN-first strategy).

#### D. Strengthen the rigor apparatus
11. Search for the **most current, correct implementation references** for: Deflated Sharpe Ratio and Probability of Backtest Overfitting (confirm the canonical formula/citation — Bailey & López de Prado — and find any well-maintained open-source Python implementation, e.g., in `mlfinlab`, `PyPortfolioOpt`, or standalone repos, so this doesn't need to be implemented from scratch incorrectly).
12. Search for the standard, current best-practice implementation of Diebold-Mariano and Pesaran-Timmermann tests in Python (existing library vs. hand implementation) and any known correction/caveat papers (e.g., small-sample adjustments to the DM test, since the ~10-fold walk-forward design here has a modest number of comparisons per cell).
13. Search for recent (2024-2026) papers proposing or critiquing "constant-forecast" / "base-rate-identity" / degenerate-directional-accuracy diagnostics — i.e., does prior published work already name and formalize what this project's team independently discovered? If so, this changes how the project should cite/frame that contribution (as a formalization/application, not a novel invention) — find this precisely, don't let the project overclaim novelty it doesn't have.

### 3. Output format

Produce a single, well-organized markdown report with these exact sections:

1. **Executive Summary** (under 400 words): is this plan feasible as designed, what (if anything) must change, and what is the single biggest risk you found.
2. **Section A findings** — methodology validation, with a table of the most relevant 8-15 papers found (columns: citation, year, venue, dataset/period, method, key reported metric, relevance/risk note).
3. **Section B findings** — data feasibility, with a clear yes/no/partial verdict for each of the data sources investigated, and concrete fallback recommendations where a "no" or "partial" was found.
4. **Section C findings** — venue/timeline feasibility, with concrete dates where found.
5. **Section D findings** — rigor-tooling recommendations, with specific library/implementation pointers.
6. **Full source list** with links, and the date each was accessed/found.

Be exhaustive rather than fast — this report will directly determine what gets built over the next 4 months, so a missed feasibility blocker found late is much more costly than extra research time now. Where you are uncertain or a source is ambiguous/paywalled/unverifiable, say so explicitly rather than guessing — flag it as "UNVERIFIED, needs manual check" rather than presenting an assumption as fact.

## PROMPT END
