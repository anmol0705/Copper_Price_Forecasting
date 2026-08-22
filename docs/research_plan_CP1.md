# Research Plan — Capstone Project-I (CS4095)
### Autumn Semester 2026–27

---

## 1. Student Details

| Field | Details |
|---|---|
| **Name of the Candidate** | Anmol Jain |
| **Registration Number** | *[to be filled in]* |
| **Supervisor(s)** | Dr. Sibarama Panigrahi |
| **Department** | Computer Science & Engineering, NIT Rourkela |
| **Tentative Title of the Research Project** | A Leakage-Audited, Regime-Aware Machine Learning Benchmark for Volatility and Return Forecasting in Industrial Metal Markets: The Case of Copper |
| **Broad Area of Research** | Machine Learning for Financial/Commodity Time-Series Forecasting, Applied Quantitative Finance |

---

## 2. Brief Problem Statement

### 2.1 Research Problem

Copper is often called "Dr. Copper" for its sensitivity to global industrial activity, and it sits at the center of the energy-transition supply chain (electric vehicles, grid infrastructure, renewable generation). Reliable short-to-medium-horizon forecasting of its price behaviour is therefore of genuine economic interest — yet the machine learning literature that claims to solve this problem is, on close inspection, methodologically fragile. A large fraction of published commodity/financial forecasting papers report high directional accuracy or R² using evaluation protocols that do not survive scrutiny: single train/test splits instead of multi-regime walk-forward validation, no correction for multiple-hypothesis testing across model/horizon combinations, and no check for a specific but under-recognised failure mode in which a model's "accuracy" is a trivial artifact of predicting the same direction repeatedly and coincidentally matching the test period's majority class — a pattern indistinguishable from real skill unless explicitly tested for.

This project addresses two entangled research problems:

1. **The forecasting problem itself**: given the current state of commodity/financial econometrics — which shows that copper does not reject weak-form market efficiency under formal statistical testing (variance-ratio tests on LME base metals), and that daily commodity returns behave close to a random walk in direction while exhibiting genuine, exploitable structure in volatility (via well-established volatility-clustering mechanisms) — what is the best-evidenced, honestly-evaluated modelling approach for copper, and what is a realistic, defensible performance ceiling?

2. **The evaluation-methodology problem**: how should a forecasting claim in this domain be constructed so that it is *credible* — i.e., survives regime-segmented, walk-forward, multiple-testing-corrected, transaction-cost-adjusted scrutiny — rather than merely *impressive-looking* on a single, potentially leakage-prone test split?

### 2.2 Motivation and Relevance

Copper price volatility has direct downstream consequences for industrial procurement, hedging desks, and policy planning around the energy transition, given copper's outsized role in electrification. At the same time, the broader field of financial machine learning has a well-documented replication crisis: a significant share of published "beats-the-market" results are attributable to data snooping, undisclosed multiple comparisons, or leakage rather than genuine predictive skill. A project that (a) builds a forecasting system grounded in validated econometric and market-microstructure signal sources specific to industrial metals, and (b) applies an unusually rigorous, pre-registered, regime-aware evaluation protocol to it, contributes something the field currently under-supplies: not another architecture claiming an edge, but a transparently-evaluated benchmark whose positive and negative findings can both be trusted.

### 2.3 Research Gap Identified

An extensive multi-track literature review (covering copper market fundamentals, practitioner/trading-desk methodology, rigorous financial econometrics, modern machine-learning time-series architectures, macro/geopolitical leading indicators, and publication-standard requirements at top venues such as ICAIF) surfaced the following specific, actionable gaps this project is positioned to fill:

- **Volatility vs. direction gap**: the practitioner and econometrics literature is unambiguous that volatility/regime is substantially more forecastable than directional sign, yet most published commodity-forecasting ML papers still target point/direction prediction as the primary (or only) objective.
- **Underused validated factors for industrial metals specifically**: carry/roll-yield and momentum factors (Bakshi, Gao & Rossi, *Management Science*, 2019) are documented, peer-reviewed, out-of-sample-validated predictors that are explicitly reported as strongest for industrial metals among commodity sectors — yet are rarely used as engineered features in ML pipelines for this asset class, which tend to rely on raw price/technical inputs alone.
- **Architecture mismatch for the data regime**: recent large-scale, rigorous benchmarks (including an 18-million-observation, decade-spanning study) show that gradient-boosted tree ensembles decisively outperform both deep sequence models and time-series foundation models (e.g., Chronos, TimesFM) on daily financial return prediction — a small-sample, low-signal-to-noise regime that does not favour large parametric architectures. Much of the applied literature nonetheless defaults to deep learning without this baseline comparison.
- **Absence of a credible-claim standard applied to commodities**: tools that have become mandatory at top quantitative-finance venues for defending a claimed edge — the Deflated Sharpe Ratio, the Probability of Backtest Overfitting, multiple-testing-corrected statistical tests, and regime-segmented (not single-split) walk-forward validation — are inconsistently applied in the commodity-forecasting sub-literature specifically, as compared to equities.
- **No systematic base-rate-artifact check**: no standard diagnostic in the reviewed literature explicitly tests whether a model's reported directional accuracy is a trivial restatement of the test period's majority-class frequency (a constant-sign-prediction artifact) — a gap this project's own prior diagnostic work has already shown to be a real, silent contaminant of result tables, including in this project's own earlier experimentation.

---

## 3. Objectives for Autumn Semester 2026–27

1. **O1** — Construct a leakage-audited, causally-correct feature pipeline for daily copper price data (2010–2025) combining four validated signal families: price-trend/momentum, futures carry/term-structure, macro-financial indicators, and volatility/regime state.
2. **O2** — Build and rigorously benchmark a gradient-boosting-based forecasting system (Carry-Trend-Regime, "CTR") against a comprehensive baseline roster: statistical nulls, classical econometric models (ARIMA, GARCH-family, HAR-RV), linear models, alternative tree ensembles, and deep sequence models.
3. **O3** — Design and pre-register (before any model training) a complete, regime-segmented, walk-forward evaluation protocol, including multiple-testing-corrected significance testing, transaction-cost-adjusted backtesting, and a formal check for base-rate/constant-forecast artifacts in every reported result.
4. **O4** — Quantify, with statistical rigor, the realistic predictive ceiling for copper price direction, return magnitude, and volatility at 1-, 5-, and 22-day horizons, and identify which specific engineered feature blocks (trend, carry, macro, regime) contribute genuine, ablation-verified incremental value.
5. **O5** — Produce a complete manuscript reporting the benchmark, methodology, and findings, prepared to the standard required for submission to a recognised venue in quantitative finance / AI-in-finance (e.g., ICAIF) or as an arXiv preprint.

---

## 4. Detailed Month-wise Work Plan (August – November 2026)

| Month | Phase | Key Tasks | Output |
|---|---|---|---|
| **August 2026** | Foundation & Protocol Design | Literature consolidation and gap analysis (already substantially completed as groundwork for this plan); data-source feasibility checks (futures term-structure availability, macro data sources — FRED real yields, credit spreads, China PMI); extension of the existing data pipeline to source OHLC data (not close-only) to enable realized-volatility estimators; **write and freeze a pre-registration document** specifying all targets, feature blocks, models, and evaluation criteria before any model training begins, to guard against post-hoc result-shaping | Feasibility report; committed pre-registration document; extended raw data pipeline |
| **September 2026** | Feature Engineering & Model Implementation | Implement the four feature blocks — trend/momentum, carry/term-structure, macro-financial, volatility/regime (GARCH/GJR-GARCH with expanding-window refit, HAR-RV components); implement and unit-test a causal-leakage check (feature at time *t* uses only information available at or before *t*); implement the full model roster (statistical nulls, ARIMA, GARCH-family, HAR-RV, ElasticNet, LightGBM/XGBoost/CatBoost/Random Forest, LSTM/Transformer baselines); build the expanding-window, annually-refit walk-forward evaluation engine with embargo periods at fold boundaries | Complete feature pipeline with passing leakage tests; full model implementation; walk-forward evaluation engine |
| **October 2026** | Experimentation, Evaluation & Rigor Analysis | Execute the complete experimental grid across all models, targets (volatility, return quantiles, direction), and horizons across ~10 non-overlapping out-of-sample years; compute Diebold-Mariano and Pesaran-Timmermann tests with Holm-Bonferroni correction; compute Deflated Sharpe Ratio and Probability of Backtest Overfitting; run transaction-cost-adjusted backtests; run regime-segmented analysis (volatility-tercile and macro-calendar regimes); run block-wise ablations (price-only → +trend → +carry → +macro → +regime) to isolate each feature block's marginal contribution; apply the base-rate/constant-forecast artifact diagnostic to every reported cell; SHAP-based interpretability analysis segmented by regime | Complete, statistically-validated results tables; ablation study; interpretability analysis; internal draft findings summary |
| **November 2026** | Manuscript Preparation, Internal Review & Milestone Submission | Draft the full manuscript (introduction, related work, data and feature methodology, evaluation protocol, results, ablations, limitations); internal review pass for methodological soundness, statistical correctness, and clarity; revise based on supervisor feedback; finalize figures/tables and recompute all reported numbers directly from raw output files as a correctness check; prepare final Capstone Project-I submission and presentation materials | Complete manuscript draft (submission-ready or near-submission-ready); final CP-I presentation; supervisor sign-off |

---

## 5. Expected Deliverables by November 2026

- A **fully implemented, reproducible forecasting and evaluation pipeline** (data ingestion, feature engineering, model training, walk-forward evaluation, statistical testing) for copper price and volatility forecasting.
- A **pre-registered, leakage-audited experimental benchmark** covering ~12 models across 3 prediction targets (volatility, return quantiles, direction) and 3 horizons (1, 5, 22 days), evaluated across roughly a decade of non-overlapping out-of-sample periods spanning multiple genuine market regimes.
- A **quantified, statistically rigorous account of predictive skill** for copper — including honest, evidence-calibrated reporting of where no model beats a naive baseline, wherever that is the true finding — together with a **feature-block ablation study** identifying which of trend, carry, macro, and volatility/regime signals contribute genuine incremental value.
- A **manuscript** (conference or preprint-ready) reporting the full methodology and findings, targeting a recognised quantitative-finance/AI-in-finance venue.
- A **generalisable evaluation-rigor contribution**: a formalized, reusable diagnostic for detecting base-rate/constant-forecast artifacts in directional-accuracy claims, applicable beyond this specific project.

---

## 6. Milestones and Evaluation Points

| Milestone | Target Date | Evaluation Criteria |
|---|---|---|
| **Mid-Semester Milestone** | End of September 2026 | Feature pipeline complete and leakage-verified; full model roster implemented; walk-forward evaluation engine operational and tested on a pilot subset of the data |
| **Pre-Final Milestone** | End of October 2026 | Full experimental grid executed; all statistical significance, ablation, and rigor analyses (DSR/PBO, DM/PT tests, regime segmentation, base-rate artifact checks) complete; internal findings summary reviewed with supervisor |
| **Final Milestone** | November 2026 (before CP-I examination) | Complete manuscript draft finalized; all reported results independently recomputed from raw outputs for correctness; CP-I presentation and documentation submitted |

---

## 7. Supervisor's Recommendation

*"The proposed research plan is realistic, well-structured, and recommended for implementation during Autumn Semester 2026-27."*

**Signature:** _______________________
**Name:** Dr. Sibarama Panigrahi
**Date:** _______________________
