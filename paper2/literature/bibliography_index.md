# CuBench (Paper 2) — Literature Review Bibliography Index

Compiled: 2026-09-03. All arXiv entries were located and verified via the arXiv MCP (`search_papers`, `get_abstract`, `download_paper`) during this session. PDFs were then retrieved directly from `arxiv.org/pdf/<id>` (via `curl --ssl-no-revoke`, working around a local schannel/TLS revocation-check issue — the same class of environment hazard already documented in `docs/cubench_implementation_plan.md` §1.1 for FRED/`requests`) and saved to this directory. The MCP `download_paper` tool itself only returns extracted text into its own local cache, not a PDF file placed in an arbitrary directory, so it was used here for verification/metadata and `get_abstract` for citation-grade metadata, while `curl` supplied the actual PDF bytes.

Every citation below was checked against a live `get_abstract` (or `search_papers`) result in this session — none is transcribed from `docs/research/cubench_feasibility_report.md` without independent re-verification. Two corrections against that report surfaced in the process: the FinVerse paper (arXiv:2608.03259) has 12 named authors, first author Jaehoon Lee, not "Lee, J. et al."; and arXiv:2607.12248's actual title is "When Directional Accuracy Lies: A Base-Rate-Honest Benchmark for LoRA-Adapted TimesFM on Equity Forecasting" by the single author Taizhen Cheung, not a generic "Cheung, T. (2026)."

Filenames follow `<arXiv-id>_<first-author-surname>[_et_al]_<slug>.pdf`.

---

## Section 0 — Already identified by prior research, now downloaded (Step 1)

All 15 arXiv papers listed in `docs/research/cubench_feasibility_report.md` §6 were located, verified, and downloaded as PDF. All 15 succeeded — no arXiv ID in that list was unfetchable.

| # | Citation | arXiv ID | Local filename |
|---|---|---|---|
| 1 | Wang, Z. and Lu, X. (2024). "COMEX Copper Futures Volatility Forecasting: Econometric Models and Deep Learning." | [2409.08356](https://arxiv.org/abs/2409.08356) | `../shared/2409.08356_wang_lu_comex_copper_volatility_forecasting.pdf` (deduped; shared with paper1) |
| 2 | Wang, Z. and Li, X. (2024). "On the macroeconomic fundamentals of long-term volatilities and dynamic correlations in COMEX copper futures." (full author name per arXiv metadata: **Xinshu Li**) | [2409.08355](https://arxiv.org/abs/2409.08355) | `../shared/2409.08355_wang_li_macro_fundamentals_comex_copper.pdf` (deduped; shared with paper1) |
| 3 | Dudek, G., Kasprzyk, M., and Pełka, P. (2025). "Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting: Deterministic, Probabilistic, and Feature Importance Perspectives." | [2511.20105](https://arxiv.org/abs/2511.20105) | `2511.20105_dudek_kasprzyk_pelka_bitcoin_volatility_gbm.pdf` |
| 4 | Fang, X. and Ślepaczuk, R. (2026). "Volatility Forecasting and Return Prediction under Market Regimes: Evidence from High-Frequency Chinese Equity Data." | [2606.09478](https://arxiv.org/abs/2606.09478) | `2606.09478_fang_slepaczuk_volatility_return_market_regimes.pdf` |
| 5 | Cheung, T. (2026). "When Directional Accuracy Lies: A Base-Rate-Honest Benchmark for LoRA-Adapted TimesFM on Equity Forecasting." (single author: **Taizhen Cheung**) | [2607.12248](https://arxiv.org/abs/2607.12248) | `../shared/2607.12248_cheung_when_directional_accuracy_lies.pdf` (deduped; shared with paper1) |
| 6 | Brini, A. (2026). "Forecasting Realized Volatility with Time Series Foundation Models: A Comparison with Econometric Benchmarks." | [2607.05291](https://arxiv.org/abs/2607.05291) | `2607.05291_brini_rv_time_series_foundation_models.pdf` |
| 7 | Dudek, G., Orzeszko, W., and Fiszeder, P. (2025). "Probabilistic Forecasting Cryptocurrencies Volatility: From Point to Quantile Forecasts." | [2508.15922](https://arxiv.org/abs/2508.15922) | `2508.15922_dudek_orzeszko_fiszeder_probabilistic_crypto_volatility.pdf` |
| 8 | Muhammad, T. et al. (2026). "A Benchmark of Classical and Deep Learning Models for Agricultural Commodity Price Forecasting on A Novel Bangladeshi Market Price Dataset." (AgriPriceBD; 10 named authors) | [2604.06227](https://arxiv.org/abs/2604.06227) | `2604.06227_muhammad_et_al_agripricebd_commodity_benchmark.pdf` |
| 9 | Portnaya, V. (2026). "The Bounce Has No Direction: Sign, Magnitude, and the Microstructure of Equity Return Predictability." | [2606.29591](https://arxiv.org/abs/2606.29591) | `2606.29591_portnaya_bounce_has_no_direction.pdf` |
| 10 | Benhamou, E., Ohana, J-J., Etienne, A., Guez, B., Setrouk, E., and Jacquot, T. (2025). "Re-evaluating Short- and Long-Term Trend Factors in CTA Replication: A Bayesian Graphical Approach." | [2507.15876](https://arxiv.org/abs/2507.15876) | `2507.15876_benhamou_et_al_cta_replication_trend_factors.pdf` |
| 11 | He, P., Peters, G. W., Kordzakhia, N., and Shevchenko, P. V. (2024). "Multi-Factor Function-on-Function Regression of Bond Yields on WTI Commodity Futures Term Structure Dynamics." | [2412.05889](https://arxiv.org/abs/2412.05889) | `2412.05889_he_et_al_function_on_function_wti_term_structure.pdf` |
| 12 | Matsubara, T. (2024). "Wasserstein Gradient Boosting: A Framework for Distribution-Valued Supervised Learning." | [2405.09536](https://arxiv.org/abs/2405.09536) | `2405.09536_matsubara_wasserstein_gradient_boosting.pdf` |
| 13 | Lee, J. et al. (2026). "FinVerse: Financial Time-Series Benchmark." (12 named authors, first author **Jaehoon Lee**) | [2608.03259](https://arxiv.org/abs/2608.03259) | `2608.03259_lee_et_al_finverse_benchmark.pdf` |
| 14 | Amorin, A. P., Python, A., and Weisser, C. (2026). "Not All News Is Equal: Topic- and Event-Conditional Sentiment from Finetuned LLMs for Aluminum Price Forecasting." | [2603.09085](https://arxiv.org/abs/2603.09085) | `2603.09085_amorin_python_weisser_not_all_news_equal.pdf` |
| 15 | Lee, N. et al. (2024). "Metal Price Spike Prediction via a Neurosymbolic Ensemble Approach." | [2410.12785](https://arxiv.org/abs/2410.12785) | `2410.12785_lee_et_al_metal_price_spike_neurosymbolic.pdf` |

### Non-arXiv sources from the feasibility report's list — not fetchable via this MCP tool

These were named in the feasibility report's source list but are not arXiv papers, so the arXiv MCP cannot retrieve them. Flagged explicitly rather than skipped silently, per task instructions:

- **Angelidis, T., Sakkas, A., and Tessaromatis, N.** "Predicting Commodity Returns: Time Series vs. Cross Sectional Prediction Models." SSRN abstract 5205084 / *Journal of Commodity Markets* (2025). SSRN + journal-paywalled; no arXiv mirror found.
- **Bakshi, G., Gao, X., and Rossi, A. G.** "Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns." *Management Science* 65(2), 2019. Published journal article via INFORMS; no arXiv preprint located (searched `au:Rossi commodity risk premium futures` in this session — no match).
- **Yang, F.** "Investment Shocks and the Commodity Basis Spread." *Journal of Financial Economics*, 2013. ScienceDirect-hosted; no arXiv preprint.
- **Bailey, D. H. and López de Prado, M.** "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." SSRN abstract 2460551 / *Journal of Portfolio Management*, 2014. SSRN-hosted, no arXiv mirror. (Its methodological descendants and applications *are* on arXiv — see Section 3 below, which substitutes for direct access to the original.)

The following items from the report's "Other academic and methodology sources" list are **software/documentation references, not papers**, and are noted here for completeness rather than treated as unfetchable citations: `pypbo` (GitHub), `dieboldmariano` (PyPI), statsmodels docs, `rugarch::DACTest` docs, LightGBM/CatBoost/XGBoost parameter docs.

---

## Section 1 — Realized volatility forecasting: HAR-RV, GARCH-family, and ML/GBM comparisons

This is the paper's central empirical question (RQ1) and the section with the most important new findings.

**Christensen, K., Siggaard, M., and Veliyev, B. (2026). "A machine learning approach to volatility forecasting."** arXiv:[2601.13014](https://arxiv.org/abs/2601.13014). `2601.13014_christensen_siggaard_veliyev_ml_volatility_forecasting.pdf`
Compares regularization, regression-tree, and neural-network ML models against multiple HAR specifications on realized variance of Dow Jones constituents, with **minimal hyperparameter tuning**. Finds ML is competitive and **beats the HAR lineage even using only daily/weekly/monthly RV lags**, with gains more pronounced at longer horizons, attributed to ML's higher persistence approximating RV's long memory. Directly relevant to RQ1: it is head-to-head evidence *for* gradient-boosting-style methods over HAR, in tension with D2's framing (see flag below) and with Audrino & Chassot immediately below. Also supports the plan's own light-touch HPO policy (§3.4): even "minimal tuning" was enough for ML to win here.

**Audrino, F. and Chassot, J. (2024). "HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning."** arXiv:[2406.08041](https://arxiv.org/abs/2406.08041). `2406.08041_audrino_chassot_hard_to_beat_rolling_windows.pdf`
Tests HAR vs. ML (with extensive hyperparameter tuning) across 1,455 stocks and finds that **with a correctly specified HAR fitting scheme — training-window length and re-estimation frequency — HAR is not beaten by ML**, evaluated on QLIKE, MSE, and realized utility. Restricted to RV + VIX predictors only. This is close to CuBench's A0/A1 ablation rungs and QLIKE-primary metric. **Central methodological flag: see below.**

**Chung, S. (2024). "Modelling and Forecasting Energy Market Volatility Using GARCH and Machine Learning Approach."** arXiv:[2405.19849](https://arxiv.org/abs/2405.19849). `2405.19849_chung_energy_volatility_garch_ml.pdf`
GARCH-family vs. ML volatility forecasting for crude oil, gasoline, heating oil, natural gas, using financial/macro/environmental predictors and SHAP for interpretability. Finds ML has superior OOS accuracy but with a bias pattern (ML underpredicts, GARCH overpredicts), suggesting a hybrid. A commodity-market analogue (energy rather than metals) of exactly CuBench's GARCH-vs-tree-ensemble comparison, useful as a same-asset-class precedent that a GBM-over-GARCH result is plausible even where HAR-over-GBM (Audrino & Chassot) might not be.

### Central flag on RQ1 / D2

`docs/cubench_implementation_plan.md` D2 currently cites only pro-HAR evidence (Wang & Lu 2024, Brini 2026) to justify treating Log-HAR as "the model to beat." The two new papers above show the HAR-vs-ML result is **actively contested in 2024–2026 literature, and the discriminating variable is not model class but HAR's fitting scheme.** Audrino & Chassot's entire finding is that ML only loses to HAR once HAR is given a *properly re-tuned* training window and refit frequency — a sloppily-fit HAR loses. Christensen et al. get the opposite result (ML wins) with the more standard fitting setup. CuBench's `har_rv` spec (§3.2) uses a single fixed choice — expanding window, **annual** refit — chosen for engineering simplicity, not validated against alternative refit frequencies the way Audrino & Chassot's paper says is decisive. This means RQ1's outcome may be partly an artifact of an under-specified nuisance choice (HAR refit cadence) rather than a clean test of "gradient boosting vs. the best HAR." Recommendation: either (a) explicitly note this as a known limitation in the pre-registration's RQ1 section, citing both papers, or (b) add a HAR refit-frequency robustness check (e.g., monthly vs. annual refit) as a cheap addition to the ablation/robustness suite before the RQ1 headline claim is written. This does not require re-opening D2's conclusion, but the pre-registration text should acknowledge the fitting-scheme confound rather than presenting HAR-vs-GBM as a settled question in the literature.

---

## Section 2 — Gradient boosting (LightGBM/XGBoost/CatBoost) for financial/commodity time series, benchmarked against deep learning / foundation models

Coverage here largely overlaps Section 1 for this specific paper's needs (Christensen et al. and Chung above are GBM-relevant too). No new arXiv paper was found that is a large-scale, rigorous, commodity-specific GBM-vs-deep-learning benchmark beyond what the feasibility report already located (Dudek et al. Bitcoin GBM papers, Muhammad et al. AgriPriceBD, Brini's TSFM comparison — all in Section 0). One general-purpose tabular-ML benchmark reference is added for context:

**Erickson, N. et al. (2025). "TabArena: A Living Benchmark for Machine Learning on Tabular Data."** arXiv:2506.16791 (abstract reviewed, **not downloaded** — general ML-benchmarking infrastructure paper, not finance-specific; included in this index as a candidate citation for the "GBDTs remain the standard for tabular data" framing claim, but judged not essential enough to warrant a PDF given the paper's already-tight scope). Not saved locally; cite via abstract if needed, or skip.

No PDF added for this section beyond what's cross-referenced in Sections 1 and 4.

---

## Section 3 — Backtest overfitting, Deflated Sharpe Ratio, Probability of Backtest Overfitting; and statistical-testing rigor (DM/HLN under overlapping targets)

**Santoni, M. L., Jouanne, V., and Scullin, M. L. (2026). "Equity Strategy Backtesting: Luck or Edge? The MinervaScore as a Statistical Robustness Grade."** arXiv:[2608.23808](https://arxiv.org/abs/2608.23808). `2608.23808_santoni_et_al_minervascore_backtest_robustness.pdf`
Introduces a composite post-selection robustness score combining **DSR, PBO, Superior Predictive Ability, and Minimum Track Record Length** with a regime-stability diagnostic, calibrated on 359,062 production backtest records and validated (AUROC 0.989) on synthetic data with known ground truth. Directly extends the exact Bailey/López de Prado line of work CuBench's §4.3 relies on (`pypbo`), and honestly reports that in a pre-registered real-market test the score showed **no significant forward relationship** in a population with limited surviving edge — itself a useful calibration point for how skeptical CuBench should be of its own headline Sharpe numbers. Directly citable as recent practice for combining DSR+PBO into one reporting artifact, and as evidence that even a well-validated composite score can fail to predict forward performance on real markets — reinforcing CuBench's own pre-committed "PBO > 0.5 ⇒ report overfit regardless of Sharpe" rule (§4.3).

**Jacquier, A., Muhle-Karbe, J., and Mulligan, J. (2025). "In-Sample and Out-of-Sample Sharpe Ratios for Linear Predictive Models."** arXiv:[2501.03938](https://arxiv.org/abs/2501.03938). `2501.03938_jacquier_muhlekarbe_mulligan_in_out_sample_sharpe.pdf`
Derives closed-form in-sample-vs-out-of-sample Sharpe degradation for linear strategies, explicitly illustrated with a **commodity futures simulation following Gårleanu & Pedersen's methodology**. Finds OOS "replication ratio" shrinks for strategies built from many weak signals across many assets (vs. few strong signals) and improves with more training data. Directly relevant to CuBench's `elasticnet` (Tier 2) baseline and to interpreting why a many-feature (68-feature) gradient-boosting strategy might show more in-sample/OOS Sharpe degradation than a parsimonious HAR/GARCH baseline — a mechanism worth citing when discussing the cost-curve and DSR results in §4.4.

**Coroneo, L. and Iacone, F. (2024). "Testing for equal predictive accuracy with strong dependence."** arXiv:[2409.12662](https://arxiv.org/abs/2409.12662). `2409.12662_coroneo_iacone_dm_test_strong_dependence.pdf`
Shows the Diebold-Mariano test's **power decreases as loss-differential autocorrelation increases, and beyond a threshold the test has essentially no power and spuriously rejects a correct null hypothesis of equal accuracy.** **Direct methodological flag for §4.3** (see below) — CuBench's plan to use HAC/Newey-West variance with bandwidth h−1 plus the HLN small-sample correction for the overlapping h∈{5,22} targets addresses bias in the variance estimator, but this paper's finding is about a different failure mode (power loss / spurious rejection under strong dependence) that HLN alone does not necessarily fix. This should be read before finalizing `src/cubench/stats_tests.py::dm_test_hln()`.

---

## Section 4 — Point-in-time data handling / publication-lag leakage in macro-financial feature engineering

This section has the strongest and most directly on-topic new find of the whole search: a **copper-specific real-time forecasting paper**.

**Bastianin, A., Rossini, L., and Tonni, L. (2025). "A Real-Time Framework for Forecasting Metal Prices."** arXiv:[2512.16521](https://arxiv.org/abs/2512.16521). `2512.16521_bastianin_rossini_tonni_real_time_metal_prices.pdf`
Builds a genuine real-time forecasting framework for **aluminum, copper, nickel, and zinc**, explicitly reconstructing the real-time information set available to forecasters by combining daily financial variables with **first-release** (not revised) macro indicators and nowcasting to handle publication lags. Finds manufacturing-activity indicators (new orders, capacity utilization) tied to primary-metals demand **significantly improve copper and aluminum forecast accuracy**, with futures/survey-based benchmarks underperforming the real-time econometric models. This is the single most directly relevant new paper found in this pass for two reasons at once: it is copper-specific (Topic 6-adjacent) and it is a rigorous point-in-time/publication-lag methodology paper (Topic 4) — exactly the combination CuBench's §1.4/§2.3 point-in-time join procedure is designed around. It also substantively strengthens the case for D5 (PPI/INDPRO inclusion) with a copper-specific result rather than the cross-market analogy CuBench currently relies on (Wang & Li 2024, which is COMEX-copper but not built as a real-time/PIT exercise).

**Agyekum, L. and Obese, O. (2026). "Forecasting in the Fog: Real-Time versus Revised-Data Evidence on Machine Learning's Edge over the Phillips Curve."** arXiv:[2608.09033](https://arxiv.org/abs/2608.09033). `2608.09033_agyekum_obese_forecasting_in_the_fog_phillips_curve.pdf`
Directly tests whether an ML-over-econometric-baseline advantage (documented on revised data) survives when models are trained/evaluated on **real-time ALFRED vintages** instead. Finds real-time/revised accuracy differences are mostly small and DM-indistinguishable, but that **SHAP feature-importance rankings computed in-sample on revised data can be a substantial hindsight artifact** (one feature's importance rank swings from 1st to 6th between revised and real-time data). This is a concrete cautionary precedent for CuBench's own SHAP-based feature-importance reporting (`shap>=0.46` in `requirements_cubench.txt`): even though CuBench's monthly-macro PIT join (§2.3) is designed to avoid the underlying leakage, this paper is evidence that **feature-importance interpretation, not just point-forecast accuracy, is where revision/real-time effects bite hardest** — worth a sentence in the paper's discussion of feature importance results.

**Carriero, A., Pettenuzzo, D., and Shekhar, S. (2026). "MACROCAST: A Vintage-Consistent Time Series Foundation Model for Real-Time Macroeconomic Forecasting."** arXiv:[2606.28670](https://arxiv.org/abs/2606.28670). `2606.28670_carriero_et_al_macrocast_vintage_consistent_tsfm.pdf`
Names and formalizes two leakage modes explicitly relevant to CuBench's own vocabulary: **"temporal contamination"** (model has seen realized future values) and **"revision bias"** (training on fully-revised rather than vintage-specific data). Builds a foundation model trained only on synthetic + real-time-vintage (ALFRED) data to rule out both. Useful primarily as a citation that gives CuBench's own §2.3 "revisions are not modelled" disclosure precise, literature-standard terminology (CuBench's residual limitation — using final-vintage PPI/INDPRO timestamped at first-release date — is exactly what this paper calls "revision bias," distinct from the "temporal contamination" that CuBench's PIT join already fully solves).

---

## Section 5 — Commodity factor models: carry, momentum, basis (industrial metals / copper)

**This topic returned essentially nothing new and directly relevant on arXiv.** Queries tried in this session: `"commodity futures momentum term structure cross-sectional returns factor model"`, `"convenience yield theory of storage metals futures curve backwardation contango"`, `"basis momentum" commodity futures hedging pressure`, and `au:Rossi commodity risk premium futures`. This confirms the feasibility report's own observation that the Bakshi-Gao-Rossi / Angelidis-Sakkas-Tessaromatis / Yang line of carry-and-momentum evidence lives in *Management Science*, *Journal of Financial Economics*, *Journal of Commodity Markets*, and SSRN — not arXiv — and that gap is real, not a search failure. One tangentially useful paper was found and downloaded as the best available arXiv-native substitute:

**Bianchi, R. J., Fan, J. H., Miffre, J., and Zhang, T. (2023). "Exploiting the dynamics of commodity futures curves."** arXiv:[2308.00383](https://arxiv.org/abs/2308.00383). `2308.00383_bianchi_fan_miffre_zhang_commodity_futures_curves.pdf`
Uses a Nelson-Siegel level/slope/curvature decomposition of commodity futures term structure to build investment strategies exploiting short-term continuation of curve-shape changes. Not copper-specific and not carry/momentum in the classical Bakshi-Gao-Rossi sense, but it is the closest arXiv-native paper to "curve-shape-based commodity strategies," and is directly relevant to CuBench's own Block B′ curve-proxy features (§1.5: `b1_rv_term_ratio`, term-structure slope features) — it is evidence that curve-shape dynamics (not full carry) have documented predictive content, supporting Block B′'s inclusion as an ablation-only feature block even without a true contract-month ladder.

**Recommendation:** if the paper's Related Work section needs the actual Bakshi-Gao-Rossi / Angelidis et al. citations (it should — they are foundational), they must be obtained outside the arXiv MCP (e.g., institutional library access to *Management Science* and the SSRN working paper), exactly as the feasibility report already flagged.

---

## Section 6 — Weak-form market efficiency / random-walk testing on copper and base metals

The step-1 set already contains the strongest paper here (Portnaya 2026, `2606.29591`, Section 0 above — sign/magnitude decomposition of return predictability, finds commodities/FX/crypto indistinguishable from random walks). Searches in this session for `"market efficiency" copper OR "industrial metals" price predictability` and `variance ratio test commodity futures return predictability autocorrelation` did not surface additional copper/metals-specific efficiency papers beyond what's already cited. The Bastianin, Rossini & Tonni paper (Section 4 above, `2512.16521`) is the closest new paper with copper-specific predictability evidence, though framed as monthly real-price forecastability rather than a formal efficiency/random-walk test — it is cross-referenced here because its finding ("short-run metal price movements remain difficult to predict, medium-term horizons display substantial forecastability") is directly relevant to motivating CuBench's volatility-over-direction target choice (D1) with a copper-specific citation, complementing Portnaya's cross-asset evidence.

No new dedicated copper/base-metals random-walk or variance-ratio paper was found on arXiv in this pass. This is consistent with the feasibility report's finding that this specific literature (e.g., the "LME variance-ratio study" cited in D1) is not well represented on arXiv.

---

## Section 7 — Quantile regression / probabilistic forecasting for financial returns; quantile crossing and its correction

**Yao, M. and Franklin, M. (2026). "Convolution Smoothed Quantile Regression for XGBoost."** arXiv:[2608.15290](https://arxiv.org/abs/2608.15290). `2608.15290_yao_franklin_convolution_smoothed_qr_xgboost.pdf`
Develops QXGB, a convolution-smoothed quantile loss for XGBoost that **restores Hessian information for tree splitting** (native XGBoost quantile loss lacks a usable Hessian) and reports **near-zero quantile crossing**, especially with multi-output trees, versus XGBoost's native quantile objective. This is directly and immediately actionable for CuBench's `xgboost` T2 spec (§3.4), which currently uses `objective='reg:quantileerror'` "as comparison only, with the crossing caveat" — this paper is a candidate fix that would let XGBoost's quantile results be reported on a similar footing to CatBoost's `MultiQuantile`, rather than as a known-inferior comparison point. Worth flagging to the implementation team as a concrete methodological upgrade path, not just a citation.

**Park, Y., Maddix, D., Aubet, F-X., Kan, K., Gasthaus, J., and Wang, Y. (2021). "Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting."** arXiv:[2111.06581](https://arxiv.org/abs/2111.06581). `2111.06581_park_et_al_quantile_functions_no_crossing.pdf`
Proposes an incremental/spline quantile-function parameterization (I(S)QF) that structurally prevents quantile crossing (by construction, not post-hoc repair) and additionally supports interpolation to untrained quantile levels. Useful as the canonical citation for the general quantile-crossing problem and for contrasting CuBench's chosen **post-hoc isotonic-sorting repair** (§3.4: "quantile crossing is checked explicitly and, where present, repaired by post-hoc isotonic sorting") against structural alternatives — the paper's framing supports CuBench's decision to report the *pre-repair* crossing rate as an honesty metric (this is exactly the failure mode the paper is designed to eliminate by construction, so a high raw crossing rate before repair is a meaningful, literature-recognized signal, not just implementation noise).

---

## Summary counts

- **Step 1 (already identified, now downloaded):** 15 of 15 arXiv IDs successfully downloaded as verified PDFs. 4 additional non-arXiv sources from the same list (Bakshi-Gao-Rossi, Yang, Angelidis et al., Bailey & López de Prado) are confirmed unfetchable via this tool and documented above rather than skipped silently.
- **New papers found and downloaded (Step 2):** 12 total.
  - Section 1 (HAR-RV/GARCH vs. ML/GBM): 3 — `2601.13014`, `2406.08041`, `2405.19849`
  - Section 2 (GBM benchmarks vs. deep learning): 0 downloaded (1 abstract-only reference noted, not saved)
  - Section 3 (PBO/DSR/statistical rigor): 3 — `2608.23808`, `2501.03938`, `2409.12662`
  - Section 4 (point-in-time/publication lag): 3 — `2512.16521`, `2608.09033`, `2606.28670`
  - Section 5 (carry/momentum/basis, industrial metals): 1 — `2308.00383` (partial fit; genuine gap noted)
  - Section 6 (weak-form efficiency, copper/metals): 0 new (cross-references Section 0 and Section 4 finds)
  - Section 7 (quantile regression / crossing): 2 — `2608.15290`, `2111.06581`

## Most important findings for the paper (flagged for supervisor/author attention before drafting)

1. **RQ1 / D2 is contested, not settled**, and the contest hinges on HAR's refit-frequency specification (Audrino & Chassot 2024 vs. Christensen, Siggaard & Veliyev 2026) — see Section 1 flag above. Recommend adding this nuance to the pre-registration's RQ1 discussion before drafting.
2. **The DM/HLN test may not fully solve the overlapping-target (h=5, h=22) inference problem** — Coroneo & Iacone (2024) show DM loses power and can spuriously reject under strong loss-differential autocorrelation, a distinct failure mode from the bias HLN corrects. Recommend reviewing before finalizing `src/cubench/stats_tests.py::dm_test_hln()` (§4.3).
3. **Bastianin, Rossini & Tonni (2025) is a genuinely strong, copper-specific, real-time/PIT-methodology paper** that both directly supports D5 (PPI/INDPRO) with copper-specific (not just cross-market) evidence and gives CuBench's Data section a much closer point-in-time precedent than anything in the original feasibility report's list.
4. **XGBoost's quantile-crossing weakness has a concrete, citable fix** (Yao & Franklin 2026, convolution-smoothed QXGB) that could upgrade the T2 XGBoost comparison from "known caveat" to "candidate improvement," if implementation time allows.
5. Sections 5 and 6 confirm, rather than overturn, the feasibility report's finding that the classical carry/momentum and copper-efficiency literature is concentrated off-arXiv (SSRN, *Management Science*, *JFE*, *Journal of Commodity Markets*) — this is reported as a negative finding with the exact queries attempted, not left as a silent gap.

All 27 PDFs and this index are saved under `D:\copper\literature_review\paper2\`.


---

See also literature_review/shared/ for papers relevant to both paper1 and paper2 (arXiv 2409.08355, 2409.08356, 2607.12248 were downloaded independently by both literature passes and have been deduped into a single shared copy).
