# CuBench Pre-registration — Paper 2

**Title:** A Leakage-Audited, Regime-Aware ML Benchmark for Volatility and Return
Forecasting in Industrial Metal Markets: The Case of Copper

**Authors:** Anmol Jain · **Supervisor:** Dr. Sibarama Panigrahi · NIT Rourkela, CS4095
Capstone-I

**Date:** 2026-08-21

**Git SHA at commit time:** *(to be filled in by the commit that adds this file to
version control — see the Amendment Log note below. As of the drafting of this
document, the repository's HEAD is `e83fe1ab16a68fa0823f8469484b1411996bac10` on branch
`graph-fix-experiment`; the actual commit SHA that first introduces this file to the
repo is the authoritative one and should be quoted verbatim in the manuscript's Methods
section, not the SHA above.)*

**Statement:** This document was drafted and is intended to be committed before any
model is trained on out-of-sample data. `src/cubench/models/` contains no trained model
as of this writing. Per the plan's hard gate (`docs/cubench_implementation_plan.md`
Section 5), this file must be reviewed by the supervisor and committed with a signed
git commit *before* Tier-0 through Tier-5 model training begins (Week S1 onward). It is
written but **intentionally left uncommitted by the automated build pipeline** — the
plan requires supervisor review before the actual commit, so committing this file is
left to the user/orchestrator, not done here.

---

## 1. Header

See title/authors/date/SHA block above. This document governs Paper 2 (CuBench) only;
Paper 1 (VMD-MFGNN) is unaffected and out of scope here.

---

## 2. Research questions

- **RQ1:** Does gradient boosting on engineered features beat Log-HAR-RV on h-day
  copper realized volatility, by QLIKE, under 11-fold expanding walk-forward?
- **RQ2:** Do LightGBM quantile forecasts achieve better calibration (Kupiec coverage,
  pinball) than GARCH/GJR-implied and empirical-quantile baselines?
- **RQ3:** Is copper daily/weekly/monthly return direction predictable above its own
  base rate, PT-significant after Holm correction?
- **RQ4:** Which feature blocks (D, A, C-cross, C-macro) contribute ablation-verified
  incremental value, and does that differ by volatility regime and calendar regime?
- **RQ5:** Does any apparent edge survive transaction costs, DSR, and PBO?

---

## 3. Complete data specification

### 3.1 Block P — Copper price core (mandatory)

| Series | Ticker | Source | Verified range | Fields |
|---|---|---|---|---|
| COMEX copper continuous front | `HG=F` | yfinance | 2010-01-04 → 2025-12-30, 4,023 obs | Open, High, Low, Close (Volume excluded — rollover artifact, not real COMEX volume) |

Continuous/front-contract feed with an undisclosed roll rule; disclosed in Limitations
(§10).

### 3.2 Block C-cross — Cross-market daily (yfinance, Close only)

| Name | Ticker | Coverage | Decision |
|---|---|---|---|
| Aluminum | `ALI=F` | **2014-05-06 → 2025-12-30, 2,895 obs** (differs from the plan's original claim of 2010 coverage — see §10) | Include; NaN before 2014-05-06 |
| Gold | `GC=F` | 2010-01-04 → 2025-12-30, 4,022 obs | Include |
| Silver | `SI=F` | 2010-01-04 → 2025-12-30, 4,022 obs | Include |
| Crude oil | `CL=F` | 2010-01-04 → 2025-12-30, 4,023 obs | Include |
| S&P 500 | `^GSPC` | 2010-01-04 → 2025-12-30, 4,023 obs | Include |
| US 10Y nominal yield | `^TNX` | 2010-01-04 → 2025-12-30, 4,021 obs | Include; differenced in bp, never log-returned |
| E-mini copper | `QC=F` | 2010-01-04 → 2025-12-30, 4,024 obs, 3.55% gap vs a pure business-day calendar | Included as B′'s `b4_hg_qc_spread` basis — a documented judgment call (see `docs/cubench_data_availability.md` Discrepancy 2): benchmarked against `HG=F`'s own near-identical gap rate (3.57%) rather than the FX-calibrated 2% threshold, since both share COMEX's holiday calendar. |

### 3.3 Block C-macro — Macro/financial (FRED via curl, + yfinance)

| Name | ID | Source | Frequency | Verified range | Publication lag | Decision |
|---|---|---|---|---|---|---|
| US Dollar Index | `DX-Y.NYB` | yfinance | daily | 2010→2026, 4,024 obs | none | Include |
| VIX | `^VIX` | yfinance | daily | 2010→2026, 4,023 obs | none | Include — most regime-stable relationship in the study |
| Chilean peso (USD/CLP) | `CLP=X` | yfinance | daily | 2010-01-01→2025-12-30, 4,164 obs, 0.22% gap | none | Include (passed the pre-decided ≥3,800-obs/<2%-gap rule) |
| 10Y TIPS real yield | `DFII10` | FRED/curl | daily | first obs 2009-01-02 | none | Include (Granger-causal at all 3 lags despite r≈0) |
| Baa credit spread | `BAA10Y` | FRED/curl | daily | first obs 2009-01-02 | none | Include — replaces `BAMLH0A0HYM2` (D4) |
| ICE BofA HY OAS | `BAMLH0A0HYM2` | FRED/curl | daily | **2023-08-21→2025-12-31, 620 obs only** | none | **Excluded from headline**; 2023–2025 robustness-appendix only |
| PPI, all commodities | `PPIACO` | FRED/curl | monthly | first obs 2009-01-01 | **+30 calendar days** | Include (D5); PIT-joined |
| Industrial production | `INDPRO` | FRED/curl | monthly | first obs 2009-01-01 | **+22 calendar days** | Include; PIT-joined |
| China business confidence (PMI proxy) | `BSCICP03CNM665S` | FRED/curl | monthly | 2009→2025, 181 obs | **+45 calendar days** | **Ablation-only (D7)** — negative-result finding (r=0.010, p=0.51) |

### 3.4 Block B — Carry/curve: TIER 3, confirmed by audit

**Decision (D3), confirmed by the Week A1 coverage audit:** true futures term-structure
carry is omitted from the headline model. All 192 tickers of the form
`HG{F,G,H,J,K,M,N,Q,U,V,X,Z}{10..25}.CMX` were probed; **1 of 192 returned usable data**
(`HGZ25.CMX`, 2020-09-29→2025-12-29 only). The pre-committed promotion rule (≥2
simultaneously-live contract months on ≥60% of 2010–2025 trading days) fails trivially.
Full results: `results/cubench/data_audit/comex_contract_month_coverage.csv`; narrative:
`docs/cubench_data_availability.md`.

**Block B′ (curve proxy, ablation-only, 4 features, never called carry or basis):**
- `b1_rv_term_ratio` = RV₅ / RV₂₂ (Garman-Klass realized-variance term ratio)
- `b2_cu_al_relvalue` = 60-day z-score of log(HG=F / ALI=F)
- `b3_cu_al_mom_spread` = 22-day copper momentum − 22-day aluminum momentum
- `b4_hg_qc_spread` = log(HG=F close) − log(QC=F close)

### 3.5 Resolved decision — aluminum coverage and the 2010 panel start

Aluminum (`ALI=F`) data begins 2014-05-06, not 2010. **Decision: the panel starts
2010-01-04 regardless** (preserves regime diversity across the full 2010–2025 window).
Aluminum-derived features (`c_aluminum_r1`, `c_aluminum_r5`, `b2_cu_al_relvalue`,
`b3_cu_al_mom_spread`) are `NaN` for all rows before 2014-05-06 — never forward-filled,
backfilled, zero-filled, or used to drop rows. Tree-based models (LightGBM/XGBoost/
CatBoost/RandomForest) route NaN natively; classical baselines (HAR/GARCH/ARIMA) do not
use Block C or B′ features at all, so this is a non-issue for them.

---

## 4. Complete frozen feature list

**Frozen 2026-08-21** (ahead of the plan's 2026-08-30 deadline). No feature may be
added after this date without a dated, numbered Amendment Log entry (§11) disclosed in
the paper and shown separately from the headline results table.

**Total: 70 headline features + 1 ablation-only feature (`c_china_conf`).**

> **Disclosed discrepancy:** the implementation plan's Section 2.4 header states Block D
> has "26 features," but its own feature table (row by row) sums to **28**. This
> document and the built feature matrix follow the table literally (28 Block-D
> features), since the table is the operative, itemized specification and "26" is a
> summary miscount elsewhere in the same document. Total feature count is therefore 70,
> not the "~68" used as an approximate figure throughout the plan — the plan itself
> repeatedly qualifies that number as approximate. This does not change any research
> question, model, or protocol; it is recorded here as the standing practice of this
> project (catch and disclose real discrepancies, never silently resolve them).

### Block A — Trend / momentum (16 features)
`a_mom_5, a_mom_10, a_mom_22, a_mom_63, a_mom_126, a_mom_252` — `ln(c_t/c_{t-k})`
`a_ewma_x_8_32, a_ewma_x_16_64, a_ewma_x_32_128` — `EWMA(c,f)/EWMA(c,s) - 1`
`a_zscore_22, a_zscore_63` — `(c_t - mean(c,k)) / std(c,k)`
`a_rsi_14` — Wilder RSI, 14-day
`a_dist_hi_252, a_dist_lo_252` — `ln(c_t/max(h,252))`, `ln(c_t/min(l,252))`
`a_r_lag1, a_r_lag2` — `r_t`, `r_{t-1}`

### Block B′ — Curve proxy (4 features, ablation-only, never headline)
`b1_rv_term_ratio, b2_cu_al_relvalue, b3_cu_al_mom_spread, b4_hg_qc_spread`

### Block C — Macro / cross-market (22 headline features)
Daily returns (14): `c_dxy_r1, c_dxy_r5, c_vix_r1, c_vix_r5, c_sp500_r1, c_sp500_r5,
c_gold_r1, c_gold_r5, c_silver_r1, c_silver_r5, c_aluminum_r1, c_aluminum_r5,
c_crude_r1, c_crude_r5`
Yield/spread differences in bp (6): `c_tnx_d1, c_tnx_d22, c_dfii10_d1, c_dfii10_d22,
c_baa10y_d1, c_baa10y_d22`
Point-in-time-joined monthly (2): `c_ppi_yoy, c_indpro_yoy`
**Ablation-only (1, not in the 22):** `c_china_conf`

### Block D — Volatility / regime (28 features)
`d_logrv_1, d_logrv_5, d_logrv_22, d_logrv_66` (HAR-RV components, Garman-Klass)
`d_logrv_park_1, d_logrv_park_5, d_logrv_park_22` (Parkinson robustness series)
`d_logrv_sq_5, d_logrv_sq_22` (squared-return series)
`d_rv_ratio_5_22, d_rv_ratio_22_66` (vol term-structure slope)
`d_volofvol_22, d_downside_semivar_22, d_skew_63, d_kurt_63, d_jump_22`
`d_garch_sigma, d_gjr_sigma, d_garch_resid_z` (expanding-refit GARCH/GJR-GARCH, `src/cubench/garch.py`)
`d_vix_level, d_vix_ratio_5_22`
`d_regime_lo, d_regime_mid, d_regime_hi` (expanding-quantile vol terciles)
`d_dow_1, d_dow_2, d_dow_3, d_dow_4` (Tue–Fri, Monday reference)

All rolling/expanding computations are strictly causal (no `center=True`, no
whole-series `.fit()`, no global standardization); mechanically enforced by
`tests/test_leakage.py::test_causal_perturbation` and `test_garch_causality`.

---

## 5. Complete model roster and hyperparameters (reference)

Per plan Section 3, verbatim: **Tier 0** — 4 statistical nulls (`null_persist`,
`null_rollmean`, `null_zero`, `null_majority`). **Tier 1** — 4 econometric models
(`har_rv`, `har_rv_q`, `garch11`, `gjr_garch`, `arima`). **Tier 2** — `elasticnet`
(Pipeline-wrapped scaler, fit on training fold only). **Tier 3** — 4 gradient-boosted
tree models (`lgbm` primary, `xgboost`, `catboost`, `randomforest`), fixed
hyperparameters (not tuned per fold), 5 seeds (42–46), single Optuna HPO study (50
trials TPE) run once on 2010–2014 only and frozen for all 11 OOS years. **Tier 4** — 2
deep-learning comparison points (`lstm`, `transformer`), architecture ported verbatim
from `src/models/baselines.py`, retrained from scratch on CuBench's feature matrix and
protocol, 3 seeds each. **Tier 5** — `vmd_mfgnn`, imported as a results row from Paper 1
(not retrained), explicitly footnoted as not directly comparable.

---

## 6. Complete evaluation protocol (reference)

**Fold structure (11 folds, expanding window, annual refit):** initial training block
2010-01-04 → 2014-12-31; OOS years 2015–2025. Embargo `E = h + 5` trading days (6/10/27
for h=1/5/22), purged from the END of the training window only (no gap inside the OOS
block). Fold boundaries are generated by `src/cubench/folds.py::make_folds()` and
serialized to `results/cubench/folds.json` — see that file for the exact 11 date ranges
actually produced against copper's real trading calendar.

**Metrics:** T1 — RMSE/MAE (log scale), R²_OOS vs `null_rollmean`, QLIKE (variance
scale, primary ranking metric), MZ regression. T2 — pinball loss (primary), CRPS,
Kupiec coverage, quantile-crossing rate. T3 — raw DA + base rate + excess DA (bolded
number), PT statistic, McNemar vs `null_majority`, Brier score, calibration curve.

**Statistical tests:** Diebold-Mariano with Harvey-Leybourne-Newbold small-sample
correction (`src/cubench/stats_tests.py::dm_test_hln`, Student-t reference
distribution, not normal); Pesaran-Timmermann (hand-implemented, degenerate-constant-
forecast guard returns NaN + flag); Holm-Bonferroni within three pre-declared families
(T1: 11 models × 3 horizons = 33 tests; T2: 33; T3: 33), ablation/regime families
separately corrected and labelled exploratory.

**DSR/PBO:** `pypbo` (pinned commit, vendored per the packaging workaround documented
in `docs/cubench_data_availability.md` Discrepancy 3), CSCV S=16 partitions, hand-
derived DSR cross-check to 1e-6.

**Cost grid:** `{0,1,2,5,10,20,50}` bps round-trip; headline case 2 bps; realistic
gross Sharpe 0.4–0.8 pre-committed, net 0.2–0.5; gross Sharpe >1.5 triggers a mandatory
leakage-audit re-run before it may be reported.

**Regime segmentation:** volatility tercile axis (`d_regime_lo/mid/hi`) + 5 pre-declared
calendar regimes (R1 2015–2016 post-supercycle bust; R2 2017–2019 synchronized
growth/trade war; R3 2020 COVID shock; R4 2021–2022 stimulus rally/2022 shock; R5
2023–2025 energy-transition/destocking).

**Build verification (real run, 2026-08-21):** the feature matrix and folds referenced
above were actually built end-to-end from `data/cubench/raw/` and are on disk at
`data/cubench/features.parquet` (shape `(4023, 81)`: 71 features + 9 target columns +
`date`, 2010-01-04 → 2025-12-30) and `results/cubench/folds.json` (11 folds, confirmed
non-overlapping, embargo boundaries verified disjoint from OOS by
`tests/test_leakage.py::test_embargo_disjointness`). GARCH(1,1)-normal and GJR-
GARCH(1,1,1)-t were each expanding-window fit with monthly refit: **189 refits each, 0
convergence failures**, burn-in NaN through 2010-03-31 (61 rows) before the first fit
has enough history (`MIN_HISTORY=60`). All 7 leakage-safety gates in
`tests/test_leakage.py` pass (10 collected test cases including the 4 parametrized FRED
coverage checks).

---

## 7. Pre-committed success/failure criteria (stated verbatim, non-reinterpretable)

- **RQ1 success:** LightGBM QLIKE improvement over `har_rv` ≥ 2% at ≥2 of 3 horizons,
  DM-significant after Holm. **RQ1 failure:** HAR wins or ties → *"HAR-RV remains
  unbeaten on copper realized volatility by gradient boosting with 68 engineered
  features"* is reported as the headline finding, and this is an equally publishable
  result corroborating Wang & Lu 2024 and Brini 2026.
- **RQ3 expected outcome:** excess DA in [−1pp, +3pp], 0–2 of 9 cells PT-significant
  after Holm. **Explicit pre-commitment:** directional results are reported exactly as
  they land. If DA is 50%, that is the finding, corroborates the LME variance-ratio
  study and Portnaya 2026, and is reported without softening. **If DA exceeds 60% at
  any horizon, that cell is treated as a suspected leakage bug and the full §2.5
  leakage suite is re-run before the number may appear in any draft.**
- **RQ5 pre-commitment:** if PBO > 0.5, the paper states the backtest is likely
  overfit regardless of Sharpe.

---

## 8. Anti-p-hacking commitments

- No metric added after the freeze may be used to select the headline result.
- No horizon, target, model, or regime may be dropped from any results table after
  seeing results.
- The full grid is reported; no cherry-picking of "best" cells.
- Bolding rule (pre-declared): a cell may only be bolded if it is best *and* passes the
  base-rate diagnostic with verdict `ok` *and* is Holm-significant.
- **Named prohibition:** under deadline pressure in October/November, the team will
  **not** revive the Gumbel-softmax graph experiment, will **not** switch the primary
  target away from volatility, and will **not** re-tune hyperparameters on OOS folds.

---

## 9. Novelty statement

CuBench is a rigorous tabular-ML **benchmark and ablation study**, not a novel
architecture. The base-rate diagnostic applied throughout (§4.7 of the implementation
plan) is an explicit **application and extension** of **Cheung 2026** (arXiv:2607.12248),
never claimed as an invention — Cheung formalizes the base-rate-driven-accuracy
phenomenon that this project independently encountered in its own prior work (Paper 1).
Two genuine extensions are claimed: (a) the pre-hoc dispersion-ratio + sign-
concentration screen (motivated by Paper 1's Transformer_h5 case: dispersion ratio 0.42
yet 100% same-sign predictions, which dispersion alone would not have caught), and (b)
applying the diagnostic across a full multi-model × multi-horizon × multi-target
benchmark grid rather than to a single model.

**Prior art acknowledged up front:** Wang & Lu 2024 (arXiv:2409.08356), Wang & Li 2024
(arXiv:2409.08355), Brini 2026 (arXiv:2607.05291), Portnaya 2026 (arXiv:2606.29591),
Cheung 2026 (arXiv:2607.12248), Fang & Slepaczuk 2026, Dudek et al. 2025.

---

## 10. Known limitations, declared in advance

- No true carry/curve block (Block B): the free-data ecosystem does not support
  commodity term-structure reconstruction over 2010–2025 (§3.4). Block B′ is an
  explicitly-labelled, non-equivalent proxy.
- Final-vintage macro revisions: PIT joins correct release *timing* but not data
  *revision* (FRED serves latest-vintage values). Bounded empirically by ablations
  A4-vs-A3 and A7.
- Continuous-contract roll rule undisclosed by Yahoo for `HG=F`.
- Range-based (Garman-Klass) RV proxy used in place of unavailable free intraday RV;
  mitigated by a pre-registered Parkinson-based robustness re-run, but absolute R²
  levels (not relative rankings) are affected.
- Single asset (copper); findings should not be generalized to other industrial
  metals without re-verification.
- Free-data-only constraint throughout; no licensed/paid data source used anywhere.
- **Newly disclosed in this document:** the aluminum coverage discrepancy (`ALI=F`
  starts 2014-05-06, not 2010 as an earlier draft of the plan assumed) — resolved by
  keeping the 2010 panel start and leaving aluminum-derived features NaN pre-2014,
  never imputed (§3.5). The Block-D feature-count discrepancy (plan header "26" vs.
  itemized table "28") — resolved by following the itemized table (§4).

---

## 11. Amendment Log

*(Empty at commit time. Any post-freeze change to the feature list, model roster,
protocol, or success criteria must be appended here as a dated, numbered entry with a
stated reason, and disclosed in the manuscript.)*

| # | Date | Change | Reason |
|---|---|---|---|
| — | — | — | — |
