# CuBench — Complete Technical Implementation Plan

**Project:** A Leakage-Audited, Regime-Aware ML Benchmark for Volatility and Return Forecasting in Industrial Metal Markets: The Case of Copper
**Author:** Anmol Jain · **Supervisor:** Dr. Sibarama Panigrahi · NIT Rourkela, CS4095 Capstone-I
**Plan date:** 2026-08-19 · **Execution window:** 2026-08-19 → 2026-11-30
**Repo:** `D:\copper` · **Status of this document:** buildable specification, not a menu. Every open question below has been decided.

---

## 0. Governing decisions (read first)

These are the decisions that everything else in this document depends on. They are made, not proposed.

| # | Decision | Grounding |
|---|---|---|
| D1 | **Primary target is h-day forward realized volatility (log scale).** Direction is a reported secondary finding, never the headline. | LME variance-ratio study fails to reject random walk for copper specifically; Portnaya 2026 (arXiv:2606.29591) finds commodities indistinguishable from random walks in sign; 918-experiment study mean DA = 50.08%. |
| D2 | **Log-HAR-RV is the model to beat, not a strawman.** Wang & Lu 2024 (arXiv:2409.08356) report HAR lowest QLIKE on COMEX copper daily RV; Brini 2026 (arXiv:2607.05291) finds only one of nine TSFMs narrowly beats Log-HAR. If CuBench's LightGBM does not beat Log-HAR, that is the paper's finding and it gets reported. | Feasibility report §2. |
| D3 | **Block B (true carry/curve) is TIER 3 — omitted from the headline model.** Replaced by an explicitly-labelled, non-equivalent `B′` curve-*proxy* block that is ablation-only and never described as basis/carry. A one-off exhaustive contract-ticker coverage audit is run and published as a data-availability negative finding. | `HGZ25.CMX` starts 2020-09-29; `HGZ15.CMX` 404s; CME continuous requires licensed DataMine. Feasibility report §3. |
| D4 | **`BAMLH0A0HYM2` is dropped as the credit feature.** Replaced by FRED **`BAA10Y`** (Moody's Baa − 10Y Treasury, daily, 1986→present, no ICE licensing restriction). HY OAS retained only as a 2023-08→2025 robustness check. | FRED serves only 3 years of BAMLH0A0HYM2 since April 2026 — confirmed *independently twice* (Codex report + this project's own notebook, which got 754 rows from 2023-08-21). The truncation is a source-licensing restriction, so an API key will **not** recover it; changing endpoint is not a fix. |
| D5 | **PPI is added** (`PPIACO`), plus `INDPRO`. Both monthly, both publication-lagged. | Wang & Li 2024 (arXiv:2409.08355) name PPI the most efficient macro predictor of long-term copper volatility; industrial production also significant. |
| D6 | **Exchange inventory (LME/COMEX/SHFE) is EXCLUDED from the headline feature set**, with one timeboxed Week-2 spike and a pre-committed limitation write-up if it fails. | No free, reproducible, full-2010-2025 daily series exists (LME/CME bulk history is licensed). Copper-fundamentals track independently documented that visible stocks are contaminated by Chinese bonded "invisible" stock (0.5–1Mt) and SHFE-LME arbitrage flows, so even if obtained the signal is impaired. Correlation study found *nothing* leads copper at daily frequency, so the prior that inventory would is unsupported in this project's own data. |
| D7 | **China PMI is excluded from the headline model**, retained as a single ablation-only feature using the OECD proxy, explicitly labelled "not the series the literature cites." | Genuine Caixin/NBS confirmed not free by two independent investigations. Proxy r = 0.010, p = 0.51, Granger p > 0.5 at all lags. |
| D8 | **All Block C macro features are engineered contemporaneously (same-day) for daily-frequency series**, not lagged, because 9 of 11 candidates peak at CCF lag 0. Monthly series are joined **point-in-time by real release date**, which is a different mechanism from predictive lagging and is non-negotiable. | Correlation study Finding 1 + Finding-note on forward-fill look-ahead. |
| D9 | **No cointegration / VECM / levels-based modelling anywhere.** All features and targets are returns-, difference-, or volatility-based. | Every Engle-Granger p > 0.05 (min 0.059, most > 0.14, several > 0.9) across all 11 candidates. |
| D10 | **The base-rate/constant-forecast diagnostic is presented as an *application and extension* of Cheung 2026 (arXiv:2607.12248), never as a novel invention.** Wording is pre-committed in the pre-registration. | Feasibility report novelty correction. The earlier Opus "name it, it's your reputation move" framing overclaimed. |
| D11 | **Venue: arXiv + SSRN preprint in November 2026; formal submission post-capstone to Journal of Forecasting (primary) or Finance Research Letters (if compressible), ICAIF 2027 as conference target.** ICAIF 2026 is unreachable (deadline Aug 9, 2026, passed). Manuscript is written in a venue-neutral single-column preprint style, *not* ACM `sigconf`, avoiding the template-port cost entirely. | Feasibility report §4. |
| D12 | **Zero GPU. Entire grid runs on the laptop CPU.** No Colab, no Drive sync, no checkpoint-resume machinery. This eliminates the `sync_to_drive` staleness bug (todo.txt item 5) from the critical path entirely — it is not fixed, it is *not used*. | LightGBM on ~4,000 × ~70 is sub-second; full grid is ~30–90 min. |

---

## 1. Final Data Source Specification

### 1.1 Environment decision

Build on **`D:\copper\.venv_corr`** (Python 3.12, already has numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, statsmodels 0.14.6, scikit-learn 1.9.0, yfinance 1.6.0, jupyter, matplotlib, seaborn — all verified working, including a working yfinance path and a working curl-based FRED path). Do **not** create a fresh venv.

Additional installs required into `.venv_corr` (add to a new `requirements_cubench.txt`; do **not** touch the existing `requirements.txt`, which is Paper 1's frozen environment):

```
lightgbm>=4.5
xgboost>=2.1
catboost>=1.2
arch>=7.0            # GARCH / GJR-GARCH — statsmodels has no equivalent
shap>=0.46
pypbo                # DSR + PBO; install from GitHub (esvhd/pypbo), pin the commit SHA
dieboldmariano>=1.0  # reference cross-check ONLY, not the primary implementation
torch>=2.0           # CPU wheel only, for the reused LSTM/Transformer
tabulate
```

**Verified environment hazard, carry forward:** `requests` reliably times out against `fred.stlouisfed.org` on this machine (TLS/schannel), while `curl` succeeds against the identical URL. **All FRED access in CuBench must shell out to `curl` via `subprocess`**, exactly as `notebooks/correlation_feasibility_analysis.ipynb` does. Do not use `requests`, `pandas_datareader`, or `fredapi` (which wraps `requests`).

**Verified library gap:** `statsmodels 0.14.6` does **not** expose `statsmodels.tsa.stattools.diebold_mariano_test` (checked directly on 2026-08-19 — attribute absent, no near-match in `dir()`). The devel-branch function referenced by the feasibility report is not available in any released version. See §4.4 for the resolution.

### 1.2 Block P — Copper price core (mandatory)

| Series | Ticker | Source | Verified range | Fields | Notes / fallback |
|---|---|---|---|---|---|
| COMEX copper continuous front | `HG=F` | yfinance | **2010-01-04 → 2025-12-30, 4029 obs** (direct Yahoo chart-endpoint test, feasibility report) | Open, High, Low, Close, Adj Close, Volume | Already cached at `data/raw_correlation_check/yf_copper.csv` **with full OHLC** — verified, this is the enabler for range-based RV estimators. **Volume is unusable** (cached values of 404, 242 contracts/day are front-month rollover artifacts, not real COMEX volume) — Volume is **excluded from all features**. Fallback: Investing.com manual CSV; Tier-2 fallback CME `HGCP1` if licensed access materialises (it will not). |

This is a **continuous/front-contract feed with an undisclosed roll rule** and this must be stated in the Data section and the Limitations section. It is why Block B cannot be reconstructed from it.

### 1.3 Block C-cross — Cross-market daily series (yfinance, Close only)

All already cached under `data/raw_correlation_check/`, all verified 2010→2026, all ~4,000–4,200 obs.

| Name | Ticker | Verified r vs copper returns | Decision |
|---|---|---|---|
| Aluminum | `ALI=F` | +0.41 | Include |
| Gold | `GC=F` | +0.32 | Include (expectation downgraded — "great influence" claim not reproduced) |
| Silver | `SI=F` | +0.44 | Include (highest r; most sign-stable of the metals, rolling min −0.02) |
| Crude oil | `CL=F` | +0.26 | Include |
| S&P 500 | `^GSPC` | +0.30 | Include (Granger-significant at all 3 lags: 1.4e-5 / 6.1e-6 / 1.1e-4) |
| US 10Y nominal yield | `^TNX` | +0.12 | Include; **differenced in bp, never log-returned** |

Fallback for any yfinance failure: the cached CSVs in `data/raw_correlation_check/` are the frozen fallback; a `data/cubench/raw/` snapshot is written on first successful pull and thereafter is authoritative (see §7).

### 1.4 Block C-macro — Macro/financial (FRED via curl, + yfinance)

| Name | ID | Source | Frequency | Verified/expected range | Publication lag applied | Decision & justification |
|---|---|---|---|---|---|---|
| US Dollar Index | `DX-Y.NYB` | yfinance | daily | 2010→2026 verified | none (same-day) | Include. r = −0.30, but **252d rolling corr swings −0.61 → +0.07** and cointegration p = 0.97. Include with the instability documented; do **not** market as "the reliable macro feature." |
| VIX | `^VIX` | yfinance | daily | 2010→2026 verified | none | Include. **The single most regime-stable relationship in the entire study** — rolling corr never crosses zero across 16 years (−0.48 → −0.02). Cross-check against FRED `VIXCLS` in validation; reconcile non-trading-day misalignment by inner-joining to copper's calendar. |
| 10Y TIPS real yield | `DFII10` | FRED/curl | daily | 2003→present; verified pullable | none | Include **despite** r = −0.005 (n.s.). Granger-causal at all 3 lags (p = .0027/.0022/.0018) — the one candidate where a nonlinear learner is expected to find what Pearson cannot. This is an explicit, pre-registered hypothesis. |
| Baa credit spread | `BAA10Y` | FRED/curl | daily | 1986→present | none | **Include as the credit feature, replacing HY OAS (D4).** Daily, free, full history, no ICE licensing restriction. |
| ICE BofA HY OAS | `BAMLH0A0HYM2` | FRED/curl | daily | **2023-08-21→present only (754 obs)** — confirmed twice | none | **Excluded from headline.** Used only in a 2023–2025 sub-sample robustness appendix, to show the BAA10Y substitution does not change conclusions on the window where both exist. |
| PPI, all commodities | `PPIACO` | FRED/curl | monthly | 1913→present | **+30 calendar days** from reference-month end | **Include (D5).** Direct Wang & Li 2024 support. |
| Industrial production | `INDPRO` | FRED/curl | monthly | 1919→present | **+22 calendar days** | Include. Wang & Li 2024 support; the "Dr. Copper" channel's only econometrically-supported leg (Granger causality to IP). |
| China business confidence (PMI proxy) | `BSCICP03CNM665S` | FRED/curl | monthly | 2000→present, cached | **+45 calendar days** (OECD compilation + revision) | **ABLATION-ONLY (D7).** r = 0.010 (p = 0.51), Granger p > 0.5 at every lag, rolling corr −0.045 in the very 2021–23 window the econometrics track flagged. Reported as an explicit negative result: "the only freely-obtainable China-demand proxy adds nothing." |
| Chilean peso | `CLP=X` | yfinance | daily | **UNVERIFIED — Week-1 spike** | none | Conditional include. Pre-decided rule: if `CLP=X` returns ≥3,800 daily obs over 2010-01-01→2025-12-31 with <2% gaps, include; otherwise substitute FRED `DTWEXEMEGS` (broad EM FX, daily, 2006→); if that also fails, **drop the producer-currency channel entirely** and note it. No third attempt, no manual scraping. |

**FRED URL pattern (verified working via curl):**
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>&cosd=2009-01-01&coed=2025-12-31`
Note the correlation notebook's finding that `cosd=` is silently ignored for restricted series — this is not a bug to work around, it is a licensing truncation. Every FRED pull must **assert** the returned first-observation date against an expected value and hard-fail if it regressed (see §2.5 test `test_fred_coverage_assertions`).

### 1.5 Block B — Carry/curve: TIER 3, with a published audit

**Decision (D3): omitted from the headline model.** Justification, stated plainly in the paper: a genuine front-vs-deferred basis requires a contract-month ladder that is not obtainable free for 2010–2025 (yfinance `HGZ25.CMX` begins 2020-09-29; `HGZ15.CMX` returns 404; CME's official `HGCP1`/`HGCPA` continuous series, start 04-Jan-2010, require DataMine/ILA licensing). Bakshi-Gao-Rossi carry evidence is *cross-sectional across commodities*; applying it to a single asset was already a weaker claim (feasibility report §2, "Disconfirming evidence"). Substituting continuous `HG=F` for a term-structure feature would be a silent misrepresentation and is forbidden.

**Mandatory deliverable — the coverage audit (Week 1, ~2 hours):**
Programmatically probe all 192 tickers `HG{F,G,H,J,K,M,N,Q,U,V,X,Z}{10..25}.CMX` via the Yahoo chart endpoint, recording per ticker: HTTP status, first obs date, last obs date, obs count. Write the full table to `results/cubench/data_audit/comex_contract_month_coverage.csv` and a one-paragraph summary to `docs/cubench_data_availability.md`. **This is a publishable artifact**: a reproducible, dated demonstration that the free-data ecosystem cannot support commodity term-structure research, which is exactly the kind of concrete rigor finding this paper is built to supply.

**Promotion rule (pre-committed, single evaluation, Week 1):** promote Block B to Tier 1 only if the audit finds ≥2 simultaneously-live contract months on ≥60% of trading days across the *full* 2010–2025 window. Based on the two probes already run, this will not trigger. Do not re-litigate.

**Block B′ (curve *proxy*, ablation-only, 4 features).** Never called carry, never in the headline model, always footnoted "not a true basis":
- `b1_rv_term_ratio` = RV₅ / RV₂₂ (realized-vol term structure — the closest free analogue of curve shape, mechanically related to it via theory-of-storage)
- `b2_cu_al_relvalue` = 60-day z-score of log(HG=F / ALI=F)
- `b3_cu_al_mom_spread` = 22-day copper momentum − 22-day aluminum momentum
- `b4_hg_qc_spread` = log(HG=F close) − log(QC=F close) if `QC=F` (E-mini copper) passes a Week-1 coverage check; else this feature is dropped and B′ has 3 members.

### 1.6 Excluded, with reasons on record

Baltic Dry Index (confounded by iron ore/coal/grain — fundamentals track), Google Trends (no copper-specific evidence), GPR index (2025 study found no significant GPR effect on copper specifically), China credit impulse `CRDQCNAPABIS` (quarterly, self-constructed, unverifiable), COT positioning (3-day reporting lag, documented as lagging/crowding not predictive), satellite stockpile data (paid), news/NLP sentiment (the one alternative-data family with real OOS evidence, but out of scope for a 3.5-month capstone — named explicitly as future work), TC/RCs (months-long transmission lag, no free daily series), exchange inventories (D6), Caixin/NBS PMI (D7), any time-series foundation model (Chronos/TimesFM/Moirai — evidence is conclusive that fine-tuning does not close the gap to boosting, and 15 years of one series cannot fine-tune anything).

**Divergence from `docs/archive/vmd-mfgnn-protocol-SKILL.md` (stale, Paper-1-only):** that document locks scope to 8 yfinance tickers and states "FRED and BDI are explicitly out of scope." CuBench **deliberately breaks that lock**: it adds 6 FRED series (`DFII10`, `BAA10Y`, `PPIACO`, `INDPRO`, `BSCICP03CNM665S`, `VIXCLS`-as-crosscheck), adds silver `SI=F`, and adds full OHLC for copper (Paper 1 was close-only). BDI remains excluded, now for a stronger reason than before. Add a one-line header to that archive file's status in `docs/cubench_data_availability.md` noting it governs Paper 1 only. `configs/default.yaml` is Paper 1's config and must **not** be edited; CuBench gets its own config (§7).

---

## 2. Complete Feature Engineering Specification

Target design: **~68 features across 5 blocks.** All computed on a single daily DataFrame indexed by copper's trading calendar, all strictly causal (feature at index *t* uses only rows ≤ *t*).

Notation: `c_t` = copper close, `o_t`/`h_t`/`l_t` = open/high/low, `r_t = ln(c_t / c_{t-1})`.

### 2.1 Block A — Trend / momentum (16 features)

| Feature | Formula |
|---|---|
| `a_mom_{5,10,22,63,126,252}` | `ln(c_t / c_{t-k})` for k ∈ {5,10,22,63,126,252} |
| `a_ewma_x_{f}_{s}` | `EWMA(c, span=f)/EWMA(c, span=s) − 1` for (f,s) ∈ {(8,32), (16,64), (32,128)} — the three AQR/CTA-standard trend speeds |
| `a_zscore_{22,63}` | `(c_t − mean(c, k)) / std(c, k)`, k ∈ {22,63} |
| `a_rsi_14` | Wilder RSI, 14-day, `com=13` EWM formulation |
| `a_dist_hi_252`, `a_dist_lo_252` | `ln(c_t / max(h, 252))`, `ln(c_t / min(l, 252))` |
| `a_r_lag1`, `a_r_lag2` | `r_{t}`, `r_{t−1}` |

Leakage handling: every rolling window is `.rolling(k).<agg>()` with the default right-closed, label-right convention, then a **global `shift(0)`** — i.e. the window ending at *t* includes *t*, which is legitimate because the target begins at *t+1*. This is asserted by the leakage unit test (§2.5), not assumed.

### 2.2 Block B′ — Curve proxy (3–4 features, ablation-only)

As specified in §1.5.

### 2.3 Block C — Macro / cross-market (22 features)

**Daily series (same-day, per D8).** For each of `dxy, vix, sp500, gold, silver, aluminum, crude`:
- `c_<x>_r1` = 1-day log return (VIX uses log return; it is a price-type series)
- `c_<x>_r5` = 5-day log return
→ 14 features.

For yield/spread series (`tnx`, `dfii10`, `baa10y`) — **simple differences in basis points, never log returns** (DFII10 goes negative in 2020–21; log returns are undefined there — this bit the correlation notebook and is on record):
- `c_<x>_d1` = `100 * (x_t − x_{t−1})`
- `c_<x>_d22` = `100 * (x_t − x_{t−22})`
→ 6 features.

**Monthly series (point-in-time joined — the single most important leakage control in this project).**
- `c_ppi_yoy` = YoY % change of `PPIACO`
- `c_indpro_yoy` = YoY % change of `INDPRO`
→ 2 features.

**Point-in-time join procedure (mandatory, `src/cubench/pit.py`):**
1. Each monthly observation has a `reference_date` (FRED's period start, e.g. `2015-03-01`).
2. Compute `available_from = reference_month_end + PUBLICATION_LAG[series]`, where `PUBLICATION_LAG = {"PPIACO": 30 days, "INDPRO": 22 days, "BSCICP03CNM665S": 45 days}`. These are **deliberately conservative over-estimates** of the true BLS/Fed/OECD release calendars; erring long can only weaken a result, never manufacture one.
3. Merge onto the daily calendar with `pandas.merge_asof(daily, monthly, left_on='date', right_on='available_from', direction='backward', allow_exact_matches=True)`.
4. **Never** use `reindex().ffill()` on the reference date. The correlation notebook did exactly this and flagged it in its own methodology notes as an unfixed look-ahead risk; CuBench fixes it.
5. **Revisions are not modelled.** FRED serves the latest vintage, not the original print. ALFRED vintage data is out of scope for this timeline. This is disclosed as a named residual limitation: *"our point-in-time join corrects release timing but not data revision; PPI/INDPRO features therefore use final-vintage values timestamped at first-release date. We regard this as a conservative partial correction and quantify its ceiling by re-running the headline model with monthly macro removed entirely (Ablation A3)."* That ablation is what converts a limitation into a measured bound.

**Ablation-only:** `c_china_conf` (`BSCICP03CNM665S`, +45d lag, YoY change).

### 2.4 Block D — Volatility / regime (26 features)

**Daily variance proxies (range-based, using the verified OHLC).**
- Parkinson: `p_t = (1/(4 ln2)) * ln(h_t/l_t)^2`
- Garman-Klass: `gk_t = 0.5 * ln(h_t/l_t)^2 − (2 ln2 − 1) * ln(c_t/o_t)^2`
- Squared return: `sq_t = r_t^2`

**Primary realized-variance series:** `RV_t = gk_t`, floored at `1e-10` before any log. Garman-Klass is chosen over Parkinson (it uses open and close, ~7× more efficient than squared returns) and over squared returns (which are an extremely noisy 1-day proxy). Rationale is stated in the paper; a Parkinson-based robustness re-run of the full T1 grid is a pre-registered secondary analysis.

**Features:**
| Feature | Formula |
|---|---|
| `d_logrv_{1,5,22,66}` | `ln( mean(RV, k) )` for k ∈ {1,5,22,66} — **these four are the HAR-RV components** (d/w/m + a quarterly extension) |
| `d_logrv_park_{1,5,22}` | same on Parkinson |
| `d_logrv_sq_{5,22}` | same on squared returns |
| `d_rv_ratio_{5_22, 22_66}` | `d_logrv_5 − d_logrv_22`, `d_logrv_22 − d_logrv_66` (vol term-structure slope) |
| `d_volofvol_22` | `std( d_logrv_1, 22 )` |
| `d_downside_semivar_22` | `mean( r_t^2 · 1[r_t<0], 22 )` — leverage/asymmetry channel |
| `d_skew_63`, `d_kurt_63` | rolling skewness, excess kurtosis of `r` over 63 days |
| `d_jump_22` | `max(0, mean(sq,22) − mean(gk,22))` — crude jump-component proxy |
| `d_garch_sigma` | 1-step-ahead conditional σ from GARCH(1,1)-normal, spec below |
| `d_gjr_sigma` | 1-step-ahead conditional σ from GJR-GARCH(1,1,1)-t |
| `d_garch_resid_z` | standardized residual `r_t / d_garch_sigma_t` |
| `d_vix_level`, `d_vix_ratio_5_22` | `ln(VIX_t)`, `ln(mean(VIX,5)/mean(VIX,22))` |
| `d_regime_lo`, `d_regime_mid`, `d_regime_hi` | one-hot of the 3-state expanding-quantile vol regime, spec below |
| `d_dow_{1..4}` | day-of-week one-hot (4 dummies, Monday reference) — cheap, catches weekend-effect variance patterns |

**GARCH specification (exact, `arch` package):**
```
arch_model(100 * r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
arch_model(100 * r, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')   # GJR
```
- Scaling by 100 is required for `arch`'s optimizer conditioning.
- **Expanding-window fit with monthly refit.** At each month-start *m*, fit on `r[0 : m]` only; then for every day *t* in month *m*, produce σ_t by **filtering** (not refitting) the fixed parameters forward over `r[0 : t]`. This exactly mirrors the leakage discipline the project already established for `VMDDecomposerExpanding` in `src/data_pipeline.py` (`refit_interval: 21`).
- Results cached to `data/cubench/cache/garch_sigma_{spec}.parquet`, keyed by a SHA-256 hash of (return series bytes, spec dict), mirroring the existing `hashlib`-based cache pattern already present in `src/data_pipeline.py`. **Content hash, not file size** — this is the direct lesson from todo.txt item 5's size-based staleness bug.
- Failure handling: if the optimizer fails to converge at a given refit point, carry the previous month's parameters forward and log it. Convergence-failure count is reported in the paper.

**Vol regime state (leakage-safe, replaces the earlier HMM idea):**
`d_regime_*` = tercile membership of `d_logrv_22` against the **expanding-window 33rd/67th percentiles computed on `[0 : t]` only** (`.expanding(min_periods=504).quantile(q)`). An HMM is explicitly *not* used: fitting an HMM requires either the full sample (leakage) or an expanding refit whose state labels are not identified across refits (label-switching), which would silently corrupt a one-hot feature. Terciles are transparent, causal, and reproducible. Documented as a deliberate simplification with this reasoning.

### 2.5 Leakage safety — the enforcement layer

Feature engineering is not trusted; it is tested. `tests/test_leakage.py`, all of which must pass before any model trains (this is a pre-registered gate):

1. **`test_causal_perturbation`** — the decisive test. For every feature column *f* and 30 randomly-chosen indices *t*: rebuild the full feature matrix from a raw panel in which **all rows after *t* have been replaced by NaN**; assert `f[t]` is bit-identical to the value from the unmodified build. Any feature that changes has look-ahead. This catches whole-series `fit_transform`, non-causal `.rolling(center=True)`, global standardization, and future-fill in one shot.
2. **`test_pit_monthly_join`** — for `PPIACO`, assert that for every daily row *t*, the joined reference month satisfies `month_end + lag ≤ t`. Assert the naive `reindex().ffill()` join produces a *different* series (proving the PIT join is actually doing work, not a no-op).
3. **`test_fred_coverage_assertions`** — hard-assert first-observation dates: `DFII10 ≤ 2010-01-05`, `BAA10Y ≤ 2010-01-05`, `PPIACO ≤ 2009-01-01`, `INDPRO ≤ 2009-01-01`. Fail loudly on regression; the HY OAS incident proves FRED coverage silently changes under this project.
4. **`test_no_target_in_features`** — assert no feature column name matches the target-construction regex and that `|corr(feature_t, target_t)| < 0.99` for all features.
5. **`test_garch_causality`** — assert `d_garch_sigma[t]` is unchanged when `r[t+1:]` is nulled.
6. **`test_embargo_disjointness`** — for every walk-forward fold, assert `max(train_target_end_date) < min(oos_feature_date)` where `train_target_end_date` accounts for the *h*-day forward span of the target.
7. **`test_no_nan_leak`** — assert `dropna` is applied only at row level after all features are built, never column-wise imputation using future data; assert forward-fill limits are ≤5 days on daily series (existing convention in `src/data_pipeline.py`).

### 2.6 Targets

For horizon *h* ∈ {1, 5, 22}, computed at time *t* using data from *t+1 … t+h*:

- **T1 (primary, regression):** `y1_h = ln( sqrt( (252/h) * Σ_{i=1..h} gk_{t+i} ) )` — log annualized forward realized volatility. Log scale because Log-HAR is the literature's strong form (Brini 2026) and because RV is right-skewed and lognormal-ish.
- **T2 (quantile regression):** `y2_h = Σ_{i=1..h} r_{t+i}` — cumulative forward log return. Predicted at τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} (7 quantiles).
- **T3 (classification, secondary/diagnostic):** `y3_h = 1[y2_h > 0]`.

Horizons **{1, 5, 22} only**. h=10 is dropped deliberately (it carries no information not spanned by 5 and 22, and each extra horizon inflates the multiple-testing family — see §4.5).

---

## 3. Complete Model Roster

**12 model families × 3 targets × 3 horizons.** Every model consumes the identical feature matrix and identical folds. Any model that structurally cannot (ARIMA/GARCH/HAR are univariate) is explicitly flagged in the results table as `[univariate]`.

### 3.1 Tier 0 — Statistical nulls (mandatory, 4 models)

| ID | T1 (log RV) | T2 (quantiles) | T3 (direction) |
|---|---|---|---|
| `null_persist` | `ŷ = d_logrv_h(t)` (yesterday's RV, scaled to horizon) | empirical quantiles of last 252 returns × √h | `sign(r_t)` |
| `null_rollmean` | expanding-window mean of `y1_h` on training data only | — | — |
| `null_zero` | — | `ŷ_τ = 0` ∀τ | always-up (`ŷ=1`) |
| `null_majority` | — | — | training-set majority class, **and** always-down, both reported |

`null_zero`/`null_majority` are the base-rate reference against which every T3 cell's excess accuracy is measured (Cheung 2026 protocol).

### 3.2 Tier 1 — Econometric (4 models)

| ID | Spec |
|---|---|
| `har_rv` | **THE headline benchmark.** `ln RV_{t→t+h} = β₀ + β_d·d_logrv_1 + β_w·d_logrv_5 + β_m·d_logrv_22 + ε`, OLS on log RV (Log-HAR form, per Brini 2026). Refit annually on expanding window, Newey-West(h+5) standard errors. |
| `har_rv_q` | HAR with a quarterly term (`+ β_q·d_logrv_66`) and the jump component `d_jump_22` — the HAR-J/HAR-RV-Q family, included so the paper cannot be accused of using a deliberately-minimal HAR. |
| `garch11` / `gjr_garch` | `arch` package, spec per §2.4, expanding annual refit; h-step-ahead variance forecast via `.forecast(horizon=h, method='analytic')`, aggregated and log-transformed to match T1's scale. Also supplies T2 via the fitted conditional distribution's quantiles (normal for GARCH, Student-t for GJR). |
| `arima` | `statsmodels.tsa.arima.model.ARIMA` on `r_t`, order selected once per fold by AIC over `p,q ∈ {0,1,2}, d=0` on the training window only. **Reused from `src/models/baselines.py::ARIMABaseline`** as an implementation reference; re-implemented in the CuBench walk-forward harness rather than called across module boundaries. T1 via squared-residual variance forecast; T3 via `sign(r̂)`. Expect textbook behaviour: strong in-sample, weak OOS. |

### 3.3 Tier 2 — Linear ML (1 model)

`elasticnet` — `sklearn.linear_model.ElasticNetCV` for T1, `QuantileRegressor` (α=1e-4, `solver='highs'`) for T2, `LogisticRegression(penalty='elasticnet', solver='saga')` for T3. Features standardized with a `StandardScaler` **fit on the training fold only**, inside a `sklearn.pipeline.Pipeline` (this is the single most common leakage vector in linear pipelines and the Pipeline construction is what prevents it).

### 3.4 Tier 3 — Gradient-boosted trees (4 models, LightGBM PRIMARY)

**`lgbm` — the primary model. Exact objectives, one per target:**

| Target | LightGBM config |
|---|---|
| **T1** | `objective='regression'` (L2 on log RV). Additionally report **QLIKE evaluated on the variance scale** via a custom `feval`: `QLIKE = mean( RV_true/RV_pred − ln(RV_true/RV_pred) − 1 )`. L2-on-log is the training objective; QLIKE is a reported evaluation metric, not a training objective — this is stated explicitly because Wang & Lu's HAR result is QLIKE-based and the comparison must be like-for-like on the metric. **Secondary pre-registered run:** `objective='gamma'` on the *variance* scale, whose gradient is QLIKE-equivalent, reported as a robustness row. |
| **T2** | `objective='quantile', alpha=τ` — **7 independent models per (fold, horizon)**. Quantile crossing is checked explicitly and, where present, repaired by post-hoc isotonic sorting across τ; the crossing *rate before repair* is reported as a calibration honesty metric (feasibility report §5 flags this as a known failure mode). |
| **T3** | `objective='binary'`, `metric='binary_logloss'`. **Probability output is the primary artifact**; the 0.5-threshold class label is reported but the calibration curve and Brier score are what get emphasized, precisely to avoid the DA-artifact trap. |

**Hyperparameters (fixed, not tuned per fold — a pre-registered anti-overfitting choice given N≈4,000):**
```
n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=5,
min_child_samples=60, subsample=0.8, subsample_freq=1,
colsample_bytree=0.7, reg_lambda=1.0, reg_alpha=0.1,
n_jobs=-1, verbosity=-1, random_state=<seed>
```
These are deliberately shallow/heavily-regularized for a small, low-SNR tabular problem. **HPO policy:** a single Optuna study (50 trials, TPE) is run **once**, on the initial training block (2010–2014) only, using an internal 2013–2014 validation split, *before* any OOS year is touched, and the resulting parameters are then frozen for all 11 OOS years. This is written into the pre-registration. No per-fold tuning, no OOS-informed tuning. `src/hpo.py`'s Optuna scaffolding is reused.

**Seeds:** every tree model is run with **5 seeds** (42, 43, 44, 45, 46) and results reported as mean ± std across seeds. Single-seed tree results are not reported anywhere. (This is the direct methodological carry-over from the DiAG seed-instability lesson of Paper 1 — same discipline, different model class.)

| ID | Spec |
|---|---|
| `xgboost` | `xgboost.XGBRegressor/XGBClassifier`, matched capacity (`max_depth=5, eta=0.03, n_estimators=600, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, lambda=1.0`). T2 via `objective='reg:quantileerror'` with `quantile_alpha` — reported as comparison only, with the crossing caveat. |
| `catboost` | `CatBoostRegressor/Classifier`, `depth=5, learning_rate=0.03, iterations=600, l2_leaf_reg=3`. T2 via `loss_function='MultiQuantile:alpha=...'` — the one implementation that produces all quantiles jointly, so its crossing rate is a natural comparison to LightGBM's per-quantile approach. CatBoost is included specifically because it was the *winner* in the 18M-observation study that motivated this whole pivot. |
| `randomforest` | `sklearn.ensemble.RandomForestRegressor/Classifier`, `n_estimators=500, max_depth=12, min_samples_leaf=20, max_features='sqrt'`. Included because the MDPI 19(3):203 honest benchmark found RF remains a strong baseline. T2 via `QuantileRegressor`-free route: quantiles of the leaf-level training-target distribution (quantile regression forest logic, hand-implemented, unit-tested). |

### 3.5 Tier 4 — Deep learning comparison points (2 models) — what "reused" means concretely

**Decision: same architecture code, retrained from scratch on the CuBench feature matrix under the CuBench walk-forward protocol.** Not cited from Paper 1, not re-used checkpoints.

Concretely: copy `LSTMBaseline` and `TransformerBaseline` from `src/models/baselines.py` into `src/cubench/models/deep.py` **verbatim** (architecture, hidden dims, layer counts, dropout preserved exactly as in `configs/default.yaml`: `hidden_dim=64, num_heads=4, temporal_layers=2, dropout=0.1`), change only the input dimension to accept the CuBench feature vector and the output head to emit the three CuBench targets. Train on CPU (torch CPU wheel), `lookback=60`, `batch_size=32`, `epochs=100` with `EarlyStopper(patience=20)` reused from `src/utils.py`, Adam `lr=1e-3, weight_decay=1e-5`, 3 seeds each.

Rationale for retraining rather than citing: Paper 1's LSTM/Transformer numbers come from a single 2022–2025 holdout with a different feature set; pasting them next to 11-fold walk-forward CuBench numbers would be an invalid comparison and a referee would say so immediately. Retraining costs ~2–4 CPU-hours total and buys a valid row.

**PatchTST is dropped.** It was a "nice-to-have" in the earlier synthesis; with two DL comparison points already retrained under a valid protocol, a third adds a dependency and implementation risk for a marginal row. Its motivating argument (channel-independence as implicit regularization against spurious cross-asset correlation) belongs in the Discussion as a *citation explaining Paper 1's failure*, not as a model to build.

### 3.6 Tier 5 — The bridge row

`vmd_mfgnn` — **imported as a results row from Paper 1**, not retrained. Sourced from `results/all_results.json` / `results/post_hoc_analysis.json`. Displayed in a visually-separated table section with the mandatory footnote: *"VMD-MFGNN results are reproduced from Jain & Panigrahi (Paper 1) under that paper's protocol (single 2022–2025 holdout, close-only 8-variable feature set) and are **not** directly comparable to the walk-forward CuBench rows; they are included to connect the two studies, not to rank the model."* This is the explicit bridge that makes the two papers one research program.

---

## 4. Complete Evaluation Protocol

### 4.1 Walk-forward fold structure

- **Scheme:** expanding window, annual refit.
- **Initial training block:** 2010-01-04 → 2014-12-31 (~1,255 trading days).
- **OOS years:** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 → **11 non-overlapping OOS years**, ~2,770 OOS observations.
- **Fold *k* (k=0…10):** train on `[2010-01-04, (2014+k)-12-31 − E]`, evaluate on `[(2015+k)-01-01, (2015+k)-12-31]`.
- **Embargo E = h + 5 trading days**, i.e. 6 / 10 / 27 days for h = 1 / 5 / 22, **removed from the END of the training window** (purging). This is required because `y_h(t)` spans *t+1…t+h*; without it, the last *h* training targets overlap the first OOS days. Placement is at the train-side boundary only; no gap is inserted inside the OOS block (OOS days are consecutive and every OOS prediction is genuinely out-of-sample regardless).
- **Overlapping-target handling:** for h ∈ {5, 22} consecutive targets overlap by construction. This is *not* corrected by subsampling (which would discard 95% of the data); instead, **all inference on h>1 uses HAC/Newey-West variance with bandwidth h−1**, and the DM test uses the HLN small-sample correction (§4.4). This is stated explicitly.
- Fold definitions are generated once by `src/cubench/folds.py::make_folds()` and **serialized to `results/cubench/folds.json`** so every downstream analysis provably uses identical splits.

### 4.2 Metrics per target

**T1 (log RV):** RMSE and MAE on log scale; **R²_OOS vs the `null_rollmean` benchmark** (Gu-Kelly-Xiu convention: `1 − Σ(y−ŷ)²/Σ(y−ȳ_train)²`); **QLIKE on the variance scale** (primary loss for model ranking, matching Wang & Lu); MSE on the variance scale; Mincer-Zarnowitz regression `y = a + b·ŷ` with a joint test of (a,b)=(0,1).

**T2 (quantiles):** **Pinball loss per τ and averaged** (primary); **CRPS** approximated from the 7-quantile grid; **PIT/coverage** — empirical frequency of `y < ŷ_τ` vs nominal τ, with a Kupiec unconditional-coverage LR test per τ; **quantile crossing rate**; 90% and 50% interval width.

**T3 (direction):** raw DA; **test-window base rate**; **excess DA over majority baseline**; **predicted-positive rate**; full confusion matrix; Pesaran-Timmermann statistic and p-value; McNemar test vs `null_majority`; Brier score; log loss; reliability-curve calibration. **Raw DA is never reported without the base rate immediately adjacent in the same table row.**

### 4.3 Statistical tests

**Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction.**
Implementation decision (given the verified absence of `statsmodels.tsa.stattools.diebold_mariano_test` in 0.14.6): **write an audited implementation in `src/cubench/stats_tests.py::dm_test_hln()`**, extending the existing `src/utils.py::diebold_mariano_test` (which currently has a Newey-West term but **no HLN correction** and uses a normal reference distribution — both are wrong for 11-fold, h>1 settings and must be fixed).

Exact spec:
```
d_t = L(e1_t) − L(e2_t)                       # L = squared error, or QLIKE for T1, pinball for T2
d̄  = mean(d)
γ_k = autocov(d, k)
V   = (γ_0 + 2·Σ_{k=1}^{h−1} γ_k) / n
DM  = d̄ / sqrt(V)
HLN = DM · sqrt( (n + 1 − 2h + h(h−1)/n) / n )
p   = 2·(1 − t_cdf(|HLN|, df=n−1))            # Student-t(n−1), NOT normal
```
Validated by unit tests against (a) the `dieboldmariano` PyPI package on 200 random error pairs (agreement to 1e-8 on the uncorrected statistic), and (b) two hand-worked textbook examples with known values. Both validations are committed as `tests/test_stats_tests.py`.

**Pesaran-Timmermann (1992).** Python support is thin; **hand-implement, audited**:
```
P̂  = mean(1[sign(ŷ)==sign(y)])
P̂_y = mean(1[y>0]);  P̂_x = mean(1[ŷ>0])
P̂* = P̂_y·P̂_x + (1−P̂_y)(1−P̂_x)
var(P̂ ) = P̂*(1−P̂*)/n
var(P̂*) = (2P̂_y−1)²·P̂_x(1−P̂_x)/n + (2P̂_x−1)²·P̂_y(1−P̂_y)/n + 4·P̂_y·P̂_x(1−P̂_y)(1−P̂_x)/n²
PT = (P̂ − P̂*) / sqrt(var(P̂) − var(P̂*))   ~ N(0,1)
```
Validation plan: **cross-check against R's `rugarch::DACTest`** on 3 fixed test vectors. If R is not installable in the timeline, the fallback is a **bootstrap validation**: simulate 10,000 independent (ŷ, y) pairs with known base rates and no skill, confirm the empirical PT rejection rate at α=0.05 is within [0.04, 0.06]. Committed as a unit test either way. Guard against the known degenerate case: when `P̂_x ∈ {0,1}` (constant forecast) the denominator → 0; the implementation must return `p = NaN` with a `"degenerate_constant_forecast"` flag rather than a spurious value. **This is the base-rate diagnostic firing at the test level** and it is exactly the failure Paper 1 discovered in its own tables.

**Multiple-testing correction — Holm-Bonferroni, with a pre-declared family.**
The family is declared in the pre-registration and is: **{all models except nulls} × {3 horizons} × {3 targets}**, applied *separately within each target* (T1: 11 models × 3 horizons = 33 tests; T2: 33; T3: 33). Ablation and regime-segmented tests form **separate, separately-corrected families** and are labelled "exploratory" in the paper. `statsmodels.stats.multitest.multipletests(pvals, method='holm')`. Both raw and Holm-adjusted p-values are reported in every table; conclusions cite only the adjusted values.

**Deflated Sharpe Ratio and Probability of Backtest Overfitting.**
Library: **`pypbo`** (github.com/esvhd/pypbo), installed from a pinned commit SHA recorded in `requirements_cubench.txt`. Do not use `mlfinlab` (licensing/availability uncertain per feasibility report).
- **DSR** inputs: observed SR, number of independent trials `N` = the declared family size for the relevant target (33), skewness and kurtosis of the strategy's return series, and sample length. Reported for every strategy whose gross Sharpe is presented.
- **PBO** via `pypbo`'s CSCV: **S = 16 combinatorially-symmetric partitions** of the OOS period; report the PBO estimate, the logit distribution plot, and the performance-degradation regression slope. A PBO > 0.5 is pre-committed to be reported as "backtest is likely overfit" regardless of how good the headline number looks.
- Independent sanity check: implement DSR's closed form once by hand and assert agreement with `pypbo` to 1e-6 on one case (this project does not trust unvalidated third-party numerics — the standing "recompute from raw" practice).

### 4.4 Transaction-cost-adjusted backtest

**Strategy definition (single, pre-registered, no variants):** at each *t*, position `w_t = clip(ŷ_direction_signal, −1, +1)` scaled inversely by the model's own T1 volatility forecast: `w_t = sign(q̂_0.5) · min(1, σ_target / σ̂_t)` with `σ_target = 15%` annualized. This is the practitioner "forecast volatility, then size the bet" structure the practitioner track documented, and it is the *only* place the three targets combine.

**Cost model.** COMEX HG: 25,000 lb/contract, minimum tick 0.0005 $/lb = $12.50, at a ~$4.00/lb price a one-tick half-spread ≈ **1.25 bps**. Cost is charged as `cost_bps × |w_t − w_{t−1}|` per rebalance.

**Report a COST CURVE, not a single breakeven number:** net Sharpe evaluated at `cost_bps ∈ {0, 1, 2, 5, 10, 20, 50}` round-trip, plotted with the break-even cost marked. Base case for headline text: **2 bps**. Also report turnover, max drawdown, hit rate, and the payoff-asymmetry ratio (mean win / mean loss) — the practitioner track's point that CTA hit rates are routinely below 50% and the edge lives in asymmetry.

**Calibration text pre-committed:** realistic gross Sharpe 0.4–0.8, net 0.2–0.5. A gross Sharpe above 1.5 must be treated as a suspected bug and triggers the leakage-audit re-run before it may be reported.

### 4.5 Regime segmentation (two axes)

**Axis 1 — volatility tercile.** OOS days bucketed by `d_regime_{lo,mid,hi}` (expanding-quantile terciles of trailing 22-day RV, §2.4). All headline metrics recomputed within each bucket.

**Axis 2 — calendar regime** (pre-declared, chosen for economic meaning and roughly balanced sample size):
| Regime | Window | Character |
|---|---|---|
| R1 Post-supercycle bust | 2015-01-01 → 2016-12-31 | China slowdown, commodity bear |
| R2 Synchronized growth / trade war | 2017-01-01 → 2019-12-31 | Escondida strike (2017), US-China tariffs |
| R3 COVID shock & rebound | 2020-01-01 → 2020-12-31 | crash + V-recovery |
| R4 Stimulus rally & 2022 shock | 2021-01-01 → 2022-12-31 | inflation, LME nickel crisis, PMI-copper breakdown |
| R5 Energy-transition / destocking | 2023-01-01 → 2025-12-31 | Cobre Panamá shutdown (2023), Grasberg mudslide (Sep 2025) |

Regime-segmented tests are an **exploratory family**, Holm-corrected within themselves, and labelled as such. This directly serves the near-mandatory "robustness across multiple non-overlapping regime-spanning periods" requirement.

### 4.6 Block-wise ablation design

Strictly nested, added in evidence-strength order (strongest-evidenced first, so each subsequent block must earn its place):

| ID | Feature set | Rationale for position |
|---|---|---|
| **A0** | Price-only: `a_r_lag1`, `a_r_lag2`, `d_logrv_{1,5,22}` | The mandatory price-only baseline. Papers that skip this get read skeptically. |
| **A1** | A0 + **Block D** (vol/regime, 26 feat) | Volatility is where the evidence says signal lives — it goes in first, so it cannot free-ride on later blocks. |
| **A2** | A1 + **Block A** (trend, 16 feat) | AQR hedging-pressure mechanism; second-best-evidenced. |
| **A3** | A2 + **Block C-cross** (14 feat: metals/equity/oil daily returns) | Modest but real (r 0.12–0.44), contemporaneous. |
| **A4 = FULL** | A3 + **Block C-macro** (8 feat: yields/credit/PPI/INDPRO) | Weakest and most leakage-prone; goes last so its marginal contribution is measured against everything else. **A4 vs A3 is also the empirical bound on monthly-macro-revision risk** (§2.3). |
| **A5** | A4 + **Block B′** (curve proxy) | Supplementary only, never headline. |
| **A6** | A4 + China confidence proxy | Supplementary only; expected null, reported as null. |
| **A7** | A4 with **all Block C removed** | Direct answer to "does macro help at all?" |

Ablations run for LightGBM only (5 seeds), across all 3 targets × 3 horizons. Marginal contribution = ΔQLIKE (T1) / Δpinball (T2) / Δlog-loss (T3), with DM tests between consecutive rungs, Holm-corrected within the ablation family.

### 4.7 Base-rate / constant-forecast diagnostic

Applied to **every T3 result cell, without exception**, and to any T1/T2 cell whose prediction dispersion is anomalously low. Per-cell output block (`src/cubench/diagnostics.py::base_rate_report()`):

1. raw directional accuracy
2. test-window positive rate (the base rate)
3. majority-class accuracy = `max(base_rate, 1−base_rate)`
4. **excess DA = raw DA − majority-class accuracy** ← the number that gets bolded, not raw DA
5. predicted-positive rate
6. **sign-concentration** = `max(pred_pos_rate, 1−pred_pos_rate)`; flag if > 0.95
7. binomial test of predicted signs against 0.5
8. ratio `std(ŷ)/std(y)`; flag if < 0.20
9. confusion matrix
10. PT p-value (NaN + degenerate flag if constant, §4.3)
11. McNemar vs `null_majority`
12. **verdict field** ∈ {`ok`, `suspect_low_dispersion`, `constant_forecast_artifact`}

Framing in the paper: *"We apply the base-rate honesty protocol of Cheung (2026), which formalizes the phenomenon we independently encountered in our own prior work, and extend it from directional accuracy to volatility and quantile forecasts."* **Extension, not invention.** Two things are genuinely new and may be claimed: (a) the dispersion-ratio + sign-concentration pair as a *pre-hoc* screen (Paper 1 demonstrated empirically that dispersion alone is insufficient — the Transformer_h5 case had a dispersion ratio of 0.42 yet 100% same-sign predictions), and (b) applying it across a full multi-model × multi-horizon × multi-target benchmark grid rather than to a single model.

---

## 5. Pre-registration Document — `docs/preregistration_paper2.md`

**Hard gate: this file is written, reviewed by the supervisor, and committed with a signed git commit BEFORE `src/cubench/models/` contains a single trained model.** The commit SHA and timestamp are quoted verbatim in the manuscript's Methods section. Nothing in it may be edited afterwards; changes are appended as a dated, numbered **Amendment Log** with a reason, and every amendment is disclosed in the paper.

Required contents, section by section:

1. **Header** — title, authors, date, git SHA of the repo state, statement: "This document was committed before any model was trained on out-of-sample data."
2. **Research questions**, pre-numbered:
 - RQ1: Does gradient boosting on engineered features beat Log-HAR-RV on h-day copper realized volatility, by QLIKE, under 11-fold expanding walk-forward?
 - RQ2: Do LightGBM quantile forecasts achieve better calibration (Kupiec coverage, pinball) than GARCH/GJR-implied and empirical-quantile baselines?
 - RQ3: Is copper daily/weekly/monthly return direction predictable above its own base rate, PT-significant after Holm correction?
 - RQ4: Which feature blocks (D, A, C-cross, C-macro) contribute ablation-verified incremental value, and does that differ by volatility regime and calendar regime?
 - RQ5: Does any apparent edge survive transaction costs, DSR, and PBO?
3. **Complete data specification** — the exact tables of §1, with tickers, IDs, ranges, and the Block B Tier-3 decision with its justification.
4. **Complete feature list** — all ~68 features by name and formula (§2). **Feature list is frozen at end of Week 2 (2026-08-30). No feature may be added after that date.** Any post-freeze addition must be reported in the Amendment Log and shown separately from the headline table.
5. **Complete model roster and hyperparameters** — §3, verbatim, including the "HPO once on 2010–2014 only, then frozen" rule and the 5-seed rule.
6. **Complete evaluation protocol** — fold boundaries as explicit dates, embargo formula, all metrics, all tests, the declared multiple-testing families and their sizes, the cost grid, the regime definitions.
7. **Pre-committed success/failure criteria**, stated so that no outcome can be re-interpreted after the fact:
 - RQ1 *success*: LightGBM QLIKE improvement over `har_rv` ≥ 2% at ≥2 of 3 horizons, DM-significant after Holm. RQ1 *failure*: HAR wins or ties → **"HAR-RV remains unbeaten on copper realized volatility by gradient boosting with 68 engineered features" is reported as the headline finding**, and this is an equally publishable result corroborating Wang & Lu 2024 and Brini 2026.
 - RQ3 *expected outcome*: excess DA in [−1pp, +3pp], 0–2 of 9 cells PT-significant after Holm. **Explicit pre-commitment: directional results are reported exactly as they land. If DA is 50%, that is the finding, corroborates the LME variance-ratio study and Portnaya 2026, and is reported without softening. If DA exceeds 60% at any horizon, that cell is treated as a suspected leakage bug and the full §2.5 leakage suite is re-run before the number may appear in any draft.**
 - RQ5 *pre-commitment*: if PBO > 0.5, the paper states the backtest is likely overfit regardless of Sharpe.
8. **Anti-p-hacking commitments** (explicit, because this project has caught itself three times):
 - No metric added after the freeze may be used to select the headline result.
 - No horizon, target, model, or regime may be dropped from any results table after seeing results.
 - The full grid is reported; no cherry-picking of "best" cells.
 - Bolding rule pre-declared: a cell may only be bolded if it is best *and* passes the base-rate diagnostic with verdict `ok` *and* is Holm-significant.
 - **Named prohibition:** "Under deadline pressure in October/November, the team will not revive the Gumbel-softmax graph experiment, will not switch the primary target away from volatility, and will not re-tune hyperparameters on OOS folds."
9. **Novelty statement** — explicit, pre-committed: CuBench is a rigorous tabular-ML *benchmark and ablation study*, not a novel architecture. The base-rate diagnostic is an application/extension of Cheung 2026 (arXiv:2607.12248), not an invention. Prior art acknowledged up front: Wang & Lu 2024, Wang & Li 2024, Brini 2026, Portnaya 2026, Cheung 2026, Fang & Slepaczuk 2026, Dudek et al. 2025.
10. **Known limitations, declared in advance** — no true carry block; final-vintage macro revisions; continuous-contract roll rule undisclosed by Yahoo; range-based RV proxy rather than intraday RV; single asset; free-data-only constraint.
11. **Amendment Log** — empty at commit time.

---

## 6. Week-by-Week Execution Plan (2026-08-19 → 2026-11-30)

Mapped onto `docs/research_plan_CP1.md`'s four-month structure. Month phases are unchanged; each is filled in at week granularity.

### AUGUST 2026 — Foundation & Protocol Design
*(CP-1 output: feasibility report; committed pre-registration; extended raw data pipeline)*

**Week A1 — Aug 19–23 (unblock long poles + data)**
- **[Day 1, HIGHEST PRIORITY, HUMAN] Email Dr. Panigrahi requesting arXiv endorsement** for `q-fin.ST` (cross-list `cs.LG`). Endorsement is confirmed still required for first-time submitters with no category-shopping shortcut; latency 3–7 days and it is the single largest schedule risk. **Same day, register an SSRN account** — this removes arXiv from the critical path entirely. *Blocks: November posting. Nothing else depends on it.* (Also closes todo.txt item 7's arXiv sub-item.)
- **[Day 1]** Create `src/cubench/` package skeleton + `configs/cubench.yaml` + `requirements_cubench.txt`; install the new packages into `.venv_corr`; smoke-test `import lightgbm, xgboost, catboost, arch, shap, pypbo`.
- **[Day 1–2] Re-verification spike, 90 min timeboxed each, pre-decided fallbacks:**
 - S1: `HG=F` full OHLC pull 2010-01-04→2025-12-31 via yfinance; assert ≥4,000 rows and non-null H/L/O. *Fallback: the cached `data/raw_correlation_check/yf_copper.csv`.*
 - S2: **COMEX contract-month coverage audit** (192 tickers, §1.5) → `results/cubench/data_audit/comex_contract_month_coverage.csv`. *Outcome pre-decided: Tier 3.*
 - S3: FRED pulls for `DFII10`, `BAA10Y`, `PPIACO`, `INDPRO` via curl; **assert first-obs dates** per §2.5 test 3. *Fallback if `BAA10Y` truncated: `STLFSI4` weekly, or drop the credit channel and report it.*
 - S4: `CLP=X` and `QC=F` coverage checks with the drop rules of §1.4/§1.5.
- **[Day 3–4]** Write `src/cubench/data.py` (unified downloader: yfinance + curl-FRED), `src/cubench/pit.py` (point-in-time monthly join). Write `data/cubench/raw/` snapshot + `data_source_log.json` in the same format the correlation notebook already uses.
- **[Day 5]** Write `docs/cubench_data_availability.md` (the audit write-up, incl. the Block-B and HY-OAS negative findings). **Send a 1-page interim note to the supervisor.**

**Week A2 — Aug 24–30 (features + pre-registration freeze)**
- Implement `src/cubench/features.py`: Blocks A, B′, C, D per §2. Implement `src/cubench/garch.py` with content-hash caching.
- Implement `src/cubench/targets.py` (T1/T2/T3, h ∈ {1,5,22}).
- Implement and pass **all 7 tests in `tests/test_leakage.py`** (§2.5). *This is a gate: nothing proceeds until green.*
- Write, review with supervisor, and **commit `docs/preregistration_paper2.md`** (§5). **Feature list frozen 2026-08-30.**
- Update `docs/research_plan_CP1.md`: remove ICAIF-as-near-term-target framing (deadline passed), replace with the D11 venue path. *(This is the outstanding NEXT ACTION already flagged in STATUS.md.)*
- **Aug 31 checkpoint:** feature matrix `data/cubench/features.parquet` exists, leakage tests green, pre-registration committed.

### SEPTEMBER 2026 — Feature Engineering & Model Implementation
*(CP-1 milestone, end of Sept: pipeline leakage-verified; full roster implemented; walk-forward engine operational on a pilot subset)*

**Week S1 — Sep 1–6:** `src/cubench/folds.py` (11 folds + embargo, serialized to `folds.json`); `src/cubench/walkforward.py` (the engine: takes a model adapter, folds, target spec; emits per-fold predictions to `results/cubench/predictions/`). Implement the uniform `ModelAdapter` interface (`fit(X,y)`, `predict(X)`, `predict_quantiles(X, taus)`, `predict_proba(X)`). **Pilot: run `null_persist` + `har_rv` end-to-end on h=1** to prove the engine.

**Week S2 — Sep 7–13:** Tier 0 nulls (4) + Tier 1 econometric (`har_rv`, `har_rv_q`, `garch11`, `gjr_garch`, `arima`). Unit-test HAR coefficients against a hand-computed OLS on a 200-row toy series. Cache GARCH refits.

**Week S3 — Sep 14–20:** Tier 2 (`elasticnet`, Pipeline-wrapped scaler) + Tier 3 trees (`lgbm`, `xgboost`, `catboost`, `randomforest`) with all three objectives each. Implement quantile-crossing detection + isotonic repair. **Run the single Optuna HPO study on 2010–2014 only and freeze the parameters into `configs/cubench.yaml`.**

**Week S4 — Sep 21–30:** Tier 4 deep (`lstm`, `transformer` ported from `src/models/baselines.py`, retrained on CPU). Implement `src/cubench/stats_tests.py` (`dm_test_hln`, `pesaran_timmermann`) + `tests/test_stats_tests.py` validations. Implement `src/cubench/diagnostics.py::base_rate_report()`. **Sep 30: mid-semester milestone — demo the pilot walk-forward run to the supervisor.**

### OCTOBER 2026 — Experimentation, Evaluation & Rigor Analysis
*(CP-1 milestone, end of Oct: full grid executed; all rigor analyses complete)*

**Week O1 — Oct 1–7:** **Execute the full grid** — 12 model families × 3 targets × 3 horizons × 11 folds × 5 seeds (trees) / 3 seeds (deep). Expected wall-clock: trees + econometric ≈ 45–90 min; deep ≈ 3–5 hrs CPU. Persist every per-fold prediction array to `results/cubench/predictions/{model}_{target}_h{h}_seed{s}.npy` (raw `.npy`, so the standing recompute-from-raw practice works). Aggregate to `results/cubench/all_results.json` via `src/utils.py::save_results` (numpy-safe, existing convention). Re-run twice and diff to confirm determinism.

**Week O2 — Oct 8–14:** Statistical layer — DM(HLN) all pairs vs `har_rv` and vs `null_persist`; PT + McNemar on all T3 cells; Holm-Bonferroni across the three declared families; Kupiec coverage tests on all T2 cells; Mincer-Zarnowitz on T1. **Apply the base-rate diagnostic to every cell** and produce `results/cubench/base_rate_diagnostics.json`. **Any cell with verdict ≠ `ok` is investigated before it is written about.**

**Week O3 — Oct 15–21:** Backtest layer — the vol-scaled strategy, cost curve at 7 cost levels, DSR + PBO (CSCV S=16) via `pypbo` with the hand-derived DSR cross-check. Turnover, drawdown, payoff asymmetry.

**Week O4 — Oct 22–28:** Ablations A0→A7 (LightGBM, 5 seeds, all targets/horizons) + both regime axes (vol terciles × 5 calendar regimes) + SHAP (`shap.TreeExplainer` on LightGBM, global bar + beeswarm, **plus SHAP recomputed separately within each volatility tercile** — the interpretability story the collapsed graph in Paper 1 could never deliver).

**Week O5 — Oct 29–31:** **Full recompute-from-raw verification pass** — regenerate every reported number from the `.npy` prediction files with an independent script (`scripts/cubench_verify.py`), diff against `all_results.json`, investigate every discrepancy. *(This standing practice has caught real bugs twice in this project.)* Write `results/cubench/findings_summary.md`. **Supervisor review meeting — pre-final milestone.**

### NOVEMBER 2026 — Manuscript, Review & Submission
*(CP-1 milestone: manuscript finalized, all numbers recomputed, CP-I presentation)*

**Week N1 — Nov 1–8:** Draft the manuscript in **venue-neutral single-column preprint format** (D11 — no ACM `sigconf` port, saving 6–10 hrs). Sections: Intro (+RQ1–RQ5 block, mirroring Paper 1's structure), Related Work (all six research tracks + the 15 arXiv citations already found in the feasibility report — this is already researched, write it first), Data (incl. the Block-B/HY-OAS negative data findings), Features + point-in-time methodology, Evaluation protocol, Results, Ablations, Regime analysis, Base-rate diagnostics, Limitations, Conclusion.

**Week N2 — Nov 9–15:** All figures/tables from `results/cubench/`. **Three-reviewer internal pipeline** (the same process that caught real problems on Paper 1 every single time): reviewer 1 methodological/statistical correctness, reviewer 2 clarity/overclaiming, reviewer 3 plagiarism/AI-detection/citation-attachment.

**Week N3 — Nov 16–23:** Fix pass on the review punch list. **Second full recompute-from-raw** of every number appearing in the final draft. Supervisor revision cycle. Verify the compiled PDF is the current one before any upload (*direct callback to the stale-`main.pdf` near-miss on Paper 1*).

**Week N4 — Nov 24–30:** **Post to arXiv** (if endorsement cleared in August) **and SSRN** (unconditional). Freeze the repo, tag `cubench-v1.0`, write the reproduction README. Prepare and deliver the **CP-I presentation**; obtain supervisor sign-off. Post-capstone submission targets queued: Journal of Forecasting (primary), Finance Research Letters (if compressible), ICAIF 2027.

**Dependency-critical path:** arXiv endorsement (A1 D1) → N4 posting; leakage tests green (A2) → any training; pre-registration committed (A2) → any OOS evaluation; folds.json (S1) → all model runs; full grid (O1) → all of O2–O4; O5 recompute → N1 writing.

---

## 7. File / Directory Layout

New work lives under a `cubench` namespace so Paper 1's verified artifacts are never touched.

```
D:\copper\
├─ configs/
│  ├─ default.yaml                  # PAPER 1 — DO NOT EDIT
│  └─ cubench.yaml                  # NEW: tickers/FRED IDs, feature params, folds,
│                                   #      frozen model hyperparams, cost grid, regimes
├─ requirements.txt                 # PAPER 1 — DO NOT EDIT
├─ requirements_cubench.txt         # NEW: lightgbm, xgboost, catboost, arch, shap,
│                                   #      pypbo (pinned SHA), dieboldmariano, torch-cpu
├─ src/
│  ├─ data_pipeline.py, trainer.py, utils.py, models/   # PAPER 1 — read-only reuse
│  └─ cubench/                      # NEW package
│     ├─ __init__.py
│     ├─ data.py                    # yfinance + curl-FRED downloaders, snapshot writer
│     ├─ pit.py                     # point-in-time monthly join + PUBLICATION_LAG table
│     ├─ features.py                # Blocks A / B' / C / D
│     ├─ garch.py                   # GARCH/GJR expanding refit + content-hash cache
│     ├─ targets.py                 # T1 / T2 / T3 construction
│     ├─ folds.py                   # 11 expanding folds + embargo, serializes folds.json
│     ├─ walkforward.py             # the engine + ModelAdapter interface
│     ├─ models/
│     │  ├─ nulls.py, econometric.py, linear.py, trees.py
│     │  └─ deep.py                 # LSTM/Transformer ported verbatim from
│     │                             # src/models/baselines.py, new I/O dims only
│     ├─ metrics.py                 # QLIKE, pinball, CRPS, Kupiec, MZ, R2_OOS
│     ├─ stats_tests.py             # dm_test_hln, pesaran_timmermann, holm wrapper
│     ├─ diagnostics.py             # base_rate_report()
│     ├─ backtest.py                # vol-scaled strategy, cost curve, DSR/PBO via pypbo
│     ├─ ablations.py               # A0..A7
│     └─ regimes.py                 # vol terciles + 5 calendar regimes
├─ scripts/
│  ├─ post_hoc_analysis.py, run_experiments.py         # PAPER 1
│  ├─ cubench_build_data.py         # NEW: pull -> snapshot -> features.parquet
│  ├─ cubench_audit_contracts.py    # NEW: the 192-ticker COMEX coverage audit
│  ├─ cubench_run_grid.py           # NEW: the full experimental grid
│  ├─ cubench_evaluate.py           # NEW: stats + diagnostics + backtest + ablations
│  └─ cubench_verify.py             # NEW: independent recompute from raw .npy
├─ tests/                           # NEW directory
│  ├─ test_leakage.py               # the 7 gates of §2.5
│  ├─ test_features.py              # formula spot-checks on toy series
│  ├─ test_stats_tests.py           # DM/PT validation
│  └─ test_folds.py                 # embargo disjointness
├─ notebooks/
│  ├─ correlation_feasibility_analysis.ipynb           # EXISTING, keep
│  └─ cubench_exploration.ipynb     # NEW: figure drafting only, no pipeline logic
├─ data/
│  ├─ raw_correlation_check/        # EXISTING — the frozen fallback cache
│  └─ cubench/
│     ├─ raw/                       # snapshot CSVs + data_source_log.json
│     ├─ cache/                     # garch_sigma_*.parquet (content-hash keyed)
│     └─ features.parquet
├─ results/
│  ├─ correlation_analysis/         # EXISTING
│  └─ cubench/
│     ├─ folds.json
│     ├─ predictions/               # {model}_{target}_h{h}_seed{s}.npy  (raw arrays)
│     ├─ all_results.json           # via src/utils.py::save_results
│     ├─ significance/              # dm, pt, holm-corrected tables
│     ├─ base_rate_diagnostics.json
│     ├─ backtest/                  # cost curve, DSR, PBO
│     ├─ ablations/, regimes/, interpretability/   # SHAP
│     ├─ data_audit/comex_contract_month_coverage.csv
│     ├─ figures/
│     └─ findings_summary.md
├─ docs/
│  ├─ research_plan_CP1.md          # UPDATE in Week A2 (venue reframe)
│  ├─ preregistration_paper2.md     # NEW — the Week-A2 gate
│  ├─ cubench_data_availability.md  # NEW — Block B / HY OAS negative findings
│  └─ research/                     # EXISTING, read-only
└─ paper2/                          # NEW — separate from paper/ (Paper 1, IEEEtran)
   ├─ main.tex                      # venue-neutral single-column preprint
   ├─ refs.bib                      # external .bib this time (Paper 1's inline
   │                                # thebibliography does not scale to ~50 refs)
   └─ figures/
```

**Conventions carried forward, non-negotiable:** JSON result serialization through `src/utils.py::save_results` (numpy-safe `default=convert`); raw predictions persisted as `.npy` so every reported number is independently recomputable; `hashlib`-based content-hash caching (never file-size-based — todo.txt item 5's root cause); `logging` to both stream and a run log file, per `scripts/run_experiments.py`.

---

## 8. Risk Register

Updated to the post-feasibility-report, post-correlation-notebook state. Risks R1–R6 are new or materially revised since the superseded Opus synthesis.

| # | Risk | Severity | Evidence it is real | Mitigation (concrete) |
|---|---|---|---|---|
| **R1** | **Block B carry data unobtainable** — the "highest-value new feature" cannot be built. | High → **already realized** | `HGZ25.CMX` starts 2020-09-29; `HGZ15.CMX` 404s; CME continuous is licensed. | **Accepted and absorbed (D3).** Tier 3, `B′` proxy ablation-only and never called carry, coverage audit published as a contribution. RQ4 does not depend on Block B. Risk is retired at the design stage rather than carried. |
| **R2** | **FRED coverage silently shrinks under us** — HY OAS went from full history to 3 years in April 2026; another series could follow. | High | Confirmed independently twice (Codex report + this project's own notebook, 754 obs from 2023-08-21). Note this defeats the notebook's own proposed fix — an API key does not restore licence-restricted history. | (a) `BAA10Y` substitution (D4). (b) **Hard first-obs-date assertions in `test_fred_coverage_assertions`** — fail loudly on regression. (c) **`data/cubench/raw/` snapshot is authoritative once written**; the pipeline reads the snapshot, not the network, after Week A1. (d) HY OAS retained only as a 2023–25 robustness appendix. |
| **R3** | **Monthly-macro publication-lag leakage manufactures a beautiful fake result** — flagged as the single most likely leakage vector in the entire design. | Critical | The correlation notebook itself forward-filled monthly series without lag adjustment and flagged this in its own methodology notes as uncorrected. | (a) Mandatory `merge_asof` PIT join with conservative over-estimated lags (§2.3). (b) `test_pit_monthly_join` asserts the naive join produces a *different* series. (c) `test_causal_perturbation` catches it structurally. (d) **Ablation A4-vs-A3 and A7 bound the total contribution of monthly macro** — if macro adds ~nothing (which is the expectation), the leakage risk is arithmetically bounded to ~nothing. |
| **R4** | **HAR-RV is not beaten** — the primary RQ returns a null. | Medium | Wang & Lu 2024 report HAR lowest QLIKE on this exact asset; Brini 2026 finds Log-HAR beaten by only 1 of 9 TSFMs, narrowly; Dudek et al. 2025 find simple linear bases beat sophisticated alternatives on log RV. | **Pre-registered as a fully reportable outcome (§5.7).** "HAR remains unbeaten by 68-feature gradient boosting under 11-fold walk-forward with multiple-testing correction" is a corroborating, citable, publishable finding in a field starved for evaluation papers. The paper's contribution is the protocol + benchmark + ablations, none of which depend on the sign of RQ1. |
| **R5** | **arXiv endorsement not granted / delayed past November.** | Medium | Confirmed still required for first-time submitters; no category-shopping shortcut; 3–7 day latency and it is human-dependent. | Start Day 1 via the supervisor; **SSRN account registered the same day as an unconditional fallback** (no endorsement gate). Manuscript is venue-neutral so no format work is wasted either way. |
| **R6** | **Novelty overclaim on the base-rate diagnostic** — Cheung 2026 already formalized it. | Medium (reputational) | arXiv:2607.12248 documents an apparent ~80% DA explained by a 0.70 base rate. | Pre-registered wording (§5.9, §4.7): cite Cheung as prior art in the abstract, intro, and method. Claim only the two genuine extensions (pre-hoc dispersion+sign-concentration screen; grid-wide application). **Also softens Paper 1's framing** — a note to that effect goes in `paper/main.tex`'s discussion before submission. |
| **R7** | **Deadline-pressure p-hacking in October/November** — the project has caught itself three times. | High | Documented history: the confound-check "reversal" that was a base-rate artifact; two bolded-as-best cells that were artifacts; the retracted DA claims. | Pre-registration (§5.8) with named prohibitions; frozen feature list (Aug 30); frozen hyperparameters; bolding rule pre-declared; full grid reported with no cell dropped; three-reviewer pipeline in Week N2; recompute-from-raw twice (O5 and N3). |
| **R8** | **`requests`/TLS failure against FRED silently degrades to partial data.** | Medium | Reproducible on this machine — `requests` times out, `curl` succeeds on the identical URL. | All FRED access shells out to `curl` via `subprocess`; every download asserts non-empty response, expected column names, and expected first-obs date; any failure raises rather than returning a partial frame (**the same defensive posture `DataDownloader.download()` already takes** — it refuses to build a partial panel). |
| **R9** | **`HG=F` Volume field is garbage and quietly enters a feature.** | Low-Medium | Cached values of 404 and 242 contracts/day are front-month rollover artifacts, not real COMEX volume. | Volume is excluded by name in `features.py`; `test_features.py` asserts no feature column derives from `Volume`. |
| **R10** | **`pypbo` / third-party numerics are wrong or the package is unmaintained.** | Medium | `mlfinlab`'s availability/licensing is already flagged as uncertain; `pypbo` is a single-maintainer GitHub repo. | Pin the commit SHA. Hand-implement the DSR closed form once and assert agreement to 1e-6. If `pypbo` cannot be installed, PBO falls back to a hand-implemented CSCV (S=16) — the algorithm is ~60 lines and is unit-testable against a known-overfit synthetic strategy. |
| **R11** | **Overlapping targets (h=5, 22) inflate significance.** | Medium | Structural: 22-day targets overlap on 21 of 22 consecutive days. | HAC/Newey-West bandwidth h−1 everywhere; HLN small-sample correction; Student-t reference distribution; effective-sample-size caveat stated in the Limitations section. |
| **R12** | **Range-based RV (Garman-Klass) is a noisier target than intraday RV, weakening every T1 conclusion.** | Medium | Wang & Lu use high-frequency RV; free intraday copper history is unavailable. | Disclosed prominently in Limitations and in the pre-registration. Mitigated by (a) using GK rather than squared returns (~7× more efficient), (b) a pre-registered full Parkinson-based robustness re-run of T1, (c) reporting the two estimators' correlation. **Crucially, HAR and every baseline are evaluated on the identical target**, so relative rankings — which are the paper's actual claims — are unaffected by target noise; only absolute R² levels are. |
| **R13** | **Scope creep — "let's just add news sentiment / inventories / PatchTST."** | Medium | This project's own history of expanding scope mid-flight. | Hard feature freeze 2026-08-30 written into the pre-registration; PatchTST explicitly cut in §3.5; inventories explicitly cut in D6 with one timeboxed spike and a pre-decided failure outcome; all additions after the freeze go in the Amendment Log and are shown separately from the headline table. |

---

### Critical Files for Implementation

- `D:\copper\docs\preregistration_paper2.md` *(to be created — the Week-A2 gate that everything downstream depends on)*
- `D:\copper\src\cubench\features.py` *(to be created — Blocks A/B′/C/D per §2)*
- `D:\copper\src\cubench\walkforward.py` *(to be created — the 11-fold engine + ModelAdapter interface)*
- `D:\copper\src\cubench\pit.py` *(to be created — the point-in-time join; the single most important leakage control)*
- `D:\copper\src\utils.py` *(existing — `save_results`, `EarlyStopper`, `compute_metrics` reused; its `diebold_mariano_test` must be superseded by the HLN-corrected version in `src/cubench/stats_tests.py`)*
- `D:\copper\src\models\baselines.py` *(existing — `LSTMBaseline`/`TransformerBaseline` ported verbatim into `src/cubench/models/deep.py`)*
