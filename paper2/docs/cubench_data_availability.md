# CuBench Data Availability — Week A1 Findings

**Date:** 2026-08-19
**Status:** Week A1 (Aug 19–23) re-verification spikes S1–S4, complete. This document is
self-contained — it does not assume the reader has seen `docs/cubench_implementation_plan.md`,
though that document is the authoritative technical specification this work follows.

**Scope note:** `docs/archive/vmd-mfgnn-protocol-SKILL.md` (the archived Paper 1 data-scope
lock) governs **Paper 1 only**. It states "FRED and BDI are explicitly out of scope" and
locks scope to 8 yfinance tickers; CuBench (this project) deliberately and explicitly breaks
that lock — it adds 6 FRED series, adds silver, and adds full OHLC for copper. Do not read
that archived file as current guidance for CuBench.

---

## Summary table — what is confirmed working

| Source | Series | Coverage (verified 2026-08-19) | Status |
|---|---|---|---|
| yfinance | `HG=F` (copper, full OHLC) | 2010-01-04 → 2025-12-30, **4,023 rows**, no null O/H/L/C | **Confirmed.** Volume excluded (rollover artifact, not real volume). |
| yfinance | `GC=F` (gold) | 2010-01-04 → 2025-12-30, 4,022 rows | Confirmed |
| yfinance | `SI=F` (silver) | 2010-01-04 → 2025-12-30, 4,022 rows | Confirmed |
| yfinance | `CL=F` (crude oil) | 2010-01-04 → 2025-12-30, 4,023 rows | Confirmed |
| yfinance | `^GSPC` (S&P 500) | 2010-01-04 → 2025-12-30, 4,023 rows | Confirmed |
| yfinance | `^TNX` (US 10Y yield) | 2010-01-04 → 2025-12-30, 4,021 rows | Confirmed; differenced in bp only, never log-returned |
| yfinance | `DX-Y.NYB` (Dollar Index) | 2010-01-04 → 2025-12-30, 4,024 rows | Confirmed |
| yfinance | `^VIX` | 2010-01-04 → 2025-12-30, 4,023 rows | Confirmed |
| yfinance | `ALI=F` (aluminum) | **2014-05-06 → 2025-12-30, 2,895 rows** | **Confirmed, but coverage differs from the plan's claim — see Discrepancy 1 below.** |
| yfinance | `CLP=X` (Chilean peso) | 2010-01-01 → 2025-12-30, 4,164 rows, 0.22% gap | **Passes the pre-decided inclusion rule (≥3,800 obs, <2% gaps). Included.** |
| yfinance | `QC=F` (E-mini copper) | 2010-01-04 → 2025-12-30, 4,024 rows, 3.55% gap | **Ambiguous against the literal rule — see Discrepancy 2 below. Judgment call: include, pending Week A2 confirmation.** |
| FRED (curl) | `DFII10` (10Y TIPS real yield) | first obs 2009-01-02 (expected ≤ 2010-01-05) | Confirmed, passes coverage assertion |
| FRED (curl) | `BAA10Y` (Baa credit spread) | first obs 2009-01-02 (expected ≤ 2010-01-05) | Confirmed, passes coverage assertion |
| FRED (curl) | `PPIACO` (PPI, all commodities) | first obs 2009-01-01, 204 monthly obs | Confirmed, passes coverage assertion |
| FRED (curl) | `INDPRO` (industrial production) | first obs 2009-01-01, 204 monthly obs | Confirmed, passes coverage assertion |
| FRED (curl) | `BAMLH0A0HYM2` (HY OAS) | 2023-08-21 → 2025-12-31, 620 obs | **Confirmed truncated as predicted. Excluded from headline (per D4); robustness-appendix only.** |
| FRED (curl) | `BSCICP03CNM665S` (China PMI proxy) | 2010-2025, 181 obs | Confirmed, ablation-only per D7 |
| — | COMEX contract-month ladder (192 tickers) | **1 of 192 tickers usable** | **Tier 3 confirmed — see Block B section below.** |

---

## S1 — `HG=F` full OHLC pull

Pulled directly from yfinance (live network call, not the cache) for 2010-01-04 →
2025-12-31: **4,023 rows**, all Open/High/Low/Close non-null. This exceeds the plan's
≥4,000-row assertion. **The plan's stated fallback (`data/raw_correlation_check/yf_copper.csv`)
was not needed** — but for the record, that cached file was also checked and does in
fact contain full OHLC (not close-only), consistent with the plan's claim. Both the
live pull and the cache agree on full OHLC availability.

**Verdict: PASS, as predicted.**

## S2 — COMEX contract-month coverage audit (Block B)

All 192 tickers of the form `HG{F,G,H,J,K,M,N,Q,U,V,X,Z}{10..25}.CMX` were probed via
the Yahoo Finance chart HTTP endpoint directly. Results:

- **192 tickers probed, 21 returned HTTP 200, only 1 (`HGZ25.CMX`) returned any usable
  price data** (2020-09-29 → 2025-12-29, 1,321 observations). The other 20 "HTTP 200"
  tickers returned a well-formed response with an empty/all-null close series (delisted
  or never-listed contract codes that Yahoo still resolves without erroring).
- The remaining 171 tickers returned HTTP 404.
- **Promotion rule** (promote Block B to Tier 1 only if ≥2 simultaneously-live contract
  months exist on ≥60% of trading days across 2010–2025): trivially fails, since fewer
  than 2 tickers have any data at all. `frac_days_with_ge2_live_contracts = 0.0`.

Full results: `results/cubench/data_audit/comex_contract_month_coverage.csv` (all 192
rows). Summary: `results/cubench/data_audit/comex_contract_month_coverage_summary.json`.

**Verdict: Tier 3 CONFIRMED — and more decisively than the plan's own two prior spot-checks
suggested.** The plan predicted this outcome based on two individual ticker checks
(`HGZ25.CMX` live, `HGZ15.CMX` 404). The full systematic audit not only confirms the
prediction but shows the problem is more total than even those two spot-checks implied:
of 192 candidate tickers, literally one is usable, and it only covers a single 2020–2025
contract. **Block B (true carry/curve) remains omitted from the headline model, exactly
as the plan's D3 decision specifies.** The `B′` curve-proxy block (ablation-only, never
called carry) is unaffected by this finding since it doesn't depend on the contract ladder.

## S3 — FRED pulls (`DFII10`, `BAA10Y`, `PPIACO`, `INDPRO`)

All four series pulled successfully via `curl` (not `requests` — see Environment Hazards
below) and passed their first-observation-date assertions:

| Series | Expected first obs (plan) | Actual first obs | Pass? |
|---|---|---|---|
| `DFII10` | ≤ 2010-01-05 | 2009-01-02 | Yes |
| `BAA10Y` | ≤ 2010-01-05 | 2009-01-02 | Yes |
| `PPIACO` | ≤ 2009-01-01 | 2009-01-01 | Yes |
| `INDPRO` | ≤ 2009-01-01 | 2009-01-01 | Yes |

`BAA10Y` is **not** truncated — no fallback (`STLFSI4` or dropping the credit channel)
is needed. `BAA10Y` was itself already the plan's D4 replacement for the previously-
truncated `BAMLH0A0HYM2`.

**`BAMLH0A0HYM2` (HY OAS) was re-checked directly and is confirmed truncated**, starting
2023-08-21 (620 observations through 2025-12-31). This matches the plan's account of the
April 2026 licensing truncation (previously reported as 754 obs by an earlier check that
used a later end-date — not a *new* truncation since the earlier check; the start date
2023-08-21 is identical in both checks, only the pull windows differ). Per the plan's D4
decision, `BAMLH0A0HYM2` is **excluded from the headline model** and retained only for a
2023–2025 robustness appendix.

`STLFSI4` (the pre-decided fallback) was also pulled successfully as a precaution (weekly,
back to 2009) but is **not needed** since `BAA10Y` itself is not truncated.

**Verdict: PASS, as predicted**, including the pre-registered fallback logic firing
correctly on the (expected) HY OAS truncation.

## S4 — `CLP=X` and `QC=F` coverage checks

- **`CLP=X` (Chilean peso, USD/CLP)**: 4,164 daily observations, 2010-01-01 →
  2025-12-30, gap fraction 0.22% against a pure business-day calendar. **Passes the
  pre-decided rule (≥3,800 obs, <2% gaps) cleanly. Included.**
- **`QC=F` (E-mini copper)**: 4,024 daily observations, 2010-01-04 → 2025-12-30, gap
  fraction 3.55% against a pure business-day calendar. **This exceeds the literal <2%
  gap threshold** — see Discrepancy 2 below for the resolution.

**Verdict: PASS for `CLP=X` as predicted (rule passes cleanly); `QC=F` required a
judgment call the plan did not fully specify — resolved below, flagged for Week A2 review.**

---

## Discrepancies between the plan's predictions and what was actually found

### Discrepancy 1 — aluminum (`ALI=F`) does not have 2010 coverage

The plan's Section 1.3 states all Block C-cross series (including aluminum) are "already
cached under `data/raw_correlation_check/`, all verified 2010→2026, all ~4,000–4,200 obs."
**This is not accurate for aluminum.** Both a fresh live pull and the existing cached file
(`data/raw_correlation_check/yf_aluminum.csv`) show `ALI=F` beginning **2014-05-06**, not
2010, with **2,895 observations**, not ~4,000–4,200. All seven other Block C-cross/macro
yfinance tickers (gold, silver, crude oil, S&P 500, US 10Y yield, DXY, VIX) were individually
re-verified and do match the plan's claimed 2010-01-04 start and ~4,000+ row count exactly.

**Resolution:** aluminum is still included (per D-series decisions, aluminum was always a
planned Include with r=+0.41), but any feature built from it will have ~1,150 fewer daily
observations than the other Block C-cross series and will be NaN for 2010-01-04 →
2014-05-05. This must be handled explicitly in `features.py`/`folds.py` (Week A2/S1) —
either the initial training block (2010–2014) trains without an aluminum feature, or
aluminum-derived features are NaN-filled with a flag for that period and the model
naturally down-weights them via LightGBM's native NaN handling. This should be decided
during Week A2 feature-freeze discussion, not silently defaulted.

### Discrepancy 2 — `QC=F`'s gap fraction fails the literal `CLP=X` threshold, but the threshold does not clearly transfer

The plan's Section 1.4 gives an explicit numeric rule for `CLP=X` (≥3,800 obs, <2% gaps)
but for `QC=F` (Section 1.5) only says "if `QC=F` (E-mini copper) passes a Week-1 coverage
check" without stating the same numeric thresholds apply. Applying the `CLP=X` thresholds
literally, `QC=F` fails on gap fraction (3.55% vs. the 2% bar).

However, this appears to be an artifact of comparing an FX series (which trades on many
exchange holidays) against a pure Mon–Fri business-day calendar, rather than a genuine
data-quality problem with `QC=F`. As a sanity check, `HG=F` itself — the plan's own
primary, verified-good copper series — has an almost identical gap fraction (**3.57%**)
against the same business-day calendar, because both `HG=F` and `QC=F` share COMEX's
holiday calendar (Good Friday, Memorial Day, Independence Day, etc.), which `CLP=X`'s FX
market mostly trades through.

**Resolution (judgment call, flagged for Week A2 confirmation, not silently decided):**
`QC=F` is judged to have adequate, "same-quality-as-copper-itself" coverage and is
tentatively **included** as the basis for `b4_hg_qc_spread`, on the reasoning that
benchmarking it against copper's own gap rate is the more apples-to-apples test than the
FX-calibrated <2% rule. The raw `QC=F` data has been pulled and snapshotted regardless
(`data/cubench/raw/yf_qc_emini_copper.csv`) so this decision can be revisited without a
re-pull. **This should be explicitly confirmed (not just inherited) when `features.py` is
built in Week A2** — it is a genuine interpretation gap in the plan, not a pre-decided rule.

### Discrepancy 3 (packaging, not data) — `pypbo` is not pip-installable from GitHub

Not a data-availability finding, but recorded here for completeness since it was found
during this same Week A1 pass: `pip install git+https://github.com/esvhd/pypbo.git@<SHA>`
fails — the pinned commit (`4d723f06498267a2a6280cb9d7d5649348e961d1`) has no
`setup.py`/`pyproject.toml` at all. Resolved by copying the pinned commit's `pypbo/`
source directory directly into `.venv_corr/Lib/site-packages/pypbo` (still pinned to the
exact commit; reproduction steps recorded in `requirements_cubench.txt`). This confirms
Risk R10 in the implementation plan (pypbo packaging risk) as real rather than
hypothetical.

---

## Environment hazards carried forward (confirmed again in Week A1)

- **`requests` reliably times out against `fred.stlouisfed.org`** on this machine
  (TLS/schannel issue); `curl` succeeds against the identical URL. All FRED access in
  `src/cubench/data.py` shells out to `curl` via `subprocess`. Confirmed again during
  the S3 spike (all 4+2 FRED pulls succeeded via curl).
- **`requests` works fine against Yahoo Finance's endpoints** (`query1.finance.yahoo.com`)
  — confirmed directly during the S2 audit (192 HTTP requests, no TLS failures). The
  FRED-specific hazard does not extend to Yahoo.

---

## Block B (true carry/curve): confirmed Tier 3

See the S2 section above for the full audit result. **Decision, per the plan's D3 (unchanged
by this audit — the audit confirms rather than revises the pre-committed decision):** Block
B (true futures term-structure/carry) is omitted from the headline model. The `B′`
curve-proxy block (4 features, ablation-only, never described as basis/carry:
`b1_rv_term_ratio`, `b2_cu_al_relvalue`, `b3_cu_al_mom_spread`, and conditionally
`b4_hg_qc_spread`) is unaffected and will be built in Week A2's `features.py`.

## What's now pulled and snapshotted

`data/cubench/raw/` contains 17 CSV files (9 yfinance OHLC/Close series + 2 conditional
yfinance series + 6 FRED series) plus `data_source_log.json` recording pull timestamp,
row counts, and date ranges for every series, in the same JSON convention as
`data/raw_correlation_check/data_source_log.json`. Per Risk R2's mitigation, this
snapshot — not the live network — becomes the authoritative data source for all
downstream Week A2+ work.
