# Correlation Feasibility Findings: Checking the Literature Review Against This Project's Own Data

**Date:** 2026-08-19
**Notebook:** `notebooks/correlation_feasibility_analysis.ipynb`
**Raw data cache:** `data/raw_correlation_check/`
**Full outputs:** `results/correlation_analysis/`

## Purpose

The two literature-review documents in this repo (`literature_review_copper_price_forecasting.md`
and `copper_fundamentals.md`) make specific claims about which macro/cross-asset variables move
with copper prices -- e.g. "DXY and copper exhibit a well-documented negative correlation,
typically -0.3 to -0.6"; gold and silver have "great"/"high" influence on copper; China PMI shows
"positive correlation... particularly at 1-3 month horizons"; VIX is negatively correlated with
copper. Every one of these claims is a citation of *someone else's* finding on *someone else's*
dataset and time period. None had been checked against data this project actually pulled.

This notebook pulls daily data 2010-01-01 to present for copper and ten candidate variables, and
runs six different tests (contemporaneous correlation, rolling correlation, lead-lag
cross-correlation, Granger causality, cointegration, and stationarity) against each one, then
cross-checks the results against what the literature documents claimed.

## What was tested and how

- **Data**: daily closes via `yfinance` for copper (HG=F), aluminum (ALI=F), gold (GC=F), silver
  (SI=F), crude oil (CL=F), DXY (DX-Y.NYB), S&P 500 (^GSPC), VIX (^VIX), 10Y Treasury yield
  (^TNX); plus FRED free-CSV-endpoint series for the 10Y TIPS real yield (DFII10) and ICE BofA
  High Yield OAS (BAMLH0A0HYM2).
- **China PMI**: the genuine Caixin/S&P Global and NBS China Manufacturing PMI series are **not
  obtainable free** -- Caixin PMI is licensed through S&P Global Market Intelligence
  (subscription only), and NBS's official PMI is published only as press-release numbers with no
  free bulk-download API. As a fallback we used FRED series `BSCICP03CNM665S` (OECD Business
  Tendency Survey confidence indicator for China manufacturing, monthly, back to 2000) as a
  proxy. This is *not* the same series the literature cites, and that distinction matters for
  interpreting the result below.
- **Statistics, all computed on log returns (or, for yield/spread series, simple differences --
  see note below) unless stated otherwise**: Pearson + Spearman contemporaneous correlation,
  252-trading-day rolling correlation, lead-lag cross-correlation (-10 to +10 trading days),
  Granger causality (lags 1, 5, 22 trading days), Engle-Granger cointegration on price *levels*,
  and ADF stationarity tests on both levels and returns.

## Headline finding: every candidate is "significant" but the significance is doing less work than it looks like

With ~4,000-4,200 daily observations, Pearson correlations as small as |r| ≈ 0.03 are
statistically significant at p<0.05 just from sample size. All 10 candidates showed a
"statistically significant" contemporaneous correlation with copper returns except the China-PMI
proxy and the real yield. That significance is not, by itself, evidence of a decision-useful
relationship -- what matters is magnitude, direction stability, and whether the relationship
gives copper a genuine *lead* time (a variable that moves with copper but not before it is not
useful as a predictive feature). Those are exactly the axes where several of the literature's
claims run into trouble.

## Finding 1: No candidate genuinely *leads* copper at daily frequency

For **9 of 11 candidates**, the lead-lag cross-correlation function peaks exactly at lag 0
(same-day). That means the strongest relationship between copper's return and every candidate's
return is contemporaneous, not predictive -- the two markets move together on the same day, but
knowing yesterday's (or last week's) DXY, gold, VIX, S&P 500, etc. move does not meaningfully
improve the correlation over knowing today's move at the same time. This directly qualifies the
literature review's framing of PMI-type variables as "leading indicators": that framing is about
monthly economic-release timing (a PMI print released today describes conditions that will show
up in copper demand over coming weeks), not about daily-return cross-correlation, which is what
this notebook tested. At daily frequency, none of the tested series showed a genuine, exploitable
lead over copper. This is consistent with markets being roughly informationally efficient at
daily granularity, and it's a real constraint CuBench's Block C feature engineering needs to
respect: a lagged daily macro feature is not obviously superior to a same-day one for any of
these variables.

## Finding 2: DXY -- the literature's most confident claim is the one that fails hardest under scrutiny

`copper_fundamentals.md` calls the DXY relationship "the most reliable macro feature" and cites a
"-0.3 to -0.6" range. The full-sample number we measured, r = -0.30, sits right at the *weak* end
of that claimed range -- already a softer relationship than the literature's framing suggests.
More importantly:

- **The 252-day rolling correlation ranges from -0.61 to +0.07 across 2010-present** -- meaning
  there were stretches where the DXY-copper relationship was near the claimed strong end, and
  other stretches where it was flat or even mildly positive. This is not a stable structural
  relationship; it is regime-dependent, and averaging across regimes is what produces the
  "-0.3 to -0.6" summary number in the first place.
- **Cointegration test on price levels: p = 0.97** -- copper and DXY show no evidence of a
  long-run equilibrium (cointegrating) relationship in levels over this sample, which is
  noteworthy because DXY is exactly the variable the literature review's VECM-based papers
  (Section 2.3) would predict *should* cointegrate with copper if the "reliable macro feature"
  claim extended to the levels domain.
- Granger causality *is* significant at all three lags (p < 0.003), so DXY does contain some
  incremental predictive information about copper's next-day/next-week/next-month return beyond
  copper's own history -- but this is a different, weaker claim than "reliable feature with a
  robust -0.3 to -0.6 correlation."

**Net: the literature's directional claim (negative) is correct, but its claims of magnitude
strength and stability are not well supported by this project's own data.**

## Finding 3: Gold and silver -- "great/high influence" is an overstatement

`literature_review_copper_price_forecasting.md` cites Chen et al. (2023) as finding gold and
silver to be the *most influential* features for copper, via feature-importance analysis. Our
direct correlation numbers: gold r = 0.32, silver r = 0.44. These are real, statistically robust
relationships (both p < 1e-90), but "high/great influence" is a strong characterization for
correlations in the 0.3-0.45 range -- roughly 10-20% of variance explained, similar in size to
DXY and S&P 500, not obviously larger. Gold's relationship is also regime-unstable (rolling
correlation ranges -0.11 to +0.63, crossing zero); silver's is more consistent in sign (rolling
min -0.02, essentially never negative) but still contemporaneous-only with no lead. Neither is
cointegrated with copper in levels (p = 0.14 and 0.19). This doesn't mean gold/silver are useless
features -- they are among the stronger contemporaneous correlates in this data -- but "great
influence" oversells what a feature-importance ranking in someone else's ML model (which can
reflect nonlinear or interaction effects, not just linear correlation) implies for a linear/
tree-based correlation-driven feature-selection exercise like CuBench's.

## Finding 4: VIX is the most literature-consistent relationship found

VIX shows a modest but *directionally stable* negative correlation with copper returns (r =
-0.26, p < 1e-63). Unlike DXY, gold, aluminum, crude oil, and the 10Y yield, VIX's 252-day rolling
correlation **never crosses zero across the entire 2010-2026 sample** (range: -0.48 to -0.02) --
it is consistently negative, just varying in strength. This is the one candidate where the
literature's qualitative claim ("VIX spikes coincide with copper sell-offs, negative correlation")
holds up cleanly against the full 16-year record, even though, like everything else tested, it
shows no genuine daily-frequency lead over copper.

## Finding 5: China PMI claim could not be confirmed -- and the reason matters

The literature review claims China PMI has "positive correlation... particularly at 1-3 month
horizons," and this project's own econometrics review previously flagged a reported breakdown of
that relationship to near-zero around 2022. We could not pull the genuine Caixin/NBS PMI series
for free, so we tested the OECD business-confidence proxy instead: **r = 0.010, p = 0.51 --
statistically indistinguishable from zero across the full sample**, and the rolling correlation
during the specific mid-2021-to-mid-2023 window flagged by the prior econometrics review averaged
-0.045 (slightly negative, not the claimed positive relationship at any point in that window).
Granger causality was also not significant at any lag (p > 0.5 throughout).

This is a genuinely disconfirming result for the literature's claim, but it comes with an
important caveat: this test used (a) a monthly-frequency *proxy* series, forward-filled onto a
daily calendar without publication-lag adjustment, and (b) an OECD confidence indicator, not the
actual Caixin/NBS PMI cited by the literature. Both of these could suppress a real relationship
that the genuine PMI series, tested properly at monthly frequency with correct lags, would show.
**Verdict: not confirmed with freely available data -- treat with real caution, but do not treat
this as definitive proof the literature's claim is wrong.** If China PMI is wanted as a Block C
feature, it needs a paid data source (S&P Global/Caixin) or a scraped/manually-updated NBS
press-release series, tested properly at monthly-release frequency with a realistic publication
lag, before it should be trusted.

## Finding 6: real yield and high-yield spread -- mixed, and one has a real data problem

- **10Y real yield (DFII10)**: essentially zero linear correlation with copper returns (r =
  -0.005, not significant) -- but Granger-causality tests *are* significant at all three lags (p
  < 0.003), meaning it carries some nonlinear or timing-related predictive information not
  captured by simple correlation. Worth including as a candidate feature and re-testing with a
  model that can pick up more than linear structure, rather than dropping on correlation grounds
  alone.
- **High-yield OAS (BAMLH0A0HYM2)**: shows a real contemporaneous negative correlation (r =
  -0.20, p < 1e-7), consistent with "credit stress coincides with commodity sell-offs." However,
  **the free `fredgraph.csv` endpoint for this series only returns data from 2023-08-21 onward
  (754 rows) despite the underlying series existing back to 1996-12-31 on FRED's own site** --
  explicit start-date parameters (`cosd=`) made no difference. This is a genuine limitation of
  the free CSV access method, not a bug in this analysis, and it means every statistic reported
  for this series is based on ~2 years of data, not the full 2010-present window. Any production
  use of this feature should pull the full history via the FRED API (requires a free API key,
  which was out of scope for the "no-API-key" constraint of this exploratory pass) rather than
  the CSV endpoint used here.

## Finding 7: no candidate is cointegrated with copper's price level

Every single Engle-Granger cointegration p-value came back well above 0.05 (lowest was aluminum
at p = 0.059, everything else p > 0.14, several p > 0.9). None of the candidate variables shows
evidence of a long-run equilibrium relationship with copper's price *level* over 2010-present.
This is worth flagging against literature_review's Section 2.3 (VAR/VECM/cointegration models),
which reports other papers finding cointegrating relationships between copper and various
macro/energy variables -- those findings may be sample-period-specific or specific to the exact
variable pairs and frequencies those papers used (e.g. monthly LME prices vs. our daily COMEX
futures continuous contract). This project's own data does not reproduce a cointegrating
relationship with any of the 11 variables tested here.

## Finding 8: stationarity check -- the standard practice is correctly justified

ADF tests confirm the standard assumption: every price/level series (copper, metals, equities,
yields) fails to reject the unit-root null on levels (p > 0.05, mostly p > 0.4), while every
return/diff series rejects it (p < 0.05, most p ≈ 0). This means the project's plan to build
Block A/B/D features from returns rather than raw levels is econometrically well-founded, and
levels should only be used deliberately for cointegration-style tests, not for correlation or
regression features.

## Summary verdict table

| Variable | Pearson r (returns) | p-value | Best lead-lag | Granger p (1/5/22d) | Cointegration p | Verdict |
|---|---|---|---|---|---|---|
| Aluminum | 0.41 | <0.001 | lag 0 (contemp.) | 0.86 / 1.00 / 0.16 | 0.06 | Correlated but regime-unstable (rolling corr -0.16 to 0.66, crosses zero) |
| Gold | 0.32 | <0.001 | lag 0 | 0.58 / 0.35 / 0.32 | 0.14 | Correlated but regime-unstable (-0.11 to 0.63) -- literature's "great influence" overstated |
| Silver | 0.44 | <0.001 | lag 0 | 0.08 / 0.49 / 0.32 | 0.19 | Contemporaneous only, no lead -- literature's "high influence" overstated |
| Crude oil | 0.26 | <0.001 | lag 0 | 0.16 / 0.63 / 0.45 | 0.95 | Correlated but regime-unstable (-0.16 to 0.63) |
| DXY | -0.30 | <0.001 | lag 0 | 1.4e-5 / 0.001 / 0.003 | 0.97 | Correlated but regime-unstable (-0.61 to +0.07) -- literature's flagship claim overstated on stability, not on direction |
| S&P 500 | 0.30 | <0.001 | lag 0 | 1.4e-5 / 6e-6 / 1.1e-4 | 0.58 | Contemporaneous only, no lead |
| VIX | -0.26 | <0.001 | lag 0 | 6.7e-5 / 4.1e-5 / 0.019 | 0.97 | Contemporaneous only, no lead -- but the most regime-stable relationship found (always negative) |
| 10Y yield (nominal) | 0.12 | <0.001 | lag 0 | 0.52 / 0.09 / 0.03 | 0.74 | Correlated but regime-unstable (-0.15 to 0.47) |
| 10Y real yield (DFII10) | -0.01 | 0.74 (n.s.) | lag 1 | 0.003 / 0.002 / 0.002 | 0.85 | Weak/marginal linear correlation, but genuine Granger-causal signal -- worth testing in a non-linear model |
| High-yield OAS | -0.20 | <0.001 | lag 0 | 0.014 / 0.13 / 0.28 | 0.73 | Contemporaneous only, no lead; **data quality caveat: only 754 obs from 2023-08 due to FRED CSV endpoint truncation** |
| China PMI (OECD proxy, NOT genuine Caixin/NBS) | 0.01 | 0.51 (n.s.) | lag 4 (weak) | 0.51 / 0.53 / 0.73 | 0.96 | Weak/marginal, not decision-useful with this proxy -- literature's claim not confirmed but genuine series untested |

## Recommendation for CuBench Block C (macro features)

1. **Keep, with the stability caveat documented**: DXY, VIX. Both show real, literature-consistent
   directional relationships. VIX is the more stable of the two across the full sample and should
   probably be weighted more confidently; DXY should be used but the model (and anyone reading
   feature-importance output later) should not assume the -0.3 to -0.6 magnitude the literature
   cites is a fixed constant -- it swings by regime.
2. **Keep with caution, re-test in a non-linear model**: 10Y real yield (DFII10). Zero linear
   correlation but consistently significant Granger causality across all three horizons is an
   unusual combination worth investigating further rather than dropping outright -- LightGBM's
   tree splits may pick up structure a Pearson r cannot.
3. **Downgrade expectations, don't drop**: Gold, silver, S&P 500, aluminum, crude oil, nominal 10Y
   yield. All show real but modest (r 0.12-0.44), non-leading, and (except silver) regime-unstable
   contemporaneous correlations. They are plausible Block C/cross-market features but should not
   be marketed internally as "high-influence" drivers on the strength of the cited literature --
   the literature's characterization is not reproduced at this magnitude in this data.
4. **Fix before using**: High-yield OAS (BAMLH0A0HYM2). The relationship looks real but the free
   CSV endpoint only yielded ~2 years of data; get a full 1996-present pull via the FRED API (free
   API key) before trusting any statistic derived from it, including the ones in this report.
5. **Cannot currently validate China PMI as specified in the literature.** The real Caixin/NBS
   series is not free. The free OECD proxy shows no relationship at all with copper returns,
   directly contradicting the literature's "positive correlation, particularly at 1-3 month
   horizons" claim -- but because the proxy series, the daily-frequency test design, and the
   naive forward-fill (with its look-ahead-bias risk, see below) are all imperfect stand-ins for
   what the literature actually tested, this should be read as "not confirmed," not "definitively
   false." If China PMI stays in the CuBench plan, it needs either a paid data source or a
   manually-curated NBS press-release series and a monthly-frequency (not daily-ffill) test
   before being trusted.
6. **No candidate is safe to use as a raw-level regressor via a cointegration/VECM-style approach**
   -- none showed cointegration with copper's price level in this sample, despite that being a
   documented finding for some of these variable pairs in other papers' data.

## Methodology notes and caveats

- **Look-ahead bias in forward-fill**: monthly-frequency series (China PMI proxy, and had it
  loaded with full history, the high-yield OAS series) were forward-filled onto copper's daily
  trading calendar. This assumes each monthly value is known and usable starting the day it is
  timestamped, when in reality PMI-type releases lag their reference month by anywhere from a few
  business days to several weeks, and OECD confidence indicators lag considerably more due to
  survey compilation and revision cycles. **This notebook does not correct for that lag** -- it
  is flagged here as a risk that must be fixed (with an explicit publication-lag shift) before any
  of these forward-filled features are used in an actual forecasting pipeline, since using them
  as-is risks leaking information the model would not have had access to on that date.
- **Yield/spread series and log returns**: DFII10 (real yield) goes negative for stretches of
  2020-2021, so log returns are undefined for those days. Nominal 10Y yield, real yield, and the
  high-yield spread were therefore transformed via simple differences (basis-point changes)
  rather than log returns; all price-type series (copper, metals, equities, DXY, VIX) used log
  returns as originally planned.
- **Statistical significance vs. practical significance**: with ~4,000+ daily observations,
  p-values below 0.05 are easy to achieve for economically small effect sizes. Every correlation
  and Granger p-value in this report should be read alongside the effect size (r, or the rolling
  correlation range), not the p-value alone.
- **Reproducibility**: raw data is cached in `data/raw_correlation_check/` so the notebook can be
  re-run without re-hitting Yahoo Finance or FRED. The `requests` library reliably times out
  against `fred.stlouisfed.org` on the machine this was run on (a TLS/schannel issue) even though
  `curl` succeeds against the same URL; the notebook's FRED downloader shells out to `curl`
  instead of using `requests` for this reason.
