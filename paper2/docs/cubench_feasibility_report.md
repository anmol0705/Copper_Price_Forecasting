# CuBench Research and Feasibility Report

Access date: 2026-08-19. Primary literature search used the arXiv MCP (`search_papers` and `export_citations`) for q-fin/cs.LG/stat.ML coverage, then provider and venue pages were checked with live web access for current data mechanics and deadlines.

## 1. Executive Summary

CuBench is feasible as designed if the paper is framed as a rigor-first volatility and distributional forecasting benchmark for copper, not as a high-accuracy return-direction paper. The arXiv MCP search found direct 2024 copper-volatility literature: Wang and Lu benchmark COMEX copper realized-volatility forecasting with GARCH, HAR, RNN, LSTM, and GRU, and report that HAR has the lowest QLIKE for daily realized volatility overall. Wang and Li separately use GARCH-MIDAS/DCC-MIDAS for COMEX copper and find macro variables, especially PPI, useful for long-term volatility/correlation structure. These papers validate the volatility-primary target and establish HAR/GARCH as mandatory baselines.

No source found in this pass appears to have already done the exact CuBench combination: LightGBM/XGBoost/CatBoost plus trend, carry/curve, macro, and volatility/regime features, with primary realized-volatility and quantile targets, on copper specifically, under expanding walk-forward, multiple-testing correction, transaction costs, and base-rate diagnostics. The contribution remains non-duplicative, but it should be positioned as a rigorous tabular-ML benchmark and ablation study, not as a novel model architecture.

The biggest risk is data, specifically Block B. Yahoo Finance/yfinance works for `HG=F` continuous copper history back to 2010 in a direct endpoint test, but named contract-month tickers are not reliable for building a 2010-2025 futures curve. `HGZ25.CMX` only returned observations from 2020-09-29 onward; expired examples such as `HGZ15.CMX` returned 404. CME has official continuous copper series starting 2010, but access is through licensed CME channels/DataMine, not a simple free API. If true free contract-month ladders cannot be obtained, Block B should become optional or be replaced by a weaker proxy with a transparent limitation statement.

ICAIF 2026 is no longer viable for a November 2026 capstone start-to-finish cycle because its full-paper deadline was August 9, 2026. The realistic route is arXiv/SSRN first, then ICAIF 2027 or a journal/letter submission after the capstone.

## 2. Section A findings - methodology validation

### Verdict

The methodology is defensible with three changes:

1. Treat HAR and GARCH-family models as strong baselines, not strawmen. The closest copper-volatility arXiv paper reports HAR as the strongest daily RV model overall.
2. Make realized volatility and quantile loss the primary scoreboard. Recent arXiv work supports probabilistic volatility forecasting, and several papers warn that raw directional accuracy is fragile or misleading.
3. Make carry/curve an ablation conditional on real futures-curve data. Do not silently substitute continuous `HG=F` for a genuine term-structure feature.

### Most relevant papers found

| Citation | Year | Venue/source | Dataset/period | Method | Key reported metric | Relevance/risk note |
|---|---:|---|---|---|---|---|
| Wang and Lu, "COMEX Copper Futures Volatility Forecasting: Econometric Models and Deep Learning" ([arXiv:2409.08356](https://arxiv.org/abs/2409.08356)) | 2024 | arXiv, found via MCP | COMEX copper futures realized volatility; daily and hourly high-frequency RV; exact sample period not in accessible abstract | GARCH, HAR, RNN, LSTM, GRU; rolling-window forecasting | Abstract reports HAR achieves the lowest QLIKE for daily RV overall; exact QLIKE number not in accessible abstract | Directly overlaps CuBench target. Requires CuBench to include HAR as a serious baseline. |
| Wang and Li, "On the macroeconomic fundamentals..." ([arXiv:2409.08355](https://arxiv.org/abs/2409.08355)) | 2024 | arXiv, found via MCP | COMEX copper futures high-frequency returns plus macro variables; exact sample period not in accessible abstract | GARCH-MIDAS and DCC-MIDAS | Abstract reports RV, interest rates, industrial production, PPI, slope volatility, consumer sentiment, and dollar index have significant effects; PPI is described as most efficient | Strong support for macro/regime block. Also suggests adding PPI/IP as robustness alternatives to the planned small macro set. |
| Dudek, Kasprzyk, and Pelka, "Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting" ([arXiv:2511.20105](https://arxiv.org/abs/2511.20105)) | 2025 | arXiv, found via MCP | Bitcoin realized volatility, 69 predictors | LightGBM deterministic and probabilistic forecasts; quantile regression and residual simulation | Exact numeric metrics not in accessible abstract | Strong methodological analogue for LightGBM + probabilistic volatility + feature importance, but not commodity/copper. |
| Dudek, Orzeszko, and Fiszeder, "Probabilistic Forecasting Cryptocurrencies Volatility" ([arXiv:2508.15922](https://arxiv.org/abs/2508.15922)) | 2025 | arXiv, found via MCP | Bitcoin realized variance | HAR, GARCH, ARFIMA, LASSO, SVR, MLP, RF, LSTM base forecasts feeding quantile methods | Abstract reports QRS with linear base models on log RV consistently outperforms more sophisticated alternatives | Supports probabilistic volatility scoring and simple baselines. Risk: advanced ML may not dominate. |
| Fang and Slepaczuk, "Volatility Forecasting and Return Prediction under Market Regimes" ([arXiv:2606.09478](https://arxiv.org/abs/2606.09478)) | 2026 | arXiv, found via MCP | High-frequency CSI 300 Index, 2005-2023 | HARQ, Markov-switching GJR-GARCH, XGBoost return prediction, walk-forward OOS | Exact numeric metrics not in accessible abstract; abstract says return predictability is weak/state-dependent and naive strategies fail after costs | Strong support for regime-aware volatility first, return direction second, and transaction-cost-adjusted claims. |
| Brini, "Forecasting Realized Volatility with Time Series Foundation Models" ([arXiv:2607.05291](https://arxiv.org/abs/2607.05291)) | 2026 | arXiv, found via MCP | VOLARE dataset, 50 assets across equities, FX, and futures, 3 horizons | 9 zero-shot TSFMs vs 8 econometric models including HAR family | Abstract reports only Tiny Time Mixers beats Log-HAR at every horizon, narrowly; equal-weight TTM + Log-HAR enters MCS for 98-100% of assets | Reinforces that Log-HAR is hard to beat and should be included. |
| Cheung, "When Directional Accuracy Lies" ([arXiv:2607.12248](https://arxiv.org/abs/2607.12248)) | 2026 | arXiv, found via MCP | NASDAQ-100 and S&P 500 equity universes; expanding walk-forward folds | LoRA-adapted TimesFM with always-up, random-walk, persistence, AR(1), McNemar, DM, BH-FDR | Abstract reports an apparent roughly 80% DA is explained by a base rate of about 0.70; pooled LoRA has no excess directional skill | Direct prior art for the planned base-rate/constant-forecast diagnostic. CuBench should cite this and not overclaim novelty. |
| Portnaya, "The Bounce Has No Direction" ([arXiv:2606.29591](https://arxiv.org/abs/2606.29591)) | 2026 | arXiv, found via MCP | Six US instruments, 1993-2026; 21-instrument cross-asset panel | Fourier-Residue Identity decomposing autocorrelation into sign and magnitude channels; VR interpretation; GARCH-size Monte Carlo | Abstract reports SPY lag-1 autocorrelation -0.081, z=-7.4; sign test p=0.11; full test p<1e-12; commodities/FX/crypto indistinguishable from random walks | Strong disconfirming evidence for direction-first claims; supports magnitude/volatility framing. |
| Muhammad et al., "A Benchmark of Classical and Deep Learning Models..." ([arXiv:2604.06227](https://arxiv.org/abs/2604.06227)) | 2026 | arXiv, found via MCP | AgriPriceBD, 1,779 daily retail mid-prices, July 2020-June 2025 | Persistence, SARIMA, Prophet, BiLSTM, Transformer, Time2Vec Transformer, Informer; DM tests | Abstract reports Time2Vec causes +146.1% MAE on green chilli, p<0.001; Informer variance up to 50x ground truth | Commodity benchmark warning: naive persistence may dominate near-random-walk commodities; deep models can degrade. |
| Lee et al., "Metal Price Spike Prediction via a Neurosymbolic Ensemble Approach" ([arXiv:2410.12785](https://arxiv.org/abs/2410.12785)) | 2024 | arXiv, found via MCP | Critical metals including cobalt, copper, magnesium, nickel; exact period not in accessible abstract | Neural ensemble plus symbolic error correction | Abstract reports up to 6.42% precision improvement, 29.41% recall improvement, 13.24% F1 improvement | Adjacent metal-price task, but spike detection rather than RV/quantiles. Useful as related metals-ML literature. |
| Amorin, Python, and Weisser, "Not All News Is Equal..." ([arXiv:2603.09085](https://arxiv.org/abs/2603.09085)) | 2026 | arXiv, found via MCP | Shanghai Metal Exchange aluminum, 2007-2024 | Finetuned LLM sentiment plus tabular data; LSTM; long-short simulations | Abstract reports high-volatility Sharpe 1.04 with Qwen3 sentiment vs 0.23 tabular baseline | Shows industrial-metal forecastability may be regime-dependent; not a copper duplicate. |
| Angelidis, Sakkas, and Tessaromatis, "Predicting commodity returns..." ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5205084), [Journal of Commodity Markets listing](https://ideas.repec.org/a/eee/jocoma/v38y2025ics2405851325000194.html)) | 2025 | SSRN / Journal of Commodity Markets | Commodity futures; exact asset list/period needs full paper check | Cross-sectional commodity momentum, basis, and basis-momentum factors vs historical-average and time-series models | Abstract says cross-sectional models produce superior time-series and cross-sectional forecasts; exact metrics not in accessible abstract | Important current factor evidence. CuBench should cite it and distinguish single-asset copper volatility from cross-sectional commodity-return prediction. |
| Bakshi, Gao, and Rossi, "Understanding the Sources of Risk..." ([Management Science/INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.2017.2840), [RePEc](https://ideas.repec.org/a/inm/ormnsc/v65y2019i2p619-641.html)) | 2019 | Management Science | Cross-section of commodity futures | Average commodity, carry, and momentum factors | Source abstract reports one- and two-factor models are rejected; average + carry + momentum describes cross-sectional variation | Supports carry/momentum, but cross-sectional factor evidence does not guarantee single copper daily predictive signal. |
| Yang, "Investment Shocks and the Commodity Basis Spread" ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X13001360)) | 2013 | Journal of Financial Economics | Cross-sectional commodity futures returns | Commodity basis/carry theory and empirics | Exact metrics not accessible from abstract in this pass | Foundational support for basis/carry. CuBench should treat it as motivation, not as proof of free single-asset predictability. |
| He et al., "Multi-Factor Function-on-Function Regression..." ([arXiv:2412.05889](https://arxiv.org/abs/2412.05889)) | 2024 | arXiv, found via MCP | WTI futures term structure and US Treasury yields | State-space functional regression vs Schwartz-Smith two-factor model | Exact numeric metrics not in accessible abstract; abstract reports superior short-end futures-curve estimation | Useful term-structure precedent; oil, not copper. |
| Matsubara, "Wasserstein Gradient Boosting" ([arXiv:2405.09536](https://arxiv.org/abs/2405.09536)) | 2024 | arXiv, found via MCP | General distribution-valued supervised learning | Wasserstein gradient boosting; tree-based evidential learning | Exact numeric metrics not in accessible abstract | Methodological reference for distribution-valued boosting, but likely too broad for capstone implementation. |

### Duplication check

No exact duplicate was found. The closest duplicate is Wang and Lu (2024), because it is COMEX copper realized-volatility forecasting and compares GARCH/HAR with deep recurrent models. CuBench differs by using tabular gradient boosting, feature blocks, quantile targets, feature ablations, base-rate diagnostics, transaction-cost checks, and formal multiple-testing correction.

### Disconfirming evidence and risks

- Copper volatility may be forecastable, but direction likely has a low ceiling. Portnaya (2026), Cheung (2026), and Muhammad et al. (2026) all support caution against raw direction or point-price metrics.
- HAR/Log-HAR may be difficult to beat. Wang and Lu (2024) and Brini (2026) both make this clear.
- Carry/momentum factors are best established cross-sectionally across commodities. Applying them to one asset, one daily copper series, is a weaker claim.
- Macro variables can become leakage-prone if publication lags are ignored. PMI, FRED, and macro release dates must be joined point-in-time, not by observation month alone.

## 3. Section B findings - data feasibility

| Data item | Verdict | Evidence | Recommendation |
|---|---|---|---|
| Daily copper continuous price, `HG=F` | Yes for price/RV; no for curve | Direct Yahoo chart endpoint test returned `HG=F` observations from 2010-01-04 to 2025-12-30, count 4029. Yahoo page also exposes Copper Futures historical data for `HG=F`. | Use for returns, realized volatility proxies, technical features, and baseline price/RV targets. Document that it is a continuous/front contract feed. |
| Contract-month COMEX copper data from Yahoo/yfinance | Partial/no for 2010 curve | Direct endpoint test: `HGZ25.CMX` returned first observation 2020-09-29 and count 1325; `HGZ15.CMX` returned 404. Search results show some named pages exist, but not reliable expired-month history. | Do not rely on Yahoo for full 2010-2025 front/deferred carry. Use only after automated audit of each month ticker and coverage. |
| CME official continuous copper series | Partial; official but licensed | CME Continuous Price Series lists Copper front ticker `HGCP1`, active ticker `HGCPA`, start date 04-Jan-2010, and says data are official settlement prices. Access requires DataMine/MDP Globex and licensing/ILA. | Best official fallback for continuous front/active price. It still may not provide the deferred contract ladder needed for a true carry spread without additional CME data products. |
| Investing.com copper futures historical | Partial | Investing.com exposes free historical continuous Copper Futures data in browser. API/reproducibility and contract-month depth are not verified. | Acceptable manual fallback for spot checks, not ideal for reproducible academic pipeline. |
| Barchart / MarketWatch / Google Finance futures chain | Partial | Current and some contract pages exist, but historical depth and automated access are not reliable/free from the checked pages. | Use for validation only unless a paid API or stable export is obtained. |
| Nasdaq Data Link / Quandl continuous futures | UNVERIFIED/likely paid | Search located Nasdaq Data Link futures products, but free copper contract ladder coverage was not verified. | Manual account check needed. If paid access is allowed, evaluate CHRIS/continuous futures datasets and licensing. |
| FRED `DFII10` 10-year TIPS real yield | Yes | FRED page exists and is updated; series is daily, percent, not seasonally adjusted. | Use FRED CSV/API. No major blocker. |
| FRED `BAMLH0A0HYM2` high-yield OAS | Partial for 2010-2025 | FRED page exists and was updated Aug 18, 2026, but its notes state that starting April 2026 the series includes only 3 years of observations on FRED. | For long 2010-2025 backtest, do not assume full FRED history is downloadable now. Use ICE source, institutional access, saved historical FRED snapshots/ALFRED if available, or replace with another freely available credit spread proxy. |
| China official NBS manufacturing PMI | Yes, monthly; point-in-time handling needed | NBS English press releases publish monthly PMI values, e.g. June 2026 manufacturing PMI 50.3. ChinaData.live exposes NBS-sourced API coverage from 2005-01 to 2026-07, but it is third-party. | Prefer NBS releases for audit trail; use third-party API only if license and revision behavior are acceptable. Lag PMI to release date. |
| Caixin/RatingDog manufacturing PMI | Partial | Caixin/S&P Global pages describe the PMI and current reports, but historical free bulk access appears less straightforward. TradingEconomics/Investing show values but may have access limits. | Use official NBS PMI as primary. Treat Caixin as optional unless a stable licensed/free history is obtained. |
| VIX daily data | Yes | FRED `VIXCLS` page covers 1990-01-02 through current dates. Cboe official historical page links VIX Index data for 1990 to present, updated daily, and includes caveats that accuracy is not guaranteed. | Use FRED `VIXCLS` for convenience; reconcile with Cboe CSV in validation. Expect non-trading-day/missing-day alignment issues. |

### Block B recommendation

Set Block B to one of three tiers:

1. Tier 1, preferred: licensed contract-month settlement data, enough to compute front vs deferred slope using a documented roll rule.
2. Tier 2: CME official continuous front and active series only, reported as a price-continuation/roll proxy, not true carry.
3. Tier 3: omit Block B from headline model and report it as a feasibility limitation.

## 4. Section C findings - venue/timeline feasibility

### ICAIF

ICAIF 2026 is already scheduled and its paper deadline has passed for a project beginning in August 2026. The official call lists:

- Paper submission deadline: August 2, 2026, extended to August 9, 2026.
- Author notification: September 27, 2026.
- Main conference: November 14-17, 2026, Milan.
- Format: ACM `sigconf`, double blind, 8 pages total including figures and references, no supplementary material/appendices, no rebuttal.
- Preprints/technical reports are permitted, including arXiv and workshops at AAAI/KDD/ICML/NeurIPS, but submissions cannot be under review at another archival venue during ICAIF review.

Recommendation: target arXiv/SSRN in November 2026, then ICAIF 2027 or a suitable workshop/journal.

### Alternative venues

| Venue | Fit | Timeline evidence | Recommendation |
|---|---|---|---|
| Finance Research Letters | Good for concise empirical finding if results are sharp | ScienceDirect journal page reports 4 days submission to first decision, 31 days after review, 87 days to acceptance; guide says suitable submissions are typically sent to at least one reviewer | Good post-capstone target if paper can be compressed to a letter-style contribution. |
| Journal of Commodity Markets | Best topical fit | ScienceDirect journal page reports 7 days to first decision, 65 days to decision after review, 260 days to acceptance | Stronger topical home, but acceptance timeline likely longer than capstone window. |
| Journal of Forecasting / International Journal of Forecasting | Good methodological fit | Wiley/Elsevier author pages verify forecasting scope; older IJF editor report cited 20.6 days average first decision and around 45 days reviewer turnaround, but this is historical and should be treated cautiously | Good if CuBench emphasizes forecasting protocol and probabilistic scoring more than finance novelty. |
| ICAIF 2027 / finance ML workshops | Strong AI-finance audience | ICAIF 2026 call explicitly lists financial time series, factor models, validation/calibration, robustness, uncertainty quantification | Best conference target after arXiv/SSRN preprint. |

### arXiv endorsement

arXiv currently requires endorsement before a user's first paper or first paper in a new category. The official help page says institutional email can expedite the process if the author has claimed prior papers; otherwise personal endorsement from an established arXiv author may be needed. There is no reliable "shortcut" to avoid endorsement by category shopping. For a first-time B.Tech submitter, plan for endorsement early, ideally through the supervisor or a coauthor with q-fin/cs.LG submission history.

## 5. Section D findings - rigor-tooling recommendations

### Backtest overfitting and Sharpe inference

- Canonical citations: Bailey and Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality" (Journal of Portfolio Management, 2014; SSRN abstract 2460551) and Bailey/Borwein/Lopez de Prado work on Probability of Backtest Overfitting.
- Implementation: `pypbo` on GitHub advertises Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, and PBO. It is a better starting point than hand-implementing formulas from memory.
- Caution: `mlfinlab` historically included these tools, but current package availability/licensing should be checked before depending on it.

### Diebold-Mariano test

- Prefer a maintained package or `statsmodels` if the local version includes `statsmodels.tsa.stattools.diebold_mariano_test`.
- Use Harvey-Leybourne-Newbold small-sample correction for modest OOS fold counts and multi-step horizons. The current statsmodels development docs explicitly include a flag for this correction.
- Apply Holm-Bonferroni or another pre-specified familywise/FDR correction across models, horizons, targets, and ablations.

### Pesaran-Timmermann directional test

- Python ecosystem support is thinner. Candidate options are a small standalone implementation, a local audited implementation with unit tests, or validating against R `rugarch::DACTest`, which implements the Pesaran-Timmermann directional accuracy test.
- Report excess accuracy over base rate, not raw directional accuracy alone. Always include always-up/always-down/majority-class forecasts.

### Constant-forecast/base-rate diagnostics

- Cheung (2026) is directly relevant prior art for "directional accuracy lies" and "base-rate-honest" evaluation. Cite it when framing CuBench's diagnostic apparatus.
- Portnaya (2026) is relevant for separating sign predictability from magnitude/autocorrelation effects.
- Recommended diagnostic per result cell:
  - raw directional accuracy;
  - test-period positive-rate and majority-class baseline;
  - excess directional accuracy over majority baseline;
  - predicted-positive rate;
  - confusion matrix;
  - Pesaran-Timmermann p-value;
  - McNemar test vs majority baseline where appropriate.

### Quantile/distributional tooling

- LightGBM supports `objective="quantile"` and `alpha`; train separate models per quantile and check quantile crossing.
- CatBoost supports `Quantile` and `MultiQuantile` losses, useful as a comparison point.
- XGBoost has quantile regression support but documentation warns quantile crossing can occur. Treat XGBoost quantile results as comparison, not the primary implementation unless crossing is handled.

## 6. Full source list

### arXiv MCP sources

All arXiv entries below were found or verified via the arXiv MCP on 2026-08-19.

- Wang, Z. and Lu, X. (2024). "COMEX Copper Futures Volatility Forecasting: Econometric Models and Deep Learning." arXiv:2409.08356. https://arxiv.org/abs/2409.08356
- Wang, Z. and Li, X. (2024). "On the macroeconomic fundamentals of long-term volatilities and dynamic correlations in COMEX copper futures." arXiv:2409.08355. https://arxiv.org/abs/2409.08355
- Dudek, G., Kasprzyk, M., and Pelka, P. (2025). "Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting." arXiv:2511.20105. https://arxiv.org/abs/2511.20105
- Fang, X. and Slepaczuk, R. (2026). "Volatility Forecasting and Return Prediction under Market Regimes." arXiv:2606.09478. https://arxiv.org/abs/2606.09478
- Cheung, T. (2026). "When Directional Accuracy Lies." arXiv:2607.12248. https://arxiv.org/abs/2607.12248
- Brini, A. (2026). "Forecasting Realized Volatility with Time Series Foundation Models." arXiv:2607.05291. https://arxiv.org/abs/2607.05291
- Dudek, G., Orzeszko, W., and Fiszeder, P. (2025). "Probabilistic Forecasting Cryptocurrencies Volatility." arXiv:2508.15922. https://arxiv.org/abs/2508.15922
- Muhammad, T. et al. (2026). "A Benchmark of Classical and Deep Learning Models for Agricultural Commodity Price Forecasting." arXiv:2604.06227. https://arxiv.org/abs/2604.06227
- Portnaya, V. (2026). "The Bounce Has No Direction." arXiv:2606.29591. https://arxiv.org/abs/2606.29591
- Benhamou, E. et al. (2025). "Re-evaluating Short- and Long-Term Trend Factors in CTA Replication." arXiv:2507.15876. https://arxiv.org/abs/2507.15876
- He, P. et al. (2024). "Multi-Factor Function-on-Function Regression of Bond Yields on WTI Commodity Futures Term Structure Dynamics." arXiv:2412.05889. https://arxiv.org/abs/2412.05889
- Matsubara, T. (2024). "Wasserstein Gradient Boosting." arXiv:2405.09536. https://arxiv.org/abs/2405.09536
- Lee, J. et al. (2026). "FinVerse: Financial Time-Series Benchmark." arXiv:2608.03259. https://arxiv.org/abs/2608.03259
- Amorin, A. P., Python, A., and Weisser, C. (2026). "Not All News Is Equal." arXiv:2603.09085. https://arxiv.org/abs/2603.09085
- Lee, N. et al. (2024). "Metal Price Spike Prediction via a Neurosymbolic Ensemble Approach." arXiv:2410.12785. https://arxiv.org/abs/2410.12785

### Other academic and methodology sources

- Bakshi, G., Gao, X., and Rossi, A. G. "Understanding the Sources of Risk Underlying the Cross Section of Commodity Returns." Management Science/RePEc. Accessed 2026-08-19. https://ideas.repec.org/a/inm/ormnsc/v65y2019i2p619-641.html
- Bakshi, G., Gao, X., and Rossi, A. G. INFORMS page. Accessed 2026-08-19. https://pubsonline.informs.org/doi/10.1287/mnsc.2017.2840
- Yang, F. "Investment Shocks and the Commodity Basis Spread." ScienceDirect. Accessed 2026-08-19. https://www.sciencedirect.com/science/article/abs/pii/S0304405X13001360
- Angelidis, T., Sakkas, A., and Tessaromatis, N. "Predicting Commodity Returns: Time Series vs. Cross Sectional Prediction Models." SSRN. Accessed 2026-08-19. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5205084
- Journal of Commodity Markets listing for Angelidis et al. Accessed 2026-08-19. https://ideas.repec.org/a/eee/jocoma/v38y2025ics2405851325000194.html
- Bailey, D. H. and Lopez de Prado, M. "The Deflated Sharpe Ratio." SSRN. Accessed 2026-08-19. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- `pypbo` GitHub repository. Accessed 2026-08-19. https://github.com/esvhd/pypbo
- `dieboldmariano` PyPI package. Accessed 2026-08-19. https://pypi.org/project/dieboldmariano/
- statsmodels development docs for `diebold_mariano_test`. Accessed 2026-08-19. https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.diebold_mariano_test.html
- R `rugarch::DACTest` documentation. Accessed 2026-08-19. https://www.rdocumentation.org/packages/rugarch/versions/1.5-6/topics/DACTest
- LightGBM parameters documentation. Accessed 2026-08-19. https://lightgbm.readthedocs.io/en/latest/Parameters.html
- CatBoost regression losses documentation. Accessed 2026-08-19. https://catboost.ai/docs/en/concepts/loss-functions-regression
- XGBoost quantile regression documentation. Accessed 2026-08-19. https://xgboost.readthedocs.io/en/release_3.2.0/python/examples/quantile_regression.html

### Data and venue sources

- Yahoo Finance `HG=F` historical page. Accessed 2026-08-19. https://finance.yahoo.com/quote/HG%3DF/history/
- Direct Yahoo chart endpoint tests run locally on 2026-08-19 for `HG=F`, `QC=F`, `HGZ25.CMX`, and `HGZ15.CMX`.
- CME Group Copper product page. Accessed 2026-08-19. https://www.cmegroup.com/markets/metals/base/copper.html
- CME Group Continuous Price Series. Accessed 2026-08-19. https://www.cmegroup.com/market-data/cme-group-continuous-price-series.html
- CME DataMine. Accessed 2026-08-19. https://www.cmegroup.com/datamine.html
- Investing.com Copper Futures historical data. Accessed 2026-08-19. https://www.investing.com/commodities/copper-historical-data
- FRED `DFII10`. Accessed 2026-08-19. https://fred.stlouisfed.org/series/DFII10
- FRED `BAMLH0A0HYM2`. Accessed 2026-08-19. https://fred.stlouisfed.org/series/BAMLH0A0HYM2
- FRED `VIXCLS`. Accessed 2026-08-19. https://fred.stlouisfed.org/series/VIXCLS
- Cboe VIX historical data page. Accessed 2026-08-19. https://www.cboe.com/tradable-products/vix/vix-historical-data
- National Bureau of Statistics of China English site. Accessed 2026-08-19. https://www.stats.gov.cn/english/
- NBS Purchasing Managers' Index for June 2026. Accessed 2026-08-19. https://www.stats.gov.cn/english/PressRelease/202607/t20260701_1964047.html
- ChinaData.live China PMI API page. Accessed 2026-08-19. https://chinadata.live/data/china-pmi/
- Caixin Manufacturing PMI page. Accessed 2026-08-19. https://www.caixinglobal.com/caixin-manufacturing-pmi/
- ICAIF 2026 Call for Papers. Accessed 2026-08-19. https://icaif2026.org/call-for-papers.html
- ICAIF 2026 Important Dates. Accessed 2026-08-19. https://icaif2026.org/important-dates.html
- arXiv endorsement help. Accessed 2026-08-19. https://info.arxiv.org/help/endorsement.html
- Finance Research Letters ScienceDirect page. Accessed 2026-08-19. https://www.sciencedirect.com/journal/finance-research-letters
- Journal of Commodity Markets ScienceDirect page. Accessed 2026-08-19. https://www.sciencedirect.com/journal/journal-of-commodity-markets
- Journal of Forecasting author guidelines. Accessed 2026-08-19. https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/forauthors.html
