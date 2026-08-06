# Literature Review: State-of-the-Art in Copper Price Forecasting

**Compiled: April 2026**
**Scope: 2018--2026, with emphasis on 2020--2026**

---

## Table of Contents

1. [Introduction and Market Context](#1-introduction-and-market-context)
2. [Traditional/Econometric Approaches](#2-traditionaleconometric-approaches)
3. [Machine Learning Approaches](#3-machine-learning-approaches)
4. [Hybrid/Decomposition Approaches](#4-hybriddecomposition-approaches)
5. [Graph Neural Network Approaches](#5-graph-neural-network-approaches)
6. [Key Datasets and Features](#6-key-datasets-and-features)
7. [Synthesis and Research Gaps](#7-synthesis-and-research-gaps)
8. [Reference Table](#8-reference-table)

---

## 1. Introduction and Market Context

Copper is often called "Dr. Copper" for its perceived ability to signal economic health. Global copper output reached approximately 22.8 million metric tons in 2024, with Chile as the leading producer. China accounts for over half of global demand, making Chinese economic activity a dominant driver of copper prices. The ongoing energy transition---electric vehicles (requiring ~91 kg Cu per vehicle vs. ~24.5 kg for ICE vehicles), renewable energy infrastructure, and grid electrification---is fundamentally reshaping demand dynamics. Recycled copper supplies roughly one-third of current demand.

Copper is primarily traded on the London Metal Exchange (LME), COMEX (CME Group), and the Shanghai Futures Exchange (SHFE). Price formation involves complex interactions among supply-side constraints (mine production, smelter capacity, TC/RC charges, inventory levels), demand-side drivers (Chinese industrial activity, global manufacturing PMI, construction), macroeconomic factors (USD strength, interest rates, oil prices), and speculative positioning.

This review surveys the state-of-the-art in copper price forecasting across four methodological families: traditional econometrics, machine learning, hybrid decomposition methods, and graph neural networks.

---

## 2. Traditional/Econometric Approaches

### 2.1 ARIMA and SARIMA Models

**Core methodology.** Autoregressive Integrated Moving Average (ARIMA) models and their seasonal variant (SARIMA) remain baseline benchmarks. They model copper prices as a function of their own lagged values and past forecast errors.

| Study | Year | Method | Dataset | Horizon | Key Results |
|-------|------|--------|---------|---------|-------------|
| Becerra, Jerez, Garces & Demarco | 2022 | SARIMA / SARIMAX | LME copper, 1991--2020 (30 yrs) | 1-year rolling window | MAPE: 3.98% (train), 4.31% (test); adding China's Leading Economic Index improved test MAPE to 4.26% |
| Wang & Zhang | 2020 | X12-ARIMA-GARCH | Shanghai copper futures, 2018--2019 | Short-term | Seasonal decomposition via X12 improved GARCH forecasting (17 citations) |
| Balioz | 2022 | VAR, ARIMA, VECM | Global energy and metal prices | Short-term | Comparative evaluation; VECM outperformed VAR for cointegrated series |
| Tang, Li, Wu & Meng | 2026 | ARIMA, LSTM, SVR comparison | COMEX copper | Short and long-term | ARIMA competitive for short horizons; ML dominates at longer horizons |

**Typical features:** Univariate price series; SARIMAX variants incorporate exogenous macro variables (China PMI, inventory levels, dollar index).

**Limitations:**
- Linearity assumption fails to capture nonlinear dynamics inherent in commodity markets
- Stationarity requirement necessitates differencing, losing level information
- Poor performance at longer horizons (>1 month)
- Cannot incorporate high-dimensional feature sets
- Structural breaks (e.g., COVID-19, supply disruptions) degrade performance

### 2.2 GARCH Family Models

**Core methodology.** Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models target volatility forecasting, capturing the well-documented volatility clustering in copper prices.

| Study | Year | Method | Dataset | Key Results |
|-------|------|--------|---------|-------------|
| Garcia & Kristjanpoller | 2019 | GARCH-FIS (Fuzzy Inference System) | Monthly copper prices | Adaptive hybrid GARCH-FIS yielded best forecasting power among compared models (90 citations) |
| Alipour, Khodaiari & Jafari | 2019 | GARCH variants | Time series of monthly copper prices | Comparative evaluation of GARCH specifications |
| Wang & Zhang | 2020 | X12-ARIMA-GARCH family | Shanghai copper futures | Seasonal adjustment via X12 improved volatility forecasts |
| Singh, Ardian & Kumral | 2021 | ARFIMA-GARCH | Gold-copper mining | Long-memory ARFIMA captured persistent volatility patterns |
| Ernanto, Wiryono & Faturohman | 2025 | VECM + EGARCH | Copper prices, 1992--2023 | Identified cointegration relationships; EGARCH captured asymmetric volatility |

**Typical features:** Returns series, squared returns, realized volatility; some models incorporate exogenous regressors (dollar index, gold returns).

**Limitations:**
- Primarily for volatility, not level forecasting
- Parametric distributional assumptions (Gaussian, Student-t) may be inadequate for heavy-tailed copper returns
- Limited ability to model regime changes
- Cannot easily incorporate non-price features (text, supply data)

### 2.3 VAR / VECM / Cointegration Models

**Core methodology.** Vector Autoregression (VAR) and Vector Error Correction Models (VECM) model copper prices jointly with related macroeconomic and commodity variables, exploiting long-run equilibrium relationships (cointegration).

| Study | Year | Method | Features | Key Results |
|-------|------|--------|----------|-------------|
| Salles, Magrath & Malheiros | 2019 | VAR, VEC | International copper market variables | Error correction mechanism captures long-run equilibrium reversion |
| Cole | 2023 | ARDL-ECM | US industrial production, copper prices | Demand-supply framework; industrial production Granger-causes copper prices |
| Marioli & Letelier | 2021 | ECM | USD copper price decomposition | Fundamental valuation via error correction; identified over/undervaluation periods |
| Dai | 2025 | SVAR | Chinese macroeconomic indicators | Structural shocks identified; China demand shocks most influential |
| Balioz | 2022 | VAR, VECM | Global energy and metal prices | VECM superior when cointegration present |

**Typical features:** GDP, industrial production, exchange rates, oil prices, interest rates, inventory levels, related metal prices. SVAR models impose structural identification restrictions based on economic theory.

**Limitations:**
- Linearity in relationships
- Limited to moderate-dimensional systems (typically 3--8 variables)
- Identification in structural VAR depends on debatable theoretical assumptions
- Forecasting accuracy degrades rapidly beyond 1--3 quarters
- Cannot capture nonlinear interactions between features

---

## 3. Machine Learning Approaches

### 3.1 Recurrent Neural Networks (LSTM, GRU)

LSTM-based architectures dominate the recent copper forecasting literature, comprising the largest single methodological category.

| Study | Year | Method | Dataset | Horizon | Key Results |
|-------|------|--------|---------|---------|-------------|
| Hu, Ni & Wen | 2020 | LSTM-ANN + GARCH | Copper price volatility | Multi-horizon | Hybrid integrating LSTM/ANN with GARCH for volatility; 204 citations, highly influential |
| Luo, Wang, Cheng & Wu | 2022 | Improved LSTM with error correction | LME copper | Multi-step-ahead | Two-phase architecture with novel input strategy; error correction reduces accumulated errors (35 citations) |
| Ni, Xu, Li & Zhao | 2022 | RNN ensemble | SHFE copper futures | Multiple | Ensemble averaging of LSTM variants outperformed single-model RNNs and classic ANN (14 citations) |
| Li, Zhou, Liu & Ding | 2023 | CNN-LSTM | Multi-factor copper data | Medium to long-term | CNN extracts spatial features from multi-factor inputs; LSTM models temporal dependencies (27 citations) |
| Chen, Yi, Liu, Cheng, Feng & Fang | 2023 | LSTM + Simulated Annealing | Copper prices | Short-term | SA optimizes LSTM hyperparameters; improved convergence speed (9 citations) |
| Shi, Li, Zhang & Zhang | 2023 | LSTM-GRU hybrid | Metal resource spot prices | Multi-horizon | Combined LSTM and GRU for complementary temporal modeling (20 citations) |

**Key observations:**
- LSTM remains the workhorse architecture, but is increasingly combined with other components (CNN for spatial features, attention for focus, optimization algorithms for hyperparameters)
- Multi-step-ahead forecasting is an active area; error correction mechanisms address accumulated prediction errors
- GARCH hybridization remains popular for volatility-specific applications

### 3.2 Support Vector Regression (SVR) and Kernel Methods

| Study | Year | Method | Dataset | Key Results |
|-------|------|--------|---------|-------------|
| Astudillo, Carrasco & Fernandez-Campusano | 2020 | SVR with external recurrences | LME, SHFE, COMEX copper | SVR with tailored kernel functions competitive for short-horizon; 54 citations |
| Ling, Zhong & Wei | 2025 | Improved LSSVM + butterfly optimization + wavelet functions | Copper prices | LSSVM with wavelet kernels and metaheuristic optimization for short/medium/long-term |

### 3.3 Tree-Based and Ensemble Methods

| Study | Year | Method | Dataset | Key Results |
|-------|------|--------|---------|-------------|
| Hu | 2023 | XGBoost vs. VAR | LME copper (small dataset) | XGBoost's feature learning enhanced accuracy; competitive with VAR on limited data |
| Vancsura, Tatay & Bareith | 2023 | XGBoost, Random Forest | Commodity futures incl. copper | Decision trees provided more reliable copper price prediction than some DL methods (13 citations) |
| Nabavi et al. | 2024 | Extreme gradient boosting + metaheuristics | 30-year copper data | RMSE = 106; hybrid metaheuristic optimization improved XGBoost (21 citations) |
| Oikonomou & Damigos | 2025 | LightGBM, LightGBM-ARIMA | Base metals incl. copper | Ensemble LightGBM-ARIMA outperformed standalone models for short-term forecasting |

### 3.4 Extreme Learning Machines (ELM)

| Study | Year | Method | Dataset | Key Results |
|-------|------|--------|---------|-------------|
| Zhang, Nguyen, Bui, Pradhan & Mai | 2021 | GA-ELM, PSO-ELM | Historical copper prices | RMSE = 304.943 (GA-ELM); metaheuristic optimization of ELM significantly improved over vanilla ELM (47 citations) |
| Yu | 2026 | VMD-ELM + Beetle Moth Optimizer | Copper prices | VMD decomposition + optimized ELM; addressed high price volatility |

### 3.5 Transformer and Attention-Based Models

Transformer architectures are an emerging and rapidly growing category in copper forecasting (primarily 2024--2026).

| Study | Year | Method | Key Innovation |
|-------|------|--------|----------------|
| Tseng & Nguyen | 2025 | Transformer with pattern selection | Simplified encoder-decoder; pattern selection framework; compared against RNN, LSTM, CNN, GRU |
| Zhao, Guo & Wang | 2025 | LSTM-Transformer hybrid | Multi-scale feature fusion; LSTM for local patterns, Transformer for global attention; MAE superior to CNN-GRU |
| Wu, Shang & Cao | 2025 | GAT + BERT/LLM (NLP) | Integrated NLP-derived market risk perception with Graph Attention Networks; novel text+graph fusion |
| Waleed, Hasan, Abdullah & Alkhayyat | 2025 | 1D-CNN, LSTM, Temporal Fusion Transformer | Comprehensive comparison of attention mechanisms for copper volatility |
| Swarup & Kushwaha | 2023 | Temporal Fusion Transformer (TFT) | Applied TFT to nickel and cobalt; attention mechanism captures variable-importance interpretability |

**Key observations:**
- Transformers offer superior long-range dependency modeling compared to LSTM/GRU
- Temporal Fusion Transformers (TFT) provide built-in interpretability via attention weights
- Still early-stage for copper specifically; most copper-specific results are from 2025--2026
- Computational cost and data requirements are higher than RNN alternatives

### 3.6 CNN-Based Approaches

| Study | Year | Method | Key Results |
|-------|------|--------|-------------|
| Derakhshani, GhasemiNejad & Amani Zarin | 2024 | 1D-CNN | Enhanced global copper price forecast accuracy (6 citations) |
| Li, Zhou, Liu & Ding | 2023 | CNN-LSTM | CNN extracts cross-sectional feature patterns; LSTM handles temporal dynamics (27 citations) |
| Liu et al. | 2023 | CEEMDAN-CNN-LSTM | Multi-factor selection and fusion; decomposition feeds CNN-LSTM pipeline (14 citations) |

---

## 4. Hybrid/Decomposition Approaches

This is the most methodologically active area in recent copper price forecasting research. The core idea: decompose the nonstationary, noisy copper price series into simpler sub-components, forecast each independently, then aggregate.

### 4.1 EMD / EEMD / CEEMDAN + ML

**Empirical Mode Decomposition (EMD)** and its improved variants decompose a signal into Intrinsic Mode Functions (IMFs) of decreasing frequency. CEEMDAN (Complete Ensemble EMD with Adaptive Noise) addresses the mode mixing problem of original EMD.

| Study | Year | Pipeline | Dataset | Key Results |
|-------|------|----------|---------|-------------|
| Liu, Guo & Wei | 2024 | EMD + LASSO feature selection | 75 indicators, copper prices | LASSO selected key features; EMD captured multi-scale dynamics; 6-dimension analysis (11 citations) |
| Li, Yang, Chen & Huang | 2023 | CEEMDAN-SSA + LSTM | Non-ferrous metals incl. copper | CEEMDAN decomposes price; SSA (Singular Spectrum Analysis) refines; LSTM forecasts components (11 citations) |
| Liu, Liu, Zhou & Yan | 2022 | CEEMDAN + VMD (secondary) + LSTM | LME copper, aluminum, zinc | CEEMDAN primary decomposition; VMD applied to high-entropy components; multi-model fusion (30 citations) |
| Huang et al. | 2023 | ICEEMDAN + Bayesian-optimized GRU + ARIMA | Copper prices | ICEEMDAN addresses residual noise; Bayesian optimization tunes GRU (16 citations) |
| Liu et al. | 2023 | CEEMDAN + CNN-LSTM | Copper prices | Multi-factor selection and fusion; CNN captures cross-factor patterns (14 citations) |
| Huang et al. | 2024 | CEEMD + GRU | Daily electrolytic copper prices | Improved GRU performance on decomposed series (3 citations) |
| Antwi | 2023 | EMD, VMD + BPNN | Commodity futures | Comparative study; decomposition consistently outperformed direct forecasting |

### 4.2 VMD (Variational Mode Decomposition) + ML

VMD decomposes a signal into a set of band-limited IMFs by solving a constrained variational problem. Unlike EMD (which is recursive/heuristic), VMD is mathematically well-defined and avoids mode mixing more robustly.

| Study | Year | Pipeline | Dataset | Key Results |
|-------|------|----------|---------|-------------|
| Liu, Yang, Huang & Gui | 2020 | VMD + LSTM | LME daily futures (zinc, copper) | **Seminal paper.** VMD-LSTM significantly outperformed EEMD-LSTM, single LSTM, ARIMA, SVR. 290 citations---the most cited paper in this review. |
| Du et al. | 2020 | VMD + Outlier-Robust ELM | Metal prices | Addressed both point and interval forecasting; robust to outliers |
| Zhao et al. | 2023 | VMD-SSA-LSTM | Non-ferrous metal prices | SSA further decomposes high-frequency VMD components; point and interval prediction |
| Yu | 2026 | VMD-ELM + Beetle Moth Optimizer | Copper prices | VMD reduces mode mixing and end effects; BMO optimizes ELM weights |
| Li & Liu | 2026 | Complexity-aware VMD + LSTM/GRU | LME and SHFE copper futures | VMD parameters adapted to local complexity; noted LSTM/GRU degradation at long horizons |

### 4.3 Secondary/Quadratic Decomposition

A growing trend (2024--2026) applies two-stage decomposition: a primary decomposition (usually CEEMDAN) followed by a secondary decomposition (usually VMD) on the high-frequency or high-entropy residual components.

| Study | Year | Pipeline | Key Innovation |
|-------|------|----------|----------------|
| Liu, Liu, Zhou & Yan | 2022 | CEEMDAN + VMD (secondary) | VMD applied to high-entropy CEEMDAN components; multi-model fusion for LME metals (30 citations) |
| Zhang, Peng & Song | 2025 | GA-VMD-EEMD + GRU | Genetic algorithm optimizes VMD parameters; secondary EEMD error correction (Journal of Big Data) |
| Zhang & Ke | 2025 | CEEMDAN + VMD + feature fusion | Multi-stage optimization; addresses EEMD drawbacks via CEEMDAN primary + VMD secondary (Computational Economics) |
| Cui, Yu & Zhang | 2024 | ICEEMDAN + VMD + CNN-LSTM | Quadratic extraction; applied to stock prices with transferable methodology |
| Jiang, Miao & Tang | 2025 | CEEMDAN-VMD + LSTM | Quadratic decomposition for agricultural commodity price prediction |
| Guo, Li, Wang & Duan | 2025 | VMD + CEEMDAN (two-layer) + XGBoost + whale optimization | Gold prices; dual decomposition with optimized ensemble; methodology applicable to copper |

### 4.4 Wavelet Decomposition + ML

| Study | Year | Pipeline | Dataset | Key Results |
|-------|------|----------|---------|-------------|
| Liu, Cheng & Yi | 2022 | Wavelet + Bayesian-optimized hybrid NN | Copper prices | Bayesian optimization tunes wavelet-NN architecture; 61 citations |
| Wang | 2022 | Wavelet packet decomposition + stochastic DL | Metal futures | Wavelet packets capture multi-resolution structure (12 citations) |
| Maleky Khorram & Nourollahzadeh | 2024 | Wavelet-ARIMA + GARCH | Steel prices (copper as feature) | Wavelet denoising improved ARIMA-GARCH for volatility (12 citations) |

### 4.5 Summary: Decomposition Method Comparison

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **EMD** | Adaptive, data-driven, no basis functions needed | Mode mixing, end effects, no mathematical foundation |
| **EEMD** | Reduces mode mixing via noise-assisted averaging | Residual noise, computationally expensive, incomplete decomposition |
| **CEEMDAN** | Near-complete reconstruction, reduced residual noise | Still relies on ensemble averaging; parameter sensitivity |
| **VMD** | Mathematically rigorous, no mode mixing, concurrent decomposition | Requires pre-specifying number of modes (K); sensitive to K selection |
| **Wavelets** | Multi-resolution, well-understood mathematically | Requires choice of mother wavelet and decomposition level; less adaptive than EMD family |
| **Secondary decomposition (e.g., CEEMDAN+VMD)** | Captures both trend and high-frequency detail effectively | Increased model complexity; risk of overfitting; longer training time |

**Consensus finding:** Across the literature, VMD-based pipelines (especially VMD-LSTM) consistently outperform single-model approaches. Liu et al. (2020) with 290 citations established VMD-LSTM as the benchmark. Secondary decomposition methods (CEEMDAN+VMD) represent the current frontier (2024--2026).

---

## 5. Graph Neural Network Approaches

### 5.1 Direct Application to Copper

GNN applications to copper price forecasting are nascent but growing rapidly (primarily 2024--2026).

| Study | Year | Method | Graph Construction | Dataset | Key Results |
|-------|------|--------|--------------------|---------|-------------|
| Sun, Yang & Zhong | 2025 | Multi-view Graph Transformer + Fractional Brownian Motion augmentation | Multiple graph views capturing different relationship types | Copper prices | Outperformed ANN, LSTM, and standard GNN models in both accuracy and training efficiency (Natural Resources Research) |
| Wu, Shang & Cao | 2025 | GAT + BERT/LLM NLP | Graph Attention Network with NLP-derived sentiment nodes | Copper commodity prices | Novel integration of text-based market risk perception with graph-structured relational learning (IEEE) |
| Li & Liu | 2026 | VMD + temporal attention with multi-behavior analysis | Implicit graph structure via attention over related factors | LME/SHFE copper futures | Complexity-aware decomposition with attention-based aggregation (Entropy) |

### 5.2 GNN for Commodity/Futures Markets (Transferable to Copper)

| Study | Year | Method | Graph Construction | Dataset | Key Results |
|-------|------|--------|--------------------|---------|-------------|
| Hu, Tan, Liu & Yin | 2023/2024 | Heterogeneous Continual GNN (STGNN) | Nodes = 49 commodity futures contracts; edges = dynamic cross-sectional correlations | Chinese futures market (49 contracts) | Three heterogeneous tasks (MA regression, gap regression, change-point detection); outperformed SOTA; addresses catastrophic forgetting |
| Zhai, Guo, Jiang, Ou & Ye | 2024 | Graph Fourier Transformer | Nodes = commodity futures; edges = price co-movement; Graph Fourier Transform in spectral domain | Base metal futures incl. copper | Novel spectral-graph approach; captures inter-commodity relationships |
| Tan, Hu, Liu & Yin | 2024 | Heterogeneous Continual GNN | Dynamic correlation-based graph | 49 Chinese commodity futures | Continuous learning addresses non-stationarity in futures markets |
| Zhang, Yu, Zeng, Zhang & Lin | 2025 | Multivariate Temporal GNN + text | Graph of commodity futures with textual node features | Six types of commodity futures | Incorporates IGFT (Inverse Graph Fourier Transform); text-enhanced node embeddings (6 citations) |

### 5.3 GNN for Financial Time Series (Methodologically Relevant)

| Study | Year | Method | Graph Construction | Key Innovation |
|-------|------|--------|--------------------|----------------|
| Lazcano, Herrera & Monge | 2023 | RNN + GCN combined | Financial asset graph | RNN for temporal, GCN for cross-asset relationships; applied to precious metals; **124 citations** |
| Xiang, Cheng, Shang & Zhang | 2022 | Heterogeneous GAT with temporal layers | Heterogeneous financial graph | Multi-type nodes and edges; captures sector/industry relationships; 176 citations |
| Kim & Park | 2024 | STAD-GCN (Spatial-Temporal Attention Dynamic GCN) | Dynamic graph structure (updated over time) | Attention-weighted spatial-temporal modeling; Expert Systems with Applications (23 citations) |
| Wei et al. | 2025 | FSTGAT (Financial Spatio-Temporal GAT) | Gated causal convolution enforces temporal causality | Prevents future information leakage; tested on NYSE banking and metal sectors (10 citations) |
| Ansari | 2024 | Multi-cluster Graph (MCG) | Clustering-based multi-relation graph | Novel clustering to define graph topology; stock price volatility (15 citations) |
| Amiri, Haddadi & Mojdehi | 2025 | GCN-LSTM | Inter-stock relationship graph | Graph convolution + attention for energy stocks (25 citations) |
| Wang et al. | 2025 | STDAsh-GCN | Adaptive shared graph | Decouples spatial and temporal feature propagation; Neural Networks |
| Korablyov, Fomichov, Kobzev & Antonov | 2025 | Evolving GNN | Time-evolving graph structure | Graph topology adapts over time for stock markets |

### 5.4 Graph Construction Paradigms in Commodity Forecasting

From the surveyed literature, four main approaches to graph construction emerge:

1. **Correlation-based graphs:** Nodes = individual commodities/futures; edges = price correlation (Pearson, dynamic conditional correlation, or mutual information). Most common approach.

2. **Supply-chain graphs:** Nodes = commodities at different production stages; edges = upstream/downstream relationships (e.g., copper ore -> concentrate -> refined copper -> manufactured goods).

3. **Multi-relational/heterogeneous graphs:** Multiple edge types encoding different relationships (price similarity, sector membership, supply-chain links, geographic proximity). Xiang et al. (2022) is the exemplar.

4. **Dynamic/evolving graphs:** Graph structure changes over time as correlations shift. Kim & Park (2024) and Korablyov et al. (2025) represent this approach.

**Key gap:** No published work (as of April 2026) has constructed a comprehensive graph combining supply-chain structure, macroeconomic linkages, and cross-commodity correlations specifically for copper price forecasting.

---

## 6. Key Datasets and Features

### 6.1 Price Data Sources

| Source | Description | Frequency | Coverage |
|--------|-------------|-----------|----------|
| **LME** (London Metal Exchange) | Official settlement prices, 3-month forward, cash | Daily | Primary global benchmark; most commonly used in academic literature |
| **COMEX** (CME Group) | Copper futures (HG contract) | Daily/Intraday | Dominant in US-focused studies; Tang et al. (2026) |
| **SHFE** (Shanghai Futures Exchange) | Copper cathode futures | Daily/Intraday | Chinese market; Ni et al. (2022), Wang et al. (2024), Li & Liu (2026) |

### 6.2 Feature Categories and Key Variables

**A. Macroeconomic Indicators**

| Variable | Predictive Value | Key Studies |
|----------|-----------------|-------------|
| US Dollar Index (DXY) | Strong inverse correlation; consistently significant | Tang et al. (2026), Ling et al. (2025), Maleky Khorram & Nourollahzadeh (2024) |
| Federal Funds Rate / LIBOR / SHIBOR | Significant for medium-term; captures monetary policy | Wu, Shang & Cao (2025), Cole (2023) |
| Global Manufacturing PMI | Leading indicator of industrial demand | Becerra et al. (2022), Krampen (2023) |
| US Industrial Production | Granger-causes copper prices | Cole (2023) |
| Global GDP growth | Long-term driver | Ernanto et al. (2025), Marioli & Letelier (2021) |

**B. Chinese Economic Indicators**

| Variable | Predictive Value | Key Studies |
|----------|-----------------|-------------|
| China Manufacturing PMI (Caixin + NBS) | Leading indicator; lagged effect observed | Tang et al. (2026), Becerra et al. (2022) |
| China Leading Economic Index | Modest improvement when added to SARIMA (MAPE: 4.31% -> 4.26%) | Becerra et al. (2022) |
| Chinese copper consumption | Direct demand proxy | Becerra et al. (2022), Dai (2025) |
| China real estate investment | Relevant but underexplored in forecasting models | Dai (2025) |
| Chinese structural demand shocks | Most influential in SVAR analysis | Dai (2025) |

**C. Supply-Side Variables**

| Variable | Predictive Value | Key Studies |
|----------|-----------------|-------------|
| LME/SHFE warehouse inventory | Inversely related to price; leading indicator | Liu et al. (2024), Tang et al. (2026), Wang (2025) |
| Global mine supply / refined production | Long-term supply constraint | Tang et al. (2026), Ryter et al. (2022) |
| TC/RC (Treatment/Refining Charges) | Proxy for concentrate market tightness; discounted from LME price | Otero Vina (2019) |
| Scrap supply | Affects refined output | Ryter et al. (2022) |
| Copper inventory-to-consumption ratio | Not widely used but theoretically relevant | Simon & Wood (2025) |

**D. Related Commodity Prices**

| Variable | Predictive Value | Key Studies |
|----------|-----------------|-------------|
| Gold price | "Great influence on copper price" (high correlation) | Chen et al. (2023) |
| Silver price | High influence | Chen et al. (2023) |
| Crude oil (WTI/Brent) | Short-term correlation; shared macro drivers | Ling et al. (2025) |
| Aluminum, zinc, nickel (base metals) | Co-movement within base metals complex | Liu et al. (2022), Papenfuss (2024) |

**E. Sentiment and News Data**

| Source | Method | Key Studies |
|--------|--------|-------------|
| News articles | BERT, LLMs for sentiment extraction | Wu, Shang & Cao (2025), Wang, Li & Zhang (2025) |
| Social media | Sentiment classification | Tseng & Nguyen (2025) |
| GPT-based sentiment | Aggregated news sentiment | Sharkey (2025) |
| Market risk perception (text-derived) | NLP + GAT integration | Wu, Shang & Cao (2025) |

**F. Feature Importance Rankings**

Liu et al. (2024) conducted the most comprehensive feature analysis, examining 75 indicators across six dimensions (inventory, supply, demand, macro, financial, sentiment) using LASSO and EMD for selection. Chen et al. (2023) found gold and silver prices to be the most influential features via feature importance analysis. The emerging consensus is:

1. **Most consistently important:** USD index, LME inventory, gold/silver prices, China PMI
2. **Important but horizon-dependent:** Oil prices (short-term), mine production (long-term), interest rates (medium-term)
3. **Emerging/underexplored:** NLP-derived sentiment, TC/RC charges, Chinese real estate data, EV adoption rates

### 6.3 Forecast Horizons

| Horizon | Typical Range | Dominant Methods |
|---------|---------------|-----------------|
| Short-term | 1 day -- 1 week | LSTM, GRU, SVR, ARIMA, XGBoost |
| Medium-term | 1 week -- 3 months | CNN-LSTM, VMD-LSTM, Transformer, CEEMDAN hybrids |
| Long-term | 3 months -- 1 year+ | VECM, fundamental models, CNN-LSTM with multi-factor inputs |

---

## 7. Synthesis and Research Gaps

### 7.1 Key Findings

1. **Decomposition + DL is the dominant paradigm (2022--2026).** VMD-LSTM (Liu et al., 2020, 290 citations) established the benchmark. Secondary decomposition (CEEMDAN + VMD) represents the current frontier.

2. **Transformers are emerging but not yet dominant.** Most copper-specific Transformer results are from 2025--2026. Temporal Fusion Transformers offer interpretability advantages.

3. **GNNs for copper are nascent.** Sun et al. (2025) is the first dedicated multi-view Graph Transformer for copper. The Hu et al. (2023) heterogeneous continual GNN covers 49 Chinese futures including copper. Significant opportunity exists.

4. **NLP/sentiment integration is a 2025--2026 trend.** Wu, Shang & Cao (2025) pioneered BERT/LLM + GAT for copper. Most prior work was purely quantitative.

5. **Chinese indicators are underexploited.** Despite China consuming >50% of global copper, only a handful of papers explicitly model Chinese economic variables. Becerra et al. (2022) found modest improvement; Dai (2025) found Chinese demand shocks most influential in SVAR.

6. **Feature engineering matters as much as model architecture.** Liu et al. (2024) with 75 indicators and LASSO selection suggests the field is moving toward high-dimensional, multi-source feature sets.

### 7.2 Research Gaps and Opportunities

| Gap | Description | Potential Approach |
|-----|-------------|-------------------|
| **GNN + supply chain structure** | No published work constructs a copper supply chain graph (mine -> smelter -> refinery -> end-use) for GNN-based forecasting | Heterogeneous graph with supply-chain edges + correlation edges + macro linkages |
| **Foundation models / transfer learning** | No published copper-specific work uses pre-trained time series foundation models (e.g., TimesFM, Chronos, Lag-Llama) | Fine-tune foundation models on copper with domain-specific features |
| **Chinese micro-data** | Chinese real estate starts, grid investment, EV sales, power consumption---all directly relevant but rarely used | High-frequency Chinese data integration with ML models |
| **TC/RC as features** | Treatment/refining charges are a key supply-side signal but almost entirely absent from ML forecasting papers | Include TC/RC, concentrate supply balance, smelter utilization |
| **Probabilistic forecasting** | Most papers report point forecasts only; limited uncertainty quantification | Quantile regression, conformal prediction, Bayesian NNs, distributional forecasting |
| **Multi-horizon unified models** | Most models target a single horizon; few attempt unified short+medium+long-term | Multi-task learning, hierarchical forecasting |
| **Regime-aware models** | Copper markets exhibit distinct regimes (bull/bear, contango/backwardation); few models explicitly handle regime changes | Hidden Markov Models + DL, regime-switching networks |
| **Explainability** | Most DL models are black boxes; TFT provides attention-based interpretability but is underused for copper | SHAP, attention analysis, symbolic regression |
| **Real-time / online learning** | Most models are trained offline on historical data; copper markets are non-stationary | Online learning, continual learning (cf. Hu et al., 2023) |
| **Cross-exchange arbitrage** | LME-SHFE-COMEX spread dynamics are commercially important but rarely modeled | Multi-exchange GNN or cointegration-DL hybrid |

---

## 8. Reference Table

### Most-Cited Papers (by Google Scholar citations as of April 2026)

| Rank | Authors | Year | Title (abbreviated) | Method | Citations |
|------|---------|------|---------------------|--------|-----------|
| 1 | Liu, Yang, Huang & Gui | 2020 | VMD-LSTM for metal price forecasting | VMD + LSTM | 290 |
| 2 | Hu, Ni & Wen | 2020 | LSTM-ANN + GARCH for copper volatility | LSTM-ANN-GARCH | 204 |
| 3 | Xiang, Cheng, Shang & Zhang | 2022 | Heterogeneous GAT for financial forecasting | Temporal heterogeneous GAT | 176 |
| 4 | Lazcano, Herrera & Monge | 2023 | RNN + GCN for financial time series | RNN-GCN combined | 124 |
| 5 | Alameer et al. | 2019 | Hybrid neuro-fuzzy for copper | Neuro-fuzzy systems | 118 |
| 6 | Garcia & Kristjanpoller | 2019 | Adaptive GARCH-FIS for copper volatility | GARCH-FIS | 90 |
| 7 | Liu, Cheng & Yi | 2022 | Wavelet + Bayesian NN for copper | Wavelet-BNN | 61 |
| 8 | Astudillo, Carrasco & Fernandez-Campusano | 2020 | SVR for copper price | SVR | 54 |
| 9 | Zhang, Nguyen, Bui, Pradhan & Mai | 2021 | ELM + metaheuristics for copper | GA-ELM, PSO-ELM | 47 |
| 10 | Luo, Wang, Cheng & Wu | 2022 | Multi-step LSTM with error correction | Improved LSTM | 35 |

### Papers by Methodology

**Econometric:** Becerra et al. (2022), Wang & Zhang (2020), Balioz (2022), Garcia & Kristjanpoller (2019), Ernanto et al. (2025), Salles et al. (2019), Cole (2023), Marioli & Letelier (2021), Dai (2025)

**LSTM/GRU:** Hu et al. (2020), Luo et al. (2022), Ni et al. (2022), Li et al. (2023), Chen et al. (2023), Shi et al. (2023)

**SVR/SVM:** Astudillo et al. (2020), Ling et al. (2025)

**Tree-based:** Hu (2023), Vancsura et al. (2023), Nabavi et al. (2024), Oikonomou & Damigos (2025)

**ELM:** Zhang et al. (2021), Yu (2026)

**Transformer:** Tseng & Nguyen (2025), Zhao et al. (2025), Wu et al. (2025), Swarup & Kushwaha (2023)

**CNN:** Derakhshani et al. (2024), Li et al. (2023 -- CNN-LSTM)

**Decomposition (EMD/CEEMDAN):** Li et al. (2023), Liu et al. (2022), Huang et al. (2023), Liu et al. (2023), Huang et al. (2024)

**Decomposition (VMD):** Liu et al. (2020), Du et al. (2020), Zhao et al. (2023), Yu (2026), Li & Liu (2026)

**Secondary decomposition:** Liu et al. (2022), Zhang et al. (2025), Zhang & Ke (2025), Guo et al. (2025)

**Wavelet:** Liu et al. (2022), Wang (2022), Maleky Khorram & Nourollahzadeh (2024)

**GNN (copper-specific):** Sun et al. (2025), Wu et al. (2025)

**GNN (commodity/financial):** Hu et al. (2023), Zhai et al. (2024), Tan et al. (2024), Zhang et al. (2025), Lazcano et al. (2023), Xiang et al. (2022), Kim & Park (2024), Wei et al. (2025)

**NLP/Sentiment:** Wu et al. (2025), Sharkey (2025), Sarkheil et al. (2025), Tseng & Nguyen (2025), Wang et al. (2025)

**Feature analysis:** Liu et al. (2024), Chen et al. (2023), Li et al. (2026 -- Shapley values)

---

*End of literature review.*
