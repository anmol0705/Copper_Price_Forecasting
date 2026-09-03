# VMD-MFGNN: Multi-Frequency Graph Neural Network with Variational Mode Decomposition for Copper Price Forecasting

## Master Research Blueprint

---

## 1. THE OPPORTUNITY (Executive Summary)

### The Core Finding

After surveying 150+ papers across copper forecasting, GNN in finance, VMD decomposition, and commodity fundamentals, we identify a **triple gap** at the intersection of three active research threads that have never been connected:

1. **VMD + GNN has NEVER been applied to financial/commodity data** (9 papers exist in traffic/air quality/wind, zero in finance)
2. **No paper constructs frequency-specific inter-variable graphs from decomposed modes** (all VMD+GNN papers use VMD as dumb preprocessing on pre-existing spatial graphs)
3. **GNN for copper/base metals is barely explored** (only Sun et al. 2025 Multi-view Graph Transformer, no decomposition)

### The Thesis

> "Commodity prices are interconnected, but the nature of those interconnections changes across time scales. Low-frequency copper-oil correlations (macro trends) differ fundamentally from high-frequency copper-oil correlations (daily noise). We build the first framework that discovers and exploits frequency-specific inter-commodity graph structures using VMD decomposition and GNN."

### Defensible Novelty Claims

| # | Claim | Evidence |
|---|-------|----------|
| 1 | First to combine VMD with GNN for any financial/commodity application | Confirmed: zero papers in finance; 9 papers in other domains |
| 2 | First to construct frequency-band-specific inter-variable graphs from decomposed modes | Confirmed: all prior VMD+GNN papers apply GNN to pre-existing spatial graphs |
| 3 | First multi-scale inter-commodity graph framework for copper/base metals | Confirmed: Sun et al. 2025 is the only copper GNN paper, uses no decomposition |
| 4 | First cross-frequency inter-commodity dependency modeling in finance | Confirmed: cross-frequency graph attention exists only in EEG (An et al. 2024) |
| 5 | First interpretable frequency-specific commodity dependency maps | Confirmed: no prior work offers scale-resolved commodity graph visualizations |

---

## 2. PROPOSED FRAMEWORK: VMD-MFGNN

### 2.1 Architecture Overview

```
Input: N commodity/macro time series (copper, Al, Zn, oil, DXY, PMI, etc.)
                |
        [VMD Decomposition] (per variable, or MVMD jointly)
                |
    Mode 1 (low-freq/trend)  Mode 2 (mid-freq/cycle)  ...  Mode K (high-freq/noise)
        |                         |                              |
    [Graph Construction]     [Graph Construction]           [Graph Construction]
    G_1: N nodes, edges =    G_2: N nodes, edges =         G_K: N nodes, edges =
    band-1 correlations      band-2 correlations            band-K correlations
        |                         |                              |
    [GNN Layer]              [GNN Layer]                    [GNN Layer]
    (GAT / GCN)              (GAT / GCN)                    (GAT / GCN)
        |                         |                              |
    [Temporal Module]        [Temporal Module]               [Temporal Module]
    (LSTM / Transformer)     (LSTM / Transformer)            (LSTM / Transformer)
        |                         |                              |
    Pred_1                   Pred_2                          Pred_K
        \                        |                              /
         \                       |                             /
          [Attention-Based Fusion Layer]
          (learnable mode importance weights)
                        |
                  Final Prediction
                        |
              [Error Correction] (optional)
```

### 2.2 Key Design Decisions

**Graph Construction (per frequency band):**
- **Primary:** Learned adaptive adjacency (MTGNN-style) — per-band node embeddings E_m yield A_m = softmax(ReLU(E_m1 * E_m2^T))
- **Secondary (for interpretability):** Rolling-window Pearson correlation on mode-m signals, thresholded
- **Ablation variant:** Granger causality on mode-m signals for causal graph

**Why per-band graphs matter (the economic argument):**
- At low frequencies (trend): copper correlates strongly with DXY, Chinese PMI, oil (macro co-movement)
- At mid frequencies (business cycle): copper correlates with LME inventory cycles, construction activity
- At high frequencies (daily): copper correlates with speculative positioning, VIX, cross-exchange arbitrage signals
- A single graph collapses these distinct dependency structures into noise

**GNN Architecture:**
- Graph Attention Network (GAT) per frequency band — attention weights are interpretable
- Alternative: GCN with mix-hop propagation (MTGNN-style) for capturing multi-hop dependencies
- Node count per graph: N = 10-15 variables (manageable, no scalability concerns)

**Temporal Module:**
- LSTM or GRU per frequency band per variable (standard, well-proven)
- Alternative: Transformer encoder for longer-range dependencies
- Lookback window: mode-dependent (longer for low-freq modes, shorter for high-freq)

**Fusion Layer:**
- Learnable attention weights: w_m = softmax(MLP(h_m)) where h_m is the hidden state from band m
- Enables interpretability: which frequency bands matter most for which forecast horizons?

**VMD Configuration:**
- K = 5 modes (sensitivity analysis: K = 3, 5, 7, 9)
- Alpha = 2000 (default), optimized via Bayesian optimization
- **Critical:** Use rolling-window VMD to prevent temporal leakage (Feng et al. 2025)

### 2.3 Why This Framework Over Alternatives

| Framework | Novelty | Feasibility | Interpretability | Publication Risk |
|-----------|---------|-------------|-----------------|-----------------|
| **A: VMD-MFGNN (recommended)** | Very High | High | High | Low |
| B: Heterogeneous Mode Graph | Very High | Medium-High | Medium | Medium |
| C: Regime-Adaptive VMD-GNN | Very High | Medium | High | Medium-High |
| D: Causal-VMD-GNN | High | Medium-High | Very High | Low |
| E: VMD-GNN-Transformer | High | High | Medium | Low |

**Recommendation:** Start with Framework A (VMD-MFGNN). It's the cleanest novelty, most feasible, has the best ablation structure, and can be extended to include elements of B-E in future work.

---

## 3. CLOSEST COMPETITORS & DIFFERENTIATION

### 3.1 Direct Competitors

| Paper | Year | What They Did | How We Differ |
|-------|------|---------------|---------------|
| Sun et al. | 2025 | Multi-view Graph Transformer for copper (5 graph views, fBM augmentation) | No decomposition; monthly only; we add VMD + frequency-specific graphs + daily |
| Zheng & Zhufu | 2025 | DCT + cross-scale GNN for stock prices | Uses DCT not VMD; single-asset not inter-commodity; stock not commodity |
| Ahmad et al. | 2024-25 | VMD-driven GCN for traffic | Traffic domain; pre-existing spatial graph; no frequency-specific graph construction |
| Cheng & Tsai | 2026 | MVMD + scale-aware GNN for wind speed | Wind domain; doesn't construct graphs FROM modes; different application |
| Liu et al. | 2019 | VMD + LSTM for non-ferrous metals | No GNN; no graph structure; no inter-commodity modeling; 252 citations = established baseline |

### 3.2 Key Differentiators

1. **We construct the graph FROM the decomposition** — all prior work applies GNN to pre-existing graphs after VMD preprocessing
2. **We model inter-commodity dependencies at each frequency scale** — no prior work does this
3. **We target copper specifically with domain-motivated features** — supply/demand, inventory, TC/RC, green transition drivers
4. **We address temporal leakage** (rolling-window VMD) — most VMD papers ignore this

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Dataset

**Target variable:** LME Copper 3-month futures (daily close), 2010-2025

**Graph nodes (N=12 input variables):**

| Variable | Rationale | Data Source |
|----------|-----------|-------------|
| LME Copper 3M | Target | Yahoo Finance / Quandl |
| LME Aluminum | Co-movement, substitution | Yahoo Finance |
| LME Zinc | Base metals complex | Yahoo Finance |
| LME Nickel | Base metals complex | Yahoo Finance |
| Gold (XAU) | Safe-haven dynamics | Yahoo Finance |
| Crude Oil (WTI) | Energy cost, macro proxy | Yahoo Finance (CL=F) |
| DXY (US Dollar Index) | Inverse correlation | Yahoo Finance (DX-Y.NYB) |
| S&P 500 | Risk appetite proxy | Yahoo Finance (^GSPC) |
| VIX | Volatility/fear gauge | Yahoo Finance (^VIX) |
| US 10Y Yield | Monetary policy | FRED (DGS10) |
| Baltic Dry Index | Shipping/trade proxy | FRED / Quandl |
| LME Copper Inventory | Supply-demand balance | LME / Quandl |

**Optional enrichment (for extended version):**
- Chinese Caixin Manufacturing PMI (monthly, interpolated)
- CFTC COT managed money positioning (weekly)
- TC/RC benchmark rates (quarterly)
- Copper options implied volatility

**Data split:**
- Training: 2010-2019 (10 years)
- Validation: 2020-2021 (2 years, includes COVID)
- Test: 2022-2025 (3 years, includes supply shocks, rate hikes, green transition)

### 4.2 Forecast Horizons

| Horizon | Trading Days | Rationale |
|---------|-------------|-----------|
| 1-day ahead | t+1 | Standard short-term benchmark |
| 1-week ahead | t+5 | Tactical trading horizon |
| 2-week ahead | t+10 | Medium-term position management |
| 1-month ahead | t+22 | Strategic allocation horizon |

### 4.3 Baselines (18 models)

**Econometric (3):**
1. ARIMA (auto-tuned via AIC)
2. VAR (with all 12 variables)
3. GARCH (for volatility comparison)

**Standard ML (2):**
4. XGBoost (with engineered features)
5. LightGBM

**Deep Learning (4):**
6. LSTM
7. GRU
8. CNN-LSTM (Li et al. 2023 style)
9. Transformer (vanilla encoder)

**Decomposition + DL (4):**
10. VMD-LSTM (Liu et al. 2019 — the 252-citation benchmark)
11. EMD-LSTM
12. CEEMDAN-LSTM
13. VMD-Transformer

**GNN (3):**
14. StemGNN (Cao et al. 2020)
15. MTGNN (Wu et al. 2020)
16. FourierGNN (Yi et al. 2023)

**Hybrid baselines (2):**
17. VMD + LSTM + attention (no graph)
18. GNN + Transformer (no decomposition)

### 4.4 Ablation Studies (8 experiments)

| # | Ablation | Tests |
|---|----------|-------|
| 1 | VMD vs. no decomposition | Full model vs. GNN on raw data |
| 2 | Multi-frequency vs. single graph | Per-mode graphs vs. one aggregate graph |
| 3 | Learned vs. predefined graph | Adaptive adjacency vs. correlation-based |
| 4 | GNN vs. no GNN | VMD + independent mode prediction (no cross-variable) |
| 5 | Attention fusion vs. averaging | Learnable mode weights vs. simple sum |
| 6 | Number of VMD modes | K = 3, 5, 7, 9 sensitivity |
| 7 | Graph construction method | Correlation vs. Granger causality vs. learned |
| 8 | Variable selection | All 12 vs. metals-only (6) vs. copper-only (1) |

### 4.5 Evaluation Metrics

**Point forecast accuracy:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R-squared

**Directional accuracy:**
- DA (Directional Accuracy) — % of correctly predicted up/down moves
- Annualized return of a simple long/short strategy based on predictions

**Statistical significance:**
- Diebold-Mariano test (pairwise vs. all baselines)
- Model Confidence Set (Hansen et al. 2011)

**Interpretability analysis:**
- Frequency-specific graph visualizations (heatmaps of A_m for each mode)
- Attention weight analysis across modes and forecast horizons
- Comparison of learned graphs vs. known economic relationships

---

## 5. PAPER STRUCTURE

### Suggested Title

**"Multi-Frequency Graph Neural Network with Variational Mode Decomposition for Copper Price Forecasting"**

Alternative for domain journals:
**"Interpretable Multi-Scale Inter-Commodity Modeling via VMD-Enhanced Graph Neural Networks: Evidence from Copper Markets"**

### Outline

1. **Introduction** (1.5 pages)
   - Copper price forecasting importance (Dr. Copper, green transition, supply deficit)
   - Limitations of current approaches (no inter-commodity graph modeling, no frequency-aware dependencies)
   - Our contribution (3-4 bullet points matching the novelty claims)

2. **Related Work** (2 pages)
   - 2.1 Copper price forecasting (traditional, ML, decomposition)
   - 2.2 Graph neural networks for financial time series
   - 2.3 VMD and signal decomposition in forecasting
   - 2.4 Gap identification (VMD + GNN for commodities = empty)

3. **Methodology** (3 pages)
   - 3.1 Problem formulation
   - 3.2 VMD decomposition with rolling window (address temporal leakage)
   - 3.3 Frequency-specific graph construction
   - 3.4 Multi-frequency GNN architecture
   - 3.5 Attention-based mode fusion
   - 3.6 Training procedure and loss function

4. **Experimental Setup** (1.5 pages)
   - 4.1 Data description and preprocessing
   - 4.2 Baselines (18 models)
   - 4.3 Implementation details (hyperparameters, hardware, training time)

5. **Results and Discussion** (3 pages)
   - 5.1 Main results table (all horizons, all metrics)
   - 5.2 Ablation study
   - 5.3 Interpretability analysis
     - Frequency-specific commodity dependency graphs
     - Mode importance across forecast horizons
     - Case studies: COVID crash, 2022 supply shock, 2024 green transition rally
   - 5.4 Economic significance (directional accuracy, simple trading strategy returns)

6. **Conclusion** (0.5 page)

Total: ~12 pages (suitable for Resources Policy, Expert Systems, or conference submission)

---

## 6. PUBLICATION STRATEGY

### 6.1 Target Venues (Ranked)

| Rank | Venue | IF | Fit | Review Time | Notes |
|------|-------|-----|-----|-------------|-------|
| 1 | **Resources Policy** | ~10.2 | Perfect | 3-6 months | THE copper forecasting venue; GNN angle is novel here |
| 2 | **Energy Economics** | ~12.8 | Good | 3-6 months | Copper's role in electrification provides narrative |
| 3 | **Expert Systems with Applications** | ~8.5 | Good | 2-4 months | Accepts hybrid ML methods readily |
| 4 | **Knowledge-Based Systems** | ~8.8 | Good | 2-4 months | VMD+GNN hybrid fits well |
| 5 | **IEEE TNNLS** | ~10.4 | Medium | 6-12 months | Needs stronger theoretical analysis |
| 6 | **KDD (Applied Data Science)** | Conference | Good | Deadline-based | "Decompose-then-graph" paradigm framing |
| 7 | **ICAIF** | Conference | Perfect | Deadline-based | ACM AI in Finance — very targeted |

### 6.2 Priority Actions for Novelty Protection

1. **File arXiv preprint immediately** after first complete results to establish priority
2. **Submit to Resources Policy** as primary target (fastest path, strongest domain fit)
3. **Prepare KDD/ICAIF version** as backup with more ML framing

### 6.3 What Reviewers Will Want

**Domain journal reviewers (Resources Policy):**
- Economic interpretability of frequency-specific graphs
- Comparison against econometric baselines (ARIMA, VAR)
- Multiple forecast horizons including practical ones (1-month)
- Robustness across market regimes (train on normal, test includes crisis)
- Diebold-Mariano statistical tests
- Feature importance analysis

**ML venue reviewers (KDD, AAAI):**
- Rigorous ablation proving each component's contribution
- Computational complexity analysis (training time, memory, scalability)
- Generalization to other commodities (oil, gold as secondary experiments)
- Comparison against SOTA GNN baselines (StemGNN, MTGNN, FourierGNN)
- Theoretical motivation for architecture choices

---

## 7. IMPLEMENTATION ROADMAP

### Week 1: Days 1-2 — Data Collection & Preprocessing

```
Tasks:
- [ ] Download all 12 input variables (2010-2025) via yfinance + FRED API
- [ ] Align all series to common trading days
- [ ] Handle missing data (forward fill, interpolation)
- [ ] Compute basic statistics, stationarity tests (ADF)
- [ ] Implement rolling-window VMD decomposition
- [ ] Visualize VMD modes for copper (verify economic interpretation)
- [ ] Set up train/val/test splits
```

### Week 1: Days 3-4 — Model Implementation

```
Tasks:
- [ ] Implement VMD-MFGNN in PyTorch + PyG
  - VMD decomposition module (rolling window)
  - Per-band graph construction (correlation + learned)
  - GAT layer per frequency band
  - LSTM temporal module per band
  - Attention fusion layer
- [ ] Implement all 18 baselines (many available in existing libraries)
  - statsmodels for ARIMA, VAR, GARCH
  - sklearn/xgboost/lightgbm for ML baselines
  - PyTorch for DL baselines
  - PyG for GNN baselines (StemGNN, MTGNN, FourierGNN — repos available)
- [ ] Set up evaluation pipeline (all metrics, DM test)
```

### Week 1: Days 5-7 — Experiments & Writing

```
Tasks:
- [ ] Run main experiments (all baselines + VMD-MFGNN, all horizons)
- [ ] Run ablation studies
- [ ] Generate interpretability visualizations
  - Frequency-specific graph heatmaps
  - Mode attention weights across horizons
  - Case study graphs (COVID, supply shock)
- [ ] Write paper (use the outline from Section 5)
- [ ] File arXiv preprint
```

### Key Libraries

```python
# Core
import torch
import torch_geometric  # PyG for GNN layers
from vmdpy import VMD  # VMD decomposition
import yfinance as yf  # Price data
from fredapi import Fred  # Macro data

# Baselines
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import xgboost as xgb
import lightgbm as lgb

# Evaluation
from scipy.stats import wilcoxon  # DM test alternative
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

---

## 8. RISK MITIGATION

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VMD-MFGNN doesn't beat simpler baselines | Medium | Ablations identify which components add value; even if GNN only matches MTGNN, the decomposition + interpretability is novel |
| Temporal leakage invalidates results | Low (if careful) | Use rolling-window VMD; compare against standard VMD to quantify leakage |
| Someone publishes VMD+GNN for commodities first | Low-Medium | File arXiv preprint quickly; target domain journal where application novelty is valued |
| Cheng & Tsai (2026) MVMD+GNN gets high visibility | Medium | Different domain (wind), different graph construction (pre-existing spatial), clearly differentiate |
| Reviewers question computational cost | Low | Report training time; VMD is O(KT log T), GNN is O(N^2 per layer) — both manageable |
| Reviewers want more baselines | Medium | 18 baselines should be sufficient; add VMD-XGBoost, EMD-GNN if requested |

---

## 9. EXTENSIONS (Future Work / Paper 2)

After the core VMD-MFGNN paper:

1. **Causal-VMD-MFGNN:** Replace correlation-based graphs with Granger causality on decomposed modes (Framework D)
2. **Regime-Adaptive VMD-MFGNN:** Add HMM regime detector, adapt VMD K and graph structure to regime (Framework C)
3. **Cross-Frequency Attention:** Model how low-freq DXY modes predict high-freq copper modes (Gap 3)
4. **MVMD variant:** Use Multivariate VMD for joint decomposition preserving cross-channel relationships
5. **Generalization:** Apply to gold, oil, agricultural commodities to show framework generality
6. **Learnable VMD (VMDNet):** Make VMD end-to-end differentiable with GNN (frontier contribution)
7. **Alternative data integration:** Satellite imagery of mines, shipping AIS data, news sentiment as additional graph nodes

---

## 10. KEY REFERENCES (Must-Cite)

### Your Direct Competitors / Differentiation
- Sun et al. (2025) — MVGT for copper (only copper GNN paper)
- Zheng & Zhufu (2025) — DCT + cross-scale GNN for stocks (closest in finance)
- Ahmad et al. (2024, 2025) — VMD + GCN for traffic (closest VMD+GNN)
- Cheng & Tsai (2026) — MVMD + scale-aware GNN for wind (closest in spirit)

### Foundational Methods You Build On
- Dragomiretskiy & Zosso (2014) — VMD original paper (8,396 citations)
- Wu et al. (2020) — MTGNN (2,804 citations) — graph learning approach
- Cao et al. (2020) — StemGNN (979 citations) — spectral GNN baseline
- Yi et al. (2023) — FourierGNN — unified spectral-temporal GNN

### The Benchmark You Must Beat
- Liu et al. (2019/2020) — VMD + LSTM for non-ferrous metals (252-290 citations) — THE decomposition benchmark

### Copper-Specific Literature
- Hu et al. (2020) — LSTM-ANN-GARCH for copper volatility (204 citations)
- Liu et al. (2022) — Bayesian + wavelet + NN for copper (61 citations)
- Becerra et al. (2022) — SARIMA(X) for copper
- Li et al. (2023) — CNN-LSTM multi-factor for copper

### Methodology Support
- Feng et al. (2025) — VMDNet, temporal leakage-free VMD
- Cai et al. (2024) — MSGNet, multi-scale inter-series correlations (181 citations)
- Lee & Cho (2025) — H-ETE-GNN, regime-adaptive graphs
- Ying et al. (2019) — GNNExplainer (3,000+ citations)

---

*Blueprint compiled from 5 parallel research streams covering 150+ papers. All gaps verified through systematic search of arXiv, OpenAlex, Semantic Scholar, and Google Scholar as of April 2026.*
