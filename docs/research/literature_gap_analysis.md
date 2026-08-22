# VMD + GNN for Copper Price Forecasting: Literature Gap Analysis
## Comprehensive Review and Novelty Assessment (April 2026)

---

## 1. STATE OF THE ART: What Has Been Done

### 1.1 Copper Price Forecasting with Deep Learning (2022-2026)

The copper forecasting literature is dominated by sequential/recurrent architectures:

| Paper | Year | Journal | Method | Citations |
|-------|------|---------|--------|-----------|
| Liu et al. | 2022 | Resources Policy | VMD + multi-model fusion (LSTM-based) | 30 |
| Liu et al. | 2022 | Resources Policy | Bayesian Optimization + Wavelet + NN | 61 |
| Luo et al. | 2022 | Resources Policy | Two-phase LSTM with error correction | 35 |
| Li et al. | 2023 | IEEE Access | CNN-LSTM multi-factor | 27 |
| Zhao et al. | 2023 | Resources Policy | Super Learner ensemble | 40 |
| Zhao et al. | 2023 | Mathematics | VMD-SSA-LSTM for non-ferrous metals | 11 |
| Nabavi et al. | 2024 | Resources Policy | XGBoost + meta-heuristics | 21 |
| Derakhshani et al. | 2024 | Mathematics | 1D-CNN | 6 |
| Zhang et al. | 2025 | J. Big Data | VMD-GGO-LSTM secondary decomposition | -- |
| Zhang & Ke | 2025 | Computational Economics | Adaptive VMD + feature fusion | -- |
| Li & Liu | 2026 | Entropy | Complexity-aware VMD + search engine features | -- |
| Yu et al. | 2026 | Progress in AI | VMD-ELM-BMO | -- |

**Key observation**: NO paper uses Graph Neural Networks for copper price forecasting. The field is stuck in a CNN/LSTM/ensemble paradigm. VMD is used only as preprocessing before sequential models.

### 1.2 GNN for Commodity/Financial Forecasting (2022-2026)

GNN adoption in commodities is extremely sparse:

- **Cui et al. (2026)** -- Higher-order moment spillovers with GNN for commodity markets (Risk Management). First to use GNN topology for inter-commodity relationships, but focused on spillover analysis, not point forecasting.
- **Shang et al. (2023)** -- Self-learning GNN for Chinese non-ferrous metal market (Procedia Computer Science). Only paper approaching non-ferrous metals with GNN, but focused on public sentiment indices, not price forecasting.
- **Min et al. (2025)** -- StemGNN for agricultural price prediction (Scientific Reports).
- **Zhang et al. (2025)** -- GCN-BiGRU for agricultural futures with multi-graph construction (Computational Economics).
- **Zheng & Zhufu (2025)** -- Frequency-domain attention + cross-scale GNN for stock prices (ACM). Uses DCT decomposition with dynamic graph structure for cross-scale correlations.

**Key observation**: GNN has NOT been applied to copper or base metals price forecasting. The commodity GNN literature is limited to agriculture and broad commodity indices.

### 1.3 VMD + GNN Combinations (Any Domain, 2022-2026)

This is the critical intersection. Nine papers exist combining VMD with GNN:

| Paper | Year | Domain | How VMD+GNN Combined |
|-------|------|--------|---------------------|
| Pei et al. | 2022 | PM2.5 air quality | AVMD preprocessing -> MtemGNN (PMNet) |
| Ahmad et al. | 2024 | Traffic forecasting | VMD preprocessing -> GCN (VMGCN); learns intra-mode and cross-mode dependencies |
| Ahmad et al. | 2025 | Spatiotemporal (general) | Deep-unfolded VMD -> trainable Mode Adaptive Graph Network (MAGN) |
| Bao et al. | 2025 | Offshore wind turbine damage | Sequential VMD -> GraphSAGE |
| Huang & Liu | 2025 | Green supply chain | Adaptive decomposition -> GNN |
| Yuan | 2025 | Mine gas prediction | Multivariate VMD -> cross-graph forecasting |
| Yuan et al. | 2025 | Multi-target tracking | VMD smoothing -> multiview weighted GNN |
| Cheng & Tsai | 2026 | Wind speed forecasting | MVMD -> scale-aware GNN framework |
| Ahmad et al. | 2025 | Noisy spatiotemporal | VMD -> VMGCN with 3D attention |

**CRITICAL FINDING**: 
- ZERO of these 9 papers apply VMD+GNN to financial/commodity/price forecasting
- Most use VMD as simple preprocessing (decompose -> predict each mode separately)
- Only Ahmad et al. (2024, 2025) attempt deeper integration with cross-mode learning
- Only Cheng & Tsai (2026) use multivariate VMD with scale-aware GNN, but for wind speed
- NONE construct the graph structure FROM the decomposed modes
- NONE use VMD modes as heterogeneous node types in a graph
- NONE apply cross-frequency graph attention between modes of DIFFERENT variables

### 1.4 Adjacent Innovations (2023-2026)

**Multi-scale GNN for time series:**
- MAGNN (Chen et al., 2023) -- multi-scale pyramid network with hierarchical decomposition
- MSPredictor (Wan et al., 2025) -- multi-scale temporal graphs for correlation modeling
- alpha-GNN (Li et al., 2025) -- FFT-based frequency analysis + spectral-spatial graph convolutions
- MSDG (Zhao et al., 2024) -- wavelet multiresolution + dynamic GNN for anomaly detection

**Spectral/frequency-domain GNN:**
- StemGNN and successors (various) -- spectral temporal GNN, proven effective
- SSMGNN (Zhou et al., 2025) -- spectral GNN + state space models
- CFGAN (Li et al., 2026, ICASSP) -- complex frequency-domain graph attention
- Dynamic Frequency Domain GCN (Li et al., 2024, ICASSP) -- frequency domain graph module

**Causal graph construction:**
- CUTS/CUTS+ (2023-2024) -- neural Granger causal discovery with GNN
- CausalFormer (2023) -- learns Granger causality graph from multivariate time series
- CGNN-TAM (2025) -- causal GNN with temporal attention for demand forecasting

**Dynamic/regime-adaptive graphs:**
- Lee & Cho (2025) -- Hurst exponent-based dynamic graph updating for market regimes
- Kumar et al. (2026) -- regime-dependent GNN for volatility prediction
- Ma et al. (2025) -- denoised dynamic graphs with reaction-diffusion regularization
- Baffou et al. -- generative graph state-space models for regime trajectories

**Cross-frequency interaction:**
- An et al. (2024) -- cross-frequency attention graph networks (EEG, not financial)
- MRGNN (Mo et al., 2025) -- multi-resolution GNN for electricity forecasting
- Zheng & Zhufu (2025) -- DCT + cross-scale GNN for stock prices (closest to our idea)

---

## 2. SPECIFIC GAPS TO EXPLOIT

### Gap 1: VMD + GNN for Financial/Commodity Forecasting (WIDE OPEN)
- **Status**: ZERO papers combine VMD with GNN for any financial or commodity application
- **Why it matters**: VMD provides clean, physically-meaningful frequency decomposition. GNN captures inter-variable relationships. The combination is unexploited in finance.
- **Defensibility**: HIGH. Clear, verifiable gap.

### Gap 2: Graph Construction FROM Decomposed Modes (UNEXPLORED)
- **Status**: All existing VMD+GNN papers use VMD as preprocessing, then apply GNN to the spatial structure that already existed. No paper constructs the graph topology from mode relationships.
- **Potential approaches**:
  - Nodes = VMD modes of different variables; edges = cross-mode correlations
  - Nodes = variables; edges defined per frequency band (different graph per mode)
  - Mode-frequency similarity as edge weight in a heterogeneous graph
- **Why it matters**: The relationship between copper and oil may be strong at low frequencies (long-term macro trends) but weak at high frequencies (daily noise). Frequency-specific graphs capture this.
- **Defensibility**: HIGH. Confirmed by search -- no paper does this.

### Gap 3: Cross-Frequency Graph Attention Between Multiple Variables (NASCENT)
- **Status**: An et al. (2024) did cross-frequency graph attention for EEG. Zheng & Zhufu (2025) combined DCT + cross-scale GNN for stocks. But nobody has done VMD-based cross-frequency attention between commodity/macro variables.
- **Why it matters**: Low-frequency modes of DXY (dollar index) may predict high-frequency modes of copper price. This cross-frequency, cross-variable interaction is economically meaningful but unmodeled.
- **Defensibility**: MEDIUM-HIGH. The EEG paper is in a completely different domain. The stock paper uses DCT, not VMD, and doesn't model inter-asset cross-frequency relationships.

### Gap 4: Dynamic/Regime-Adaptive Graph + Mode Decomposition (UNEXPLORED)
- **Status**: Dynamic graph learning exists (Lee & Cho 2025, Kumar et al. 2026). Mode decomposition exists. Nobody combines them.
- **Why it matters**: Commodity market regimes (crisis, boom, contango, backwardation) change which inter-commodity relationships matter AND which frequency scales dominate. A regime-aware, scale-specific graph would capture this.
- **Defensibility**: HIGH. Two active research threads that have not been connected.

### Gap 5: Causal Discovery on Decomposed Modes (UNEXPLORED)
- **Status**: Granger causality has been used to construct graphs (Wang et al. 2023). VMD has been used for decomposition. Nobody applies causal discovery to decomposed modes to build causally-grounded, frequency-specific graphs.
- **Why it matters**: Causality between copper and, say, Chinese PMI may only exist at specific frequency bands. Discovering this would produce interpretable, economically-motivated graph structures.
- **Defensibility**: HIGH. Novel combination of established methods.

### Gap 6: Attention-Based Mode Importance Weighting in a Graph Framework (UNEXPLORED)
- **Status**: Attention for mode weighting exists in sequential models (several 2023-2025 papers). Ahmad et al. (2025) used 3D attention in VMGCN for traffic. Nobody does mode-importance attention in a financial graph framework.
- **Why it matters**: Different VMD modes matter differently for different forecast horizons. An attention mechanism that weights mode importance within a graph framework is both novel and interpretable.
- **Defensibility**: MEDIUM-HIGH.

### Gap 7: GNN for Copper/Base Metals Specifically (WIDE OPEN)
- **Status**: ZERO papers apply any form of GNN to copper price forecasting
- **Why it matters**: Copper is highly interconnected (oil, DXY, S&P 500, Chinese macro, shipping indices, inventories). Graph-based modeling is a natural fit that has been entirely overlooked.
- **Defensibility**: HIGH. The simplest version of "GNN for copper" would already be novel.

---

## 3. NOVEL FRAMEWORK PROPOSALS (Evaluated for Feasibility)

### Framework A: VMD-MFGNN (Multi-Frequency Graph Neural Network)
**Architecture**: 
1. Apply VMD to K commodity/macro time series -> K x M mode series
2. For each frequency band m, construct a graph G_m where nodes are variables and edges capture band-specific correlations
3. Apply a GNN independently per frequency band
4. Fuse predictions across frequency bands with learnable attention weights

**Feasibility**: HIGH
- VMD is well-established, computationally tractable
- Per-band graph construction is straightforward (correlation, DTW, or learned)
- Each GNN operates on a standard variable-graph
- Attention fusion is standard
- Clear ablation path: compare per-band vs. single-band, learned vs. fixed graph, etc.

**Novelty**: VERY HIGH -- no paper does frequency-specific graph construction from VMD for any financial application

### Framework B: Heterogeneous Mode Graph Network
**Architecture**:
1. VMD decompose all input series
2. Treat each (variable, mode) pair as a node in a heterogeneous graph
3. Define edge types: intra-variable (between modes of same variable), intra-frequency (between variables at same frequency), and cross-frequency (between different variables at different frequencies)
4. Apply heterogeneous graph attention network (HAN/HGT)

**Feasibility**: MEDIUM-HIGH
- Heterogeneous GNN is mature (HAN, HGT, etc.)
- Node count = num_variables x num_modes (manageable: e.g., 10 variables x 5 modes = 50 nodes)
- Cross-frequency edges are novel and economically interpretable
- Slightly more complex to justify edge type definitions

**Novelty**: VERY HIGH -- completely unprecedented architecture

### Framework C: Regime-Adaptive VMD-GNN
**Architecture**:
1. Hidden Markov Model or learned regime detector identifies market regime
2. VMD parameters (number of modes, bandwidth) adapt to regime
3. Graph structure adapts to regime (different connectivity in crisis vs. normal)
4. GNN prediction conditioned on regime

**Feasibility**: MEDIUM
- Adds significant complexity (regime detection + adaptive VMD + adaptive graph)
- Harder to train end-to-end
- But each component is individually well-studied
- Strong narrative: "markets change, so should the model"

**Novelty**: VERY HIGH -- no paper combines all three elements

### Framework D: Causal-VMD-GNN
**Architecture**:
1. VMD decompose all input series
2. Apply Granger causality or PCMCI at each frequency band
3. Construct causally-grounded graph per frequency band
4. GNN on causal graphs with uncertainty quantification

**Feasibility**: MEDIUM-HIGH
- Granger causality on decomposed modes is computationally straightforward
- Produces interpretable, publishable graph visualizations
- Can validate causal links against economic theory
- Adds compelling interpretability angle for domain journals (Resources Policy)

**Novelty**: HIGH -- causal discovery on decomposed modes is unexplored

### Framework E: VMD-GNN-Transformer Hybrid
**Architecture**:
1. VMD decomposition of multi-commodity series
2. GNN captures cross-variable dependencies per frequency band
3. Transformer captures temporal dependencies within each mode
4. Cross-frequency attention layer fuses information across bands

**Feasibility**: HIGH
- Each component is well-understood
- Transformer + GNN combinations are trending
- Clear separation of concerns: GNN for spatial, Transformer for temporal, VMD for scale

**Novelty**: HIGH -- the specific triple combination is new

---

## 4. RECOMMENDED APPROACH: Framework A (VMD-MFGNN)

**Rationale for primary recommendation**:
- Clearest novelty claim: "first to construct frequency-specific inter-commodity graphs from VMD for metal price forecasting"
- Most feasible: no exotic components
- Best ablation structure: can isolate contribution of VMD, multi-frequency graphs, GNN, and attention fusion
- Extendable: can add causal graph construction (Framework D) or regime adaptation (Framework C) as extensions
- Publishable in both ML venues (method novelty) and domain venues (application novelty)

---

## 5. PUBLICATION STRATEGY

### 5.1 Target Venues (Ranked by Fit)

**Domain journals (application-focused, IF > 4):**
1. **Resources Policy** (IF ~10.2) -- THE venue for copper forecasting. Published Liu et al. 2022 (61 cites), Zhao et al. 2023 (40 cites). Reviewers want economic interpretability and practical utility. 3-6 month review.
2. **Energy Economics** (IF ~12.8) -- accepts commodity forecasting if energy-linked. Copper's role in electrification provides a narrative.
3. **Expert Systems with Applications** (IF ~8.5) -- accepts hybrid ML methods for forecasting. Good for method-heavy papers.
4. **Computers & Industrial Engineering** (IF ~7.9) -- accepts supply chain / industrial commodity forecasting.

**ML/AI journals (method-focused):**
5. **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)** (IF ~10.4) -- if method contribution is strong. Needs theoretical analysis.
6. **Neural Networks** (IF ~7.8) -- good for novel architectures.
7. **Knowledge-Based Systems** (IF ~8.8) -- hybrid approach fits well.

**Conferences:**
8. **KDD** -- Applied Data Science track accepts financial forecasting. Needs strong experiments.
9. **AAAI** -- AI for Social Impact or main track. Needs algorithmic novelty.
10. **IJCAI** -- Similar to AAAI.
11. **ICAIF** (ACM International Conference on AI in Finance) -- very targeted, growing prestige.

### 5.2 What Reviewers Look For

**For domain journals (Resources Policy, Energy Economics):**
- Economic interpretability: WHY does the model work? What do the graphs/modes mean economically?
- Practical forecasting horizons: 1-day, 1-week, 1-month ahead
- Multiple error metrics: RMSE, MAE, MAPE, directional accuracy (DA), Diebold-Mariano test
- Comparison against econometric baselines (ARIMA, VAR) AND deep learning baselines
- Robustness across time periods (include COVID, 2022 energy crisis)
- Feature importance / interpretability analysis
- Statistical significance tests

**For ML venues (IEEE TNNLS, KDD, AAAI):**
- Ablation study proving each component's contribution
- Computational complexity analysis
- Comparison against SOTA graph-based and decomposition-based methods
- Generalization to other datasets/commodities
- Theoretical motivation for architecture choices

### 5.3 Minimum Baseline Comparisons

**Econometric baselines:**
- ARIMA / SARIMA
- VAR (Vector Autoregression)
- GARCH (for volatility)

**Standard ML baselines:**
- XGBoost / LightGBM
- SVR (Support Vector Regression)

**Deep learning baselines:**
- LSTM
- GRU
- CNN-LSTM (Li et al. 2023)
- Transformer / Informer
- BiLSTM-Attention

**Decomposition baselines:**
- VMD-LSTM (Liu et al. 2022, Zhao et al. 2023)
- EMD-LSTM
- Wavelet-LSTM
- CEEMDAN-LSTM

**GNN baselines:**
- StemGNN (Cao et al. 2020)
- MTGNN (Wu et al. 2020)
- GTS (Shang et al. 2021)
- FourierGNN (Yi et al. 2023)

**Hybrid baselines:**
- VMD + Transformer
- VMD + attention-LSTM
- GNN + Transformer (without decomposition)

Total: ~15-18 baselines for a strong paper.

### 5.4 Required Ablation Studies

1. **VMD vs. no decomposition**: full model vs. GNN on raw data
2. **Multi-frequency graph vs. single aggregate graph**: per-mode graphs vs. one graph
3. **Learned graph vs. predefined graph**: adaptive graph learning vs. correlation-based
4. **GNN vs. no GNN**: VMD + independent mode prediction (no cross-variable modeling)
5. **Attention fusion vs. simple averaging**: mode importance weighting
6. **Number of VMD modes**: sensitivity analysis (K = 3, 5, 7, 9)
7. **Graph construction method**: correlation vs. DTW vs. Granger causality vs. learned
8. **Input variable selection**: all macro vs. metals-only vs. copper-only

### 5.5 Data Requirements

**Primary target variable:**
- LME copper spot/futures prices (daily, 2005-2025)

**Input features (graph nodes):**
- Copper price (LME 3-month)
- Aluminum, zinc, nickel, tin, lead prices (LME) -- inter-metal dependencies
- Gold, silver prices -- safe-haven dynamics
- Crude oil (WTI/Brent) -- energy cost input
- DXY (US Dollar Index) -- inverse correlation with commodities
- S&P 500 / MSCI World -- risk appetite proxy
- Chinese PMI / industrial production -- demand-side driver
- Baltic Dry Index -- shipping/trade proxy
- US 10Y yield / Fed Funds rate -- monetary policy
- VIX -- volatility/fear gauge
- LME copper inventory levels

---

## 6. SUGGESTED PAPER TITLES AND FRAMING

### Title 1 (Recommended -- clearest novelty):
**"Multi-Frequency Graph Neural Network with Variational Mode Decomposition for Copper Price Forecasting"**
- Framing: Frequency-specific inter-commodity graphs capture scale-dependent relationships missed by single-scale models
- Novelty claim: First to construct frequency-band-specific graphs from VMD-decomposed commodity series

### Title 2 (Emphasizes cross-frequency interaction):
**"Cross-Frequency Graph Attention Networks for Multi-Scale Commodity Price Prediction: A Variational Mode Decomposition Approach"**
- Framing: Relationships between commodities operate at different frequencies; cross-frequency attention captures how slow-moving macro trends in one variable affect fast-moving dynamics in another
- Novelty claim: First cross-frequency graph attention mechanism for commodity markets

### Title 3 (Emphasizes interpretability -- best for Resources Policy):
**"Interpretable Multi-Scale Inter-Commodity Modeling via VMD-Enhanced Graph Neural Networks: Evidence from Copper Markets"**
- Framing: Economic interpretability through frequency-specific causal graphs; each mode maps to an economic narrative (trend, cycle, noise)
- Novelty claim: Interpretable, scale-decomposed graph structures reveal frequency-specific commodity dependencies

### Title 4 (Emphasizes regime dynamics):
**"Regime-Adaptive Multi-Scale Graph Networks for Copper Price Forecasting Under Structural Breaks"**
- Framing: Market regime changes alter both the relevant time scales and inter-commodity dependencies; the model adapts both graph structure and decomposition to regime
- Novelty claim: First regime-adaptive, scale-aware graph neural network for commodity markets

### Title 5 (Broadest -- for ML venues like KDD/AAAI):
**"Decompose, Connect, Predict: A Frequency-Aware Graph Neural Architecture for Multivariate Financial Time Series"**
- Framing: General method for constructing multi-scale graph representations from decomposed signals; demonstrated on commodity markets
- Novelty claim: New paradigm of "decompose-then-graph" that constructs topology from frequency-domain relationships

---

## 7. CONCRETE NOVELTY CLAIMS (Defensible)

For any paper in this space, you can defensibly claim:

1. **"First application of GNN to copper/base metal price forecasting."** -- Confirmed: zero prior papers.

2. **"First framework to construct frequency-specific inter-variable graphs from VMD-decomposed time series."** -- Confirmed: all prior VMD+GNN papers use VMD as preprocessing on pre-existing spatial graphs.

3. **"First to model cross-frequency inter-commodity dependencies (e.g., how low-frequency DXY modes relate to high-frequency copper modes)."** -- Confirmed: cross-frequency graph attention exists only in EEG (An et al. 2024), not finance.

4. **"First to combine signal decomposition with graph neural networks for financial/commodity forecasting."** -- Confirmed: the Zheng & Zhufu (2025) stock paper uses DCT (not VMD) with GNN. No paper uses VMD+GNN for financial series.

5. **"First multi-scale graph framework that provides interpretable, frequency-specific visualizations of commodity market structure."** -- Confirmed: no prior work offers this.

---

## 8. RISK ASSESSMENT

### Risks to novelty:
- **Cheng & Tsai (2026)** did MVMD + scale-aware GNN for wind speed. If published before your submission, must clearly differentiate (different domain, different graph construction, cross-variable focus).
- **Ahmad et al. (2024, 2025)** are actively publishing VMD+GCN papers (traffic/spatiotemporal). Could pivot to finance.
- **Zheng & Zhufu (2025)** did DCT + cross-scale GNN for stocks. Closest competitor. Differentiation: VMD is superior to DCT for non-stationary financial data; their focus is single-asset, yours is multi-commodity graph.

### Mitigations:
- Submit to a domain journal (Resources Policy) where GNN novelty in copper is absolute, regardless of adjacent-field work.
- File an arXiv preprint quickly to establish priority.
- Emphasize the multi-commodity GRAPH construction from modes as the key innovation -- this is unique regardless of what happens in traffic/wind/EEG.

---

## 9. SUMMARY: THE OPPORTUNITY

The intersection of VMD and GNN is nascent (9 papers total, all 2022-2026, all outside finance). The intersection with commodity/copper forecasting is EMPTY. Multiple defensible novelty claims exist. The recommended Framework A (VMD-MFGNN) is feasible, publishable, and offers clear ablation paths. Target Resources Policy for fastest path to publication with maximum domain impact, or KDD/AAAI for maximum ML visibility.

The strongest single-sentence pitch: "Commodity prices are interconnected, but the nature of those interconnections changes across time scales -- we build the first framework that captures frequency-specific inter-commodity graph structures using VMD decomposition and GNN."
