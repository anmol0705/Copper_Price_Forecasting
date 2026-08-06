# Variational Mode Decomposition (VMD) for Copper Price Forecasting: Research Compendium

**Compiled: April 2026**

---

## Table of Contents

1. [Foundational VMD Theory](#1-foundational-vmd-theory)
2. [VMD in Financial Time Series Forecasting (2018-2026)](#2-vmd-in-financial-time-series-forecasting-2018-2026)
3. [Signal Decomposition + Deep Learning Combinations](#3-signal-decomposition--deep-learning-combinations)
4. [VMD + Graph Approaches -- The Key Research Gap](#4-vmd--graph-approaches----the-key-research-gap)
5. [Advanced Decomposition Methods Beyond Standard VMD](#5-advanced-decomposition-methods-beyond-standard-vmd)
6. [Technical Challenges](#6-technical-challenges)
7. [Summary of Research Gaps and Opportunities](#7-summary-of-research-gaps-and-opportunities)
8. [Key References](#8-key-references)

---

## 1. Foundational VMD Theory

### 1.1 The Original VMD Algorithm

**Seminal paper:** Dragomiretskiy, K. and Zosso, D. (2014). "Variational Mode Decomposition." *IEEE Transactions on Signal Processing*, 62(3), 531-544. DOI: 10.1109/TSP.2013.2288675. **Cited 8,396+ times** (top 0.01% of all papers by citation).

VMD decomposes a signal into K band-limited intrinsic mode functions (IMFs) u_k by solving a constrained variational optimization problem:

- **Objective:** Minimize the sum of bandwidths of all modes, where bandwidth is estimated via the analytic signal's Hilbert transform and frequency shifting to baseband.
- **Constraint:** The sum of all modes reconstructs the original signal.
- **Solution method:** Augmented Lagrangian with Alternating Direction Method of Multipliers (ADMM), solved in the spectral (Fourier) domain.

**Key parameters:**
- **K** -- number of modes to extract
- **alpha** -- bandwidth constraint / penalty parameter (balancing factor for data fidelity vs. bandwidth minimization)
- **tau** -- noise tolerance (Lagrangian update step)

### 1.2 VMD vs. EMD: Fundamental Differences

| Feature | EMD | VMD |
|---------|-----|-----|
| Approach | Recursive sifting (empirical) | Concurrent variational optimization |
| Theoretical foundation | Heuristic | Solid mathematical (variational calculus) |
| Mode extraction | Sequential (one at a time) | Simultaneous (all K modes) |
| Noise sensitivity | High | Low (robust to noise) |
| Mode mixing | Prone to mode mixing | Largely avoids mode mixing |
| End effects | Significant boundary artifacts | Minimal (spectral domain) |
| Sampling robustness | Sensitive | Robust |
| Parameters | None (fully adaptive) | K and alpha must be specified |
| Computational cost | Lower | Higher (iterative ADMM) |

---

## 2. VMD in Financial Time Series Forecasting (2018-2026)

### 2.1 VMD + LSTM

This is the most widely explored VMD + deep learning combination for price forecasting.

**Key papers:**

1. **Liu, Y., Yang, C., Huang, K., and Gui, W. (2019).** "Non-ferrous metals price forecasting based on variational mode decomposition and LSTM network." *Knowledge-Based Systems*, 188, 105006. **252 citations.**
   - Directly relevant: applies VMD-LSTM to non-ferrous metal prices (including copper as a category).
   - Decomposes price series into IMFs via VMD, trains separate LSTM models per mode, then aggregates.
   - Demonstrated significant improvement over standalone LSTM and EMD-LSTM.

2. **Niu, H., Xu, K., and Wang, W. (2020).** "A hybrid stock price index forecasting model based on variational mode decomposition and LSTM network." *Applied Intelligence*, 50(12), 4296-4309. **181 citations.**
   - Applied to HSI, SSE, S&P 500 indices.
   - VMD-LSTM outperformed EMD-LSTM, EEMD-LSTM, and standalone LSTM.

3. **Boadi, E. (2025).** "Bitcoin Price Forecasting Based on Hybrid Variational Mode Decomposition and Long Short Term Memory Network." arXiv:2510.15900.
   - Confirms VMD+LSTM superiority over standalone LSTM for cryptocurrency.

4. **Zeng, L., Hu, H., Tang, H., Zhang, X., and Zhang, D. (2024).** "Carbon emission price point-interval forecasting based on multivariate variational mode decomposition and attention-LSTM model." *Applied Soft Computing*, 111543. **42 citations.**
   - Uses MVMD (multivariate VMD) with attention-enhanced LSTM for carbon price.

5. **Xie, B., Shi, S., and Liu, W. (2025).** "Integrated Forecasting of Marine Renewable Power: An Adaptively Bayesian-Optimized MVMD-LSTM Framework." arXiv:2509.25226.
   - Bayesian optimization for automatic MVMD parameter tuning + LSTM.

### 2.2 VMD + CNN

1. **Zhang, G., Ren, T., and Yang, Y. (2020).** "A New Unified Deep Learning Approach with Decomposition-Reconstruction-Ensemble Framework for Time Series Forecasting." arXiv:2002.09695.
   - VMD + CNN + LSTM unified framework.
   - Proposes a general decomposition-reconstruction-ensemble (DRE) architecture.

2. **Zhao et al. (2023).** "Hybrid VMD-CNN-GRU-based model for short-term forecasting of wind power."
   - VMD + CNN + GRU hybrid achieving strong results in energy forecasting.

3. **Malakar, S., Goswami, S., Chakrabarti, A., and Ganguli, B. (2024).** "A Novel Denoising Technique and Deep Learning Based Hybrid Wind Speed Forecasting Model for Variable Terrain Conditions." arXiv:2408.15554.
   - VMD + Bidirectional LSTM with CNN components.

### 2.3 VMD + Transformer / Attention Architectures

This is a newer and rapidly growing area (2022-present):

1. **Xue, X., Li, S., and Wang, X. (2024).** "Enhanced forecasting of stock prices based on variational mode decomposition, PatchTST, and adaptive scale-weighted layer." arXiv:2408.16707.
   - VMD + PatchTST + Adaptive Scale-Weighted Layer (ASWL).
   - Applied to S&P 500, DJI, SSEC, FTSE indices.
   - Represents VMD integration with modern Transformer variants.

2. **Zhang, J. and Duan, H. (2023).** "Enhanced LFTSformer: A Novel Long-Term Financial Time Series Prediction Model Using Advanced Feature Engineering and the DS Encoder Informer Architecture." arXiv:2310.01884.
   - VMD + Maximal Information Coefficient + Informer (Transformer-based).

3. **Liu, C., Liu, D., and Mu, L. (2022).** "Improved Transformer Model for Enhanced Monthly Streamflow Predictions." *IEEE Access*, 10, 58240-58253.
   - VMD preprocessing + double-encoder Transformer with cross-attention.

4. **Putra, H.R.K., Yudistira, N., and Fatyanosa, T.N. (2024).** "Variational Mode Decomposition and Linear Embeddings are What You Need For Time-Series Forecasting." arXiv:2408.16122.
   - Provocative finding: VMD + simple linear embeddings can match complex deep models.

5. **Xu, Y. (2025).** "An improved wind power prediction via a novel wind ramp identification algorithm." arXiv:2502.12807.
   - VMD + Informer (Transformer-based) for wind power.

6. **VMD + Attention mechanism papers (2022-2025):**
   - Huang et al. (2022): VMD + IPSO-DBiLSTM + attention for load forecasting. *Applied Intelligence*, 53(10).
   - Lei et al. (2025): VMD + multi-head attention for demand forecasting. *Processes*, 13(2). Achieves 37% MAE reduction.
   - Fu et al. (2024): VMD + TCN-GRU + multi-head attention for PV forecasting. *Electronics*, 13(10). 55% RMSE reduction.
   - Gopalakrishnan et al. (2023): VMD + attention for intraday stock price forecasting.

### 2.4 VMD vs. EMD vs. EEMD vs. CEEMDAN -- Comparative Studies

**General consensus from the literature:**

| Method | Pros | Cons |
|--------|------|------|
| **EMD** (Huang et al., 1998) | No parameters needed; fully adaptive | Mode mixing; end effects; noise sensitivity; no theoretical basis |
| **EEMD** (Wu & Huang, 2009) | Reduces mode mixing via noise-assisted | Residual noise; ensemble size selection; computationally expensive |
| **CEEMDAN** (Torres et al., 2011) | Better noise cancellation than EEMD | Still some residual noise; computational cost; parameter sensitivity |
| **VMD** (Dragomiretskiy & Zosso, 2014) | Solid theory; concurrent extraction; noise-robust; no mode mixing | Requires K and alpha specification; higher computational cost |

**Key comparative findings:**

- **Zhou et al. (2021)**: Empirical Fourier Decomposition (EFD) as a non-recursive alternative that "overcomes the defect of mode mixing" in EMD, achieving 2-4 dB SNR improvement. *Mechanical Systems and Signal Processing*. 188 citations.
- **Hu et al. (2019)**: VMD-NWT shows "advantages of less calculation and simple implementation" versus CEEMDAN-WT, EEMD-WT methods. *IEEE Access*. 71 citations.
- **Niu et al. (2020)**: Direct comparison showing VMD-LSTM > EMD-LSTM > EEMD-LSTM for stock indices.
- **Liu et al. (2019)**: VMD-LSTM significantly outperforms EMD-LSTM and EEMD-LSTM for non-ferrous metal prices.

**VMD's advantages for financial forecasting specifically:**
1. Non-recursive nature avoids error accumulation across modes.
2. Concurrent mode extraction provides global optimization rather than greedy sequential extraction.
3. Better separation of trend, cyclical, and noise components in price data.
4. Spectral-domain solution reduces boundary artifacts critical for time series with finite windows.

### 2.5 VMD for Commodity / Metal Price Forecasting

**Directly relevant papers:**

1. **Liu et al. (2019)** -- VMD-LSTM for non-ferrous metals (see Section 2.1). **252 citations.** The most cited VMD paper for metal price forecasting.

2. **Liu, Q., Liu, M., Zhou, H., and Yan, F. (2022).** "A multi-model fusion based non-ferrous metal price forecasting." *Resources Policy*, 77, 102714.

3. **He, Z. and Huang, J. (2023).** "A novel non-ferrous metal price hybrid forecasting model based on data preprocessing and error correction." *Resources Policy*, 86, 104189.
   - Introduces error correction after initial VMD-based prediction.

4. **Huang, Y. and Deng, Y. (2020).** "A new crude oil price forecasting model based on variational mode decomposition." *Knowledge-Based Systems*, 213. **167 citations.**
   - VMD-based decompose-forecast-reconstruct for crude oil.

5. **Hu, Y., Ni, J., and Wen, L. (2020).** "A hybrid deep learning approach by integrating LSTM-ANN networks with GARCH model for copper price volatility prediction." *Physica A*, 557, 124907. **149 citations.**
   - Copper-specific, though uses LSTM-ANN-GARCH rather than VMD.
   - Establishes that hybrid decomposition approaches work for copper.

6. **Kriechbaumer, T., Angus, A., Parsons, D., and Rivas Casado, M. (2013).** "An improved wavelet-ARIMA approach for forecasting metal prices." *Resources Policy*, 39, 32-41. **172 citations.**
   - Wavelet decomposition (not VMD) for metals; provides baseline for decomposition approaches.

**Notable finding:** No paper was found that applies VMD specifically and exclusively to copper price forecasting. The closest is Liu et al. (2019) on "non-ferrous metals" which includes copper among others. **This represents a gap.**

### 2.6 VMD for Energy Price Forecasting

1. **Zhu, J., Wu, P., Chen, H., Liu, J., and Zhou, L. (2018).** "Carbon price forecasting model based on VMD and optimal combined model." *Physica A*, 519. **143 citations.**

2. **Sun, G., Chen, T., Wei, Z., Sun, Y., Zang, H., and Chen, S. (2016).** "A Carbon Price Forecasting Model Based on VMD and Spiking Neural Networks." *Energies*, 9(1). **139 citations.**

3. **Feng, W., Tao, R., Cartlidge, J., and Zheng, J. (2025).** "VMDNet: Temporal Leakage-Free Variational Mode Decomposition for Electricity Demand Forecasting." arXiv:2509.15394.
   - **Critical methodological contribution:** Addresses temporal leakage in VMD-based forecasting (see Section 6.6).

---

## 3. Signal Decomposition + Deep Learning Combinations

### 3.1 The Decompose-Predict-Ensemble (DPE) Framework

The standard pipeline in the literature follows three stages:

```
Raw Signal --> [Decomposition] --> IMF_1, IMF_2, ..., IMF_K
                                      |       |            |
                                   [Model_1] [Model_2] ... [Model_K]
                                      |       |            |
                                   Pred_1  Pred_2  ...  Pred_K
                                      \       |           /
                                       [Ensemble/Sum]
                                            |
                                      Final Prediction
```

### 3.2 Two-Stage Decomposition (Secondary Decomposition of Residual)

Several papers propose decomposing the high-frequency or residual component a second time:

1. **Wang, J., He, M., and Qiu, S. (2023).** "Two-Stage Decomposition Multi-Scale Nonlinear Ensemble Model with Error-Correction-Coupled Gaussian Process for Wind Speed Forecast." *Atmosphere*, 14(2), 395.
   - First stage: CEEMDAN decomposition.
   - Second stage: SSA (Singular Spectrum Analysis) refinement of residuals.
   - Error correction via Gaussian Process for interval prediction.

2. **Yan, H. and Tian, C. (2019).** "A novel two-stage forecasting model based on error factor and ensemble method for multi-step wind power forecasting." *Applied Energy*, 238, 368-383. **243 citations.**
   - Two-stage approach with error factors.

3. **Gui, Z., Li, H., Xu, S., and Chen, Y. (2023).** "A novel decomposed-ensemble time series forecasting framework: capturing underlying volatility information." arXiv:2310.08812.
   - Novel framework specifically designed to capture volatility through decomposition.

4. **Zhang, Z. et al. (2023).** "Implementing a new fully stepwise decomposition-based sampling technique for hybrid water level forecasting." arXiv:2309.10658.
   - Stepwise decomposition to avoid look-ahead bias.

**Common two-stage patterns:**
- VMD + VMD (secondary decomposition of high-frequency IMFs)
- VMD + EEMD (complementary decomposition of residual)
- CEEMDAN + SSA (secondary smoothing)
- VMD + wavelet (wavelet denoising of individual modes)

### 3.3 Ensemble Methods Across IMFs/Modes

**Mode recombination strategies in the literature:**

1. **Simple summation** -- Most common. Individual mode predictions are simply summed. Assumes perfect reconstruction. Used in Liu et al. (2019), Niu et al. (2020), Boadi (2025).

2. **Weighted summation** -- Learned or optimized weights for each mode's contribution:
   - Xue et al. (2024): Adaptive Scale-Weighted Layer (ASWL) learns mode weights.
   - Sun et al. (2020): AdaEnsemble approach for metro passenger flow.

3. **Nonlinear ensemble** -- A secondary model (often a simple neural network) learns to combine mode-level predictions:
   - Zhang et al. (2020): CNN + LSTM ensemble layer in the DRE framework.

4. **Selective mode forecasting** -- Not all modes are forecasted; some (e.g., noise-dominant high-frequency modes) are filtered:
   - Common in fault diagnosis; less explored in price forecasting.

### 3.4 Error Correction Mechanisms After Decomposition

1. **He, Z. and Huang, J. (2023).** Non-ferrous metal price hybrid forecasting with error correction. *Resources Policy*, 86.
   - Explicit error correction layer after VMD-based prediction.

2. **Wang et al. (2023).** Error-Correction-Coupled Gaussian Process for wind forecasting.
   - Gaussian process models the residual error from the decomposition-ensemble prediction.

3. **General pattern:** Train a secondary model on the residuals (actual - predicted) from the primary VMD-DL pipeline. Common secondary models: ARIMA, GP, simple MLP.

---

## 4. VMD + Graph Approaches -- The Key Research Gap

### 4.1 Existing VMD + GNN Papers

**This is a confirmed, significant research gap.** Across all databases searched (arXiv, OpenAlex, Semantic Scholar), only a handful of papers combine VMD with graph neural networks:

1. **Ahmad, O., Wesemann, L., Waschkowski, F., and Khalid, Z. (2024).** "Variational Mode-Driven Graph Convolutional Network for Spatiotemporal Traffic Forecasting." arXiv:2408.16191.
   - Uses VMD to decompose traffic time series, then applies GCN on each mode.
   - Traffic domain, not financial.

2. **Ahmad, O. and Khalid, Z. (2025).** "Robust and Noise-resilient Long-Term Prediction of Spatiotemporal Data Using Variational Mode Graph Neural Networks with 3D Attention." arXiv:2504.06660.
   - VMD + GCN + 3D Attention mechanism.
   - Introduces concept of "variational mode graph neural networks."

3. **Ahmad, O., Wesemann, L., Waschkowski, F., and Khalid, Z. (2025).** "Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded Variational Mode Decomposition." arXiv:2509.00703.
   - Deep unfolding of VMD combined with GNN.
   - Represents the most advanced VMD-GNN integration to date.

4. **Pei, Y., Huang, C., Shen, Y., and Ma, Y. (2022).** "An Ensemble Model with Adaptive Variational Mode Decomposition and Multivariate Temporal Graph Neural Network for PM2.5 Concentration Forecasting." *Sustainability*, 14(20).
   - AVMD + Multivariate Temporal GNN for air quality.

5. **Bao, X., Li, J., Wu, W., and Wang, T. (2025).** "Damage identification of offshore wind turbine blades based on graph neural networks and successive variational mode decomposition." *Ocean Engineering*, 340.
   - SVMD + GNN for structural health monitoring.

**Critical observation:** None of the existing VMD+GNN papers address financial or commodity price forecasting. All are in traffic, air quality, or structural engineering domains.

### 4.2 Multi-Scale Graph Construction Using Decomposed Signals

A promising but underexplored direction. Related work on multi-scale graphs (without VMD):

1. **Cai, W., Liang, Y., Liu, X., Feng, J., and Wu, Y. (2024).** "MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting." *Proceedings of AAAI*, 38(10). **181 citations.**
   - Uses frequency domain analysis to extract periodic patterns at multiple scales.
   - Applies adaptive graph convolution per scale.
   - **Key insight for our work:** Demonstrates that multi-scale graph construction improves forecasting.

2. **Chen, Y., Ding, F., and Zhai, L. (2022).** "Multi-scale Temporal Features Extraction Based Graph Convolutional Network." *Expert Systems with Applications*, 200. **63 citations.**
   - Multi-scale temporal feature extraction + GCN + attention.

3. **Rawal, K. and Ahmad, A. (2024).** "Mining Latent Patterns with Multi-Scale Decomposition for Electricity Demand and Price Forecasting using Modified Deep Graph Convolutional Neural Networks." *Sustainable Energy Grids and Networks*, 39.
   - **Directly relevant:** Multi-scale decomposition + deep GCN for price forecasting.

**Proposed gap to exploit:** No paper constructs graphs where:
- Nodes represent VMD modes of different variables (e.g., copper price mode 1, LME inventory mode 1, USD index mode 1, etc.)
- Edges capture cross-modal, cross-variable relationships at the same frequency scale
- A GNN learns inter-variable dependencies within each frequency band

### 4.3 Cross-Modal Relationships Between VMD Modes of Different Variables

This concept is essentially unexplored. The idea:
- Decompose multiple input variables (copper price, inventory, exchange rate, etc.) via VMD/MVMD.
- At each frequency scale k, construct a graph connecting the k-th modes across all variables.
- Learn which variables are correlated at which frequency scales.
- This enables frequency-aware multi-variate reasoning that no existing model achieves.

**Supporting evidence that this should work:**
- MSGNet (Cai et al., 2024) shows multi-scale graphs improve forecasting (181 citations).
- MVMD (Rehman & Aftab, 2019) enables joint multi-channel decomposition preserving inter-channel relationships.
- Ahmad et al. (2024-2025) demonstrate VMD+GCN is viable for spatiotemporal data.

---

## 5. Advanced Decomposition Methods Beyond Standard VMD

### 5.1 Successive VMD (SVMD)

**Foundational paper:** Nazari, M. and Sakhaei, S.M. (2020). "Successive variational mode decomposition." *Signal Processing*. **444 citations.**

- Extracts modes one at a time (like EMD) but using VMD's variational framework.
- Does not require pre-specifying K.
- Each step extracts one mode and passes the residual to the next iteration.
- Combines benefits of VMD's mathematical rigor with EMD's adaptive mode number determination.

**Applications:**
- Bao et al. (2025): SVMD + GNN for offshore wind turbine blade damage identification. *Ocean Engineering*.
- Guo et al. (2022): SVMD + energy position index for bearing fault diagnosis. *Sensors*.
- Ma et al. (2023): SVMD with dual-threshold correlation coefficient for maritime signal denoising. *Ocean Engineering*.
- Fu et al. (2025): SVMD + Informer for tunnel boring machine performance interval prediction. *Automation in Construction*.

### 5.2 Multivariate VMD (MVMD)

**Foundational paper:** Rehman, N. and Aftab, H. (2019). "Multivariate Variational Mode Decomposition." *IEEE Transactions on Signal Processing*.

- Extends VMD to multivariate/multichannel signals.
- Decomposes multiple related time series simultaneously, preserving cross-channel relationships.
- Critical for multi-variate financial forecasting where copper price co-moves with other variables.

**Applications in forecasting:**
- Ghanbari, E. and Avar, A. (2024): MVMD + LSTM for wind power forecasting. *Electrical Engineering*.
- Zeng et al. (2024): MVMD + Attention-LSTM for carbon price forecasting. *Applied Soft Computing*. 42 citations.
- Xie et al. (2025): Bayesian-optimized MVMD + LSTM for marine renewable energy.

**Applications in other domains:**
- Song et al. (2022): Self-Adaptive MVMD for bearing fault diagnosis. *IEEE Trans. Instrumentation and Measurement*.
- Sadiq et al. (2022): MVMD for motor imagery BCI classification. *IEEE Trans. Emerging Topics in CI*.

### 5.3 Adaptive VMD Variants

Methods for automatic parameter selection:

1. **Lian, J., Liu, Z., Wang, H., and Dong, X. (2018).** "Adaptive variational mode decomposition method for signal processing based on mode characteristic." *Mechanical Systems and Signal Processing*.
   - Adjusts parameters based on extracted mode characteristics iteratively.

2. **Xu, C., Yang, J., Zhang, T., Li, K., and Zhang, K. (2023).** "Adaptive parameter selection variational mode decomposition based on a novel hybrid entropy." *Measurement*, 217.
   - Hybrid entropy approach for automatic K and alpha selection.

3. **Xia, Y., Wang, W., and Li, X. (2024).** "Adaptive Parameter Selection VMD Based on Bayesian Optimization." *IEEE Access*, 12.
   - Bayesian optimization for VMD parameter tuning.

4. **Nassef, M.G.A., Hussein, T.M., and Mokhiamar, O. (2020).** "An adaptive VMD based on sailfish optimization and Gini index." *Measurement*.
   - Gini index + metaheuristic for K and alpha.

5. **Zhang et al. (2026).** VMD parameter optimization via Osprey-inspired Cauchy Sparrow Search Algorithm (OCSSA). *Frontiers in Signal Processing*.
   - Most recent: automated VMD parameter optimization achieving 97.5% accuracy at 20 dB SNR.

### 5.4 Time-Varying Filter EMD (TVF-EMD)

A competing approach to VMD that adapts filter characteristics over time:

1. **Zhang, X., Liu, Z., Miao, Q., and Wang, L. (2017).** "An Optimized Time Varying Filtering Based EMD with Grey Wolf Optimizer." *Journal of Sound and Vibration*. **136 citations.**

2. **Khelifi, R., Guermoui, M., Rabehi, A. et al. (2023).** "Short-Term PV Power Forecasting Using a Hybrid TVF-EMD-ELM Strategy." *International Transactions on Electrical Energy Systems*. **60 citations.**
   - TVF-EMD + Extreme Learning Machine achieving <4% NRMSE for PV power forecasting.

3. **Zhou, C., Xiong, Z., Bai, H. et al. (2022).** "Parameter-Adaptive TVF-EMD Feature Extraction Based on Improved GOA." *Sensors*.
   - Adaptive TVF-EMD using grasshopper optimization.

### 5.5 Other Advanced Decomposition Methods

- **Empirical Fourier Decomposition (EFD):** Zhou et al. (2021), *Mechanical Systems and Signal Processing*, 188 citations. Non-recursive approach overcoming mode mixing.
- **Short-time VMD (STVMD):** Jia et al. (2025). Uses STFT for non-stationary signal handling.
- **Complex VMD:** Cui et al. (2022). Extension to complex-valued signals.
- **Variational Kernel-Based 1-D CNN:** Mo et al. (2021). *IEEE Trans. Instrumentation and Measurement*. Combines variational methods with learnable CNN kernels.

### 5.6 VMDNet: Deep Learning-Native VMD (2025)

**Feng, W., Tao, R., Cartlidge, J., and Zheng, J. (2025).** "VMDNet: Temporal Leakage-Free Variational Mode Decomposition for Electricity Demand Forecasting." arXiv:2509.15394.

This is a cutting-edge contribution that:
- Integrates VMD decomposition into a trainable neural network (deep unfolding).
- Solves the temporal leakage problem inherent in standard VMD applied to forecasting.
- Makes VMD end-to-end differentiable as part of a deep learning pipeline.

**Ahmad et al. (2025):** "Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded Variational Mode Decomposition." arXiv:2509.00703.
- Another deep-unfolding VMD approach combined with GNN.

**Significance:** These represent the frontier -- making VMD a learnable layer rather than a fixed preprocessing step.

---

## 6. Technical Challenges

### 6.1 Optimal K (Number of Modes) Selection

This is the single most critical practical challenge with VMD. Methods proposed:

**A. Energy / frequency-based criteria:**
- **Minimum total mode aliasing energy** (Zhang et al., 2021; Lei et al., 2023): Select K that minimizes the total energy overlap between adjacent modes.
- **Center frequency convergence**: Monitor center frequencies as K increases; stop when new modes have center frequencies that converge to existing ones (indicating over-decomposition).
- **Energy loss coefficient**: Track the ratio of reconstructed signal energy to original; significant drop indicates too few modes.

**B. Correlation-based criteria:**
- **Dibaj et al. (2019)**: Minimize mean bandwidth across modes while keeping correlation coefficients between adjacent modes below a threshold. *Structural Health Monitoring*.
- **Correlation between adjacent modes**: When newly added modes are highly correlated with existing modes, K is too large.

**C. Metaheuristic optimization:**
- Bayesian optimization (Xia et al., 2024)
- Genetic algorithm (Li et al., 2020)
- Whale optimization (Jin et al., 2022)
- Sailfish optimization + Gini index (Nassef et al., 2020)
- Grey wolf optimization (Wang et al., 2022)
- Sparrow search algorithm with Osprey-inspired modifications (Zhang et al., 2026)

**D. Information-theoretic criteria:**
- Hybrid entropy measures (Xu et al., 2023)
- Permutation entropy monitoring across K values

**E. Scale-space approach:**
- **Wang, B. et al. (2025)**: "Improved VMD Based on Scale Space Representation." *Sensors*. Identifies K and initial center frequencies through peak detection in scale space.

**Practical guidance from the literature:**
- For financial time series: K = 3-8 is typical (trend + 1-2 cyclical + 1-2 high-frequency + noise)
- Over-decomposition (K too large): modes become correlated and redundant
- Under-decomposition (K too small): mode mixing persists, multiple frequency components in single mode

### 6.2 Alpha (Penalty Parameter) Selection

Less systematically studied than K. Key findings:

- **Large alpha** (e.g., 2000-5000): Produces narrower bandwidth modes, more frequency-separated. Better for signals with well-separated spectral content.
- **Small alpha** (e.g., 100-500): Produces wider bandwidth modes, more overlap allowed. Better for signals with close spectral components.
- **Common default:** alpha = 2000 is widely used as a starting point.
- **Joint optimization of K and alpha** is increasingly preferred using metaheuristic approaches (see Section 6.1C above).
- Alpha is treated as a hyperparameter optimized alongside K in most adaptive VMD frameworks.

### 6.3 Mode Mixing Detection and Mitigation

**Definition:** Mode mixing occurs when a single IMF contains signals from significantly different frequency bands, or when similar frequency content is distributed across multiple IMFs.

**Detection methods:**
- Spectral overlap analysis between adjacent modes
- Instantaneous frequency analysis (modes should have narrow-band IF)
- Cross-correlation between modes (should be low for well-separated modes)

**Mitigation approaches:**
1. **VMD inherently reduces mode mixing** compared to EMD due to its variational formulation.
2. **Quadratic penalty optimization** (Zhao et al., 2019): Improved penalty term in VMD to reduce mode mixing. *Mechanical Systems and Signal Processing*. 49 citations.
3. **Adaptive noise injection** (Cheng et al., 2019): Improved CEEMDAN with adaptive noise reduces mode mixing. *ISA Transactions*. 193 citations.
4. **SVMD approach**: By extracting modes sequentially but variationally, SVMD can better handle cases where the number of modes is uncertain.
5. **Initial center frequency guidance** (Jiang et al., 2018): Pre-specifying approximate center frequencies reduces mode mixing. *Journal of Sound and Vibration*.

### 6.4 End Effects / Boundary Conditions

**The problem:** VMD processes finite-length signals, and boundary treatment affects mode extraction quality, especially at signal endpoints.

**VMD's advantage:** Because VMD operates in the spectral (Fourier) domain with periodic boundary assumptions, it is inherently less susceptible to end effects than EMD (which uses envelope interpolation affected by endpoint values).

**Remaining issues:**
- For non-periodic financial time series, the periodic assumption can introduce spectral leakage.
- The most recent data points (most important for forecasting) are at the boundary.

**Mitigation strategies from the literature:**
- Mirror extension of the signal before VMD application
- Windowed VMD with overlap-and-add
- STVMD (Short-Time VMD) by Jia et al. (2025) explicitly addresses non-stationarity via time-frequency representation

### 6.5 Computational Cost Considerations

- VMD is iterative (ADMM), with complexity O(T log T) per iteration per mode (due to FFT), where T is signal length.
- Total cost: O(K * N_iter * T log T), where N_iter is typically 100-500 iterations.
- MVMD is significantly more expensive due to joint optimization across channels.
- For real-time or high-frequency trading: VMD preprocessing may be a bottleneck.
- **Mitigation:** Pre-compute decomposition offline; update incrementally for new data.

### 6.6 Temporal Leakage -- A Critical Methodological Issue

**The problem (often overlooked):** Standard VMD decomposes the entire signal at once, meaning future data points influence the decomposition of past data points. When used for forecasting, this introduces look-ahead bias.

**Specific mechanism:** The ADMM optimization uses all data points simultaneously to determine mode center frequencies and bandwidths. If applied to the full training+test dataset, information from the test period "leaks" into the training decomposition.

**Solutions:**
1. **Rolling/expanding window VMD:** Apply VMD only to data available up to each forecast point. Computationally expensive but correct.
2. **VMDNet (Feng et al., 2025):** "Temporal Leakage-Free Variational Mode Decomposition" -- integrates VMD into a neural network that explicitly prevents temporal leakage.
3. **Stepwise decomposition** (Zhang et al., 2023): Fully stepwise decomposition-based sampling to avoid look-ahead bias.

**Recommendation for our work:** Always use rolling-window VMD or VMDNet to ensure methodological validity.

---

## 7. Summary of Research Gaps and Opportunities

### 7.1 Primary Gap: VMD + GNN for Financial Forecasting

**Status:** Only ~5 papers combine VMD with GNN in any domain. Zero papers apply VMD+GNN to financial or commodity price forecasting. This is a wide-open research opportunity.

### 7.2 Secondary Gap: VMD Specifically for Copper Price Forecasting

**Status:** No paper applies VMD specifically to copper prices. Liu et al. (2019) addresses "non-ferrous metals" broadly with 252 citations, but copper-specific VMD analysis is absent.

### 7.3 Tertiary Gap: Multi-Scale Cross-Variable Graph Construction via VMD

**Status:** No paper constructs frequency-band-specific graphs connecting VMD modes of different variables. MSGNet (2024, 181 citations) demonstrates multi-scale graphs work for forecasting but does not use VMD decomposition. This is a novel architectural contribution.

### 7.4 Additional Opportunities

1. **MVMD + GNN:** Using multivariate VMD to jointly decompose copper price and its drivers, then constructing frequency-specific graphs, is completely unexplored.

2. **Learnable VMD (VMDNet/deep unfolding) + GNN:** Making VMD parameters trainable end-to-end with a GNN is at the absolute frontier (only Ahmad et al., 2025 approximates this, for traffic).

3. **Error correction after VMD-GNN:** Combining the decomposition-ensemble approach with post-hoc error correction (He & Huang, 2023) in a GNN framework.

4. **Temporal leakage-aware VMD for copper:** Properly handling the leakage issue (Section 6.6) would distinguish our work methodologically from the majority of VMD forecasting papers.

5. **Adaptive K selection for financial time series:** Most K-selection methods are developed for mechanical fault diagnosis. Financial time series have different spectral characteristics requiring domain-specific K selection criteria.

---

## 8. Key References

### Foundational

- Dragomiretskiy, K. and Zosso, D. (2014). Variational Mode Decomposition. *IEEE Trans. Signal Processing*, 62(3), 531-544. [8,396 citations]
- Nazari, M. and Sakhaei, S.M. (2020). Successive Variational Mode Decomposition. *Signal Processing*. [444 citations]
- Rehman, N. and Aftab, H. (2019). Multivariate Variational Mode Decomposition. *IEEE Trans. Signal Processing*.
- Huang, N.E. et al. (1998). The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and Non-Stationary Time Series Analysis. *Proc. Royal Society A*.

### VMD + Deep Learning for Price Forecasting

- Liu, Y. et al. (2019). Non-ferrous metals price forecasting based on VMD and LSTM. *Knowledge-Based Systems*, 188. [252 citations]
- Niu, H. et al. (2020). Hybrid stock price index forecasting with VMD and LSTM. *Applied Intelligence*, 50(12). [181 citations]
- Huang, Y. and Deng, Y. (2020). Crude oil price forecasting based on VMD. *Knowledge-Based Systems*, 213. [167 citations]
- Hu, Y. et al. (2020). LSTM-ANN-GARCH for copper price volatility. *Physica A*, 557. [149 citations]
- Zhu, J. et al. (2018). Carbon price forecasting with VMD. *Physica A*, 519. [143 citations]
- Xue, X. et al. (2024). VMD + PatchTST + ASWL for stock forecasting. arXiv:2408.16707.
- Zeng, L. et al. (2024). MVMD + Attention-LSTM for carbon price. *Applied Soft Computing*. [42 citations]

### VMD + GNN (Sparse Literature)

- Ahmad, O. et al. (2024). Variational Mode-Driven Graph Convolutional Network for Traffic. arXiv:2408.16191.
- Ahmad, O. and Khalid, Z. (2025). Variational Mode Graph Neural Networks with 3D Attention. arXiv:2504.06660.
- Ahmad, O. et al. (2025). Adaptive Deep-Unfolded VMD for Spatiotemporal Forecasting. arXiv:2509.00703.
- Pei, Y. et al. (2022). AVMD + Multivariate Temporal GNN for PM2.5. *Sustainability*, 14(20).
- Bao, X. et al. (2025). GNN + SVMD for wind turbine damage. *Ocean Engineering*, 340.

### Multi-Scale Graphs for Time Series

- Cai, W. et al. (2024). MSGNet: Multi-Scale Inter-Series Correlations. *AAAI*, 38(10). [181 citations]
- Chen, Y. et al. (2022). Multi-scale Temporal Features + GCN. *Expert Sys. with Applications*, 200. [63 citations]
- Rawal, K. and Ahmad, A. (2024). Multi-Scale Decomposition + Deep GCN for Price Forecasting. *Sustainable Energy Grids and Networks*, 39.

### Two-Stage Decomposition and Error Correction

- Yan, H. and Tian, C. (2019). Two-stage forecasting with error factor. *Applied Energy*, 238. [243 citations]
- Wang, J. et al. (2023). Two-stage CEEMDAN-SSA with error-correction GP. *Atmosphere*, 14(2).
- He, Z. and Huang, J. (2023). Non-ferrous metal price forecasting with error correction. *Resources Policy*, 86.

### Adaptive VMD and Parameter Selection

- Lian, J. et al. (2018). Adaptive VMD based on mode characteristics. *Mech. Sys. Signal Processing*.
- Xu, C. et al. (2023). Hybrid entropy for adaptive VMD parameters. *Measurement*, 217.
- Xia, Y. et al. (2024). Bayesian optimization for VMD parameters. *IEEE Access*, 12.
- Dibaj, A. et al. (2019). Fine-tuned VMD with adaptive indices. *Structural Health Monitoring*.
- Zhang, Y. et al. (2021). Minimum total mode aliasing energy for K selection. *Environmental Sci. and Pollution Res.*

### Advanced Decomposition Methods

- Zhou, W. et al. (2021). Empirical Fourier Decomposition. *Mech. Sys. Signal Processing*. [188 citations]
- Zhang, X. et al. (2017). TVF-EMD with Grey Wolf Optimizer. *J. Sound and Vibration*. [136 citations]
- Jia, H. et al. (2025). Short-time VMD (STVMD). arXiv.
- Feng, W. et al. (2025). VMDNet: Temporal Leakage-Free VMD. arXiv:2509.15394.

### Temporal Leakage and Methodology

- Feng, W. et al. (2025). VMDNet addresses temporal leakage in VMD-based forecasting.
- Zhang, Z. et al. (2023). Stepwise decomposition-based sampling for bias-free forecasting.

---

*This document synthesizes findings from searches across arXiv, OpenAlex, and related academic databases. All citation counts are approximate and reflect data available as of April 2026.*
