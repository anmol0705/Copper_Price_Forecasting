# Graph Neural Networks for Financial and Commodity Price Forecasting
## Comprehensive Literature Review (2020-2026)

Compiled April 2026. This review covers GNN architectures, graph construction methods, commodity-specific applications, key innovations, identified gaps, and implementation details relevant to copper price forecasting.

---

## Table of Contents
1. [Foundational GNN Architectures for Time Series](#1-foundational-gnn-architectures-for-time-series)
2. [GNN Architectures for Financial Forecasting](#2-gnn-architectures-for-financial-forecasting)
3. [Graph Construction Methods in Finance](#3-graph-construction-methods-in-finance)
4. [GNN for Commodity Markets](#4-gnn-for-commodity-markets)
5. [Spectral vs Spatial Graph Convolutions in Finance](#5-spectral-vs-spatial-graph-convolutions-in-finance)
6. [Dynamic Graph Construction Methods](#6-dynamic-graph-construction-methods)
7. [Temporal Decomposition Combined with GNN](#7-temporal-decomposition-combined-with-gnn)
8. [Interpretability Methods for GNN in Finance](#8-interpretability-methods-for-gnn-in-finance)
9. [Implementation Frameworks and Technical Details](#9-implementation-frameworks-and-technical-details)
10. [Key Innovations and Gaps](#10-key-innovations-and-gaps)
11. [Consolidated Paper Table](#11-consolidated-paper-table)

---

## 1. Foundational GNN Architectures for Time Series

These are the foundational works that established GNN-based multivariate time series forecasting, upon which all financial applications build.

### 1.1 MTGNN -- Connecting the Dots (Wu et al., KDD 2020)

- **Title**: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
- **Authors**: Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi Zhang
- **Venue**: KDD 2020 | arXiv: 2005.11650
- **Citations**: ~2,804

**Architecture**:
- **Graph Learning Module**: Learns a directed graph adjacency matrix via node embeddings without predefined structure. Two randomly initialized embedding dictionaries E1, E2 are optimized; the adjacency matrix A = softmax(ReLU(E1 * E2^T)) captures asymmetric (directional) dependencies.
- **Mix-hop Propagation Layer**: Captures spatial dependencies via information propagation across nodes with multiple hop distances in a single layer, avoiding over-smoothing.
- **Dilated Inception Layer**: Extracts temporal patterns using multiple parallel dilated 1D convolutions with different kernel sizes and dilation factors.
- **End-to-end training**: Graph learning, graph convolution, and temporal convolution are jointly optimized.

**Key Innovations**:
1. First to jointly learn graph structure and perform spatial-temporal forecasting end-to-end.
2. Adaptive adjacency matrix eliminates need for predefined graphs.
3. Mix-hop propagation avoids the over-smoothing issue of deep GCNs.
4. Curriculum learning strategy for training stability.

**Datasets**: Traffic (METR-LA, PEMS-BAY), Solar-Energy, Electricity, Exchange-Rate.
**Results**: Outperforms baselines on 3 of 4 benchmark datasets; competitive on traffic datasets with known graph structures.
**Limitations**: Learned graph is static after training (does not change per time step); quadratic memory in number of nodes; no explicit temporal decomposition.

**Relevance to copper**: The graph learning module directly addresses the key challenge of discovering inter-commodity relationships without predefined knowledge. The asymmetric adjacency is particularly relevant as commodity relationships are often directional (e.g., oil price changes lead copper price changes).

---

### 1.2 StemGNN -- Spectral Temporal Graph Neural Network (Cao et al., NeurIPS 2020)

- **Title**: "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting"
- **Authors**: Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, Qi Zhang
- **Venue**: NeurIPS 2020 | arXiv: 2103.07719
- **Citations**: ~979
- **Code**: github.com/microsoft/StemGNN

**Architecture**:
- **Graph Fourier Transform (GFT)**: Decomposes multivariate signals into graph spectral components, capturing inter-series correlations in the spectral domain.
- **Discrete Fourier Transform (DFT)**: Applied after GFT to capture temporal periodicity patterns for each spectral component.
- **Spectral Convolution + GRU**: After dual spectral transformation, convolution and sequential learning predict spectral coefficients.
- **Inverse GFT + IDFT**: Reconstructs time-domain predictions from spectral predictions.

**Key Innovations**:
1. First to combine Graph Fourier Transform with Discrete Fourier Transform for joint spectral-temporal modeling.
2. Learns inter-series correlations automatically from data (no predefined graph).
3. Operates entirely in the spectral domain, providing frequency-domain interpretability.

**Datasets**: 10 real-world datasets including traffic, exchange rates, and electricity.
**Limitations**: Learned graph structure is still static; GFT assumes the graph Laplacian is fixed; scalability concerns with large numbers of series; spectral approach assumes stationarity.

**Relevance to copper**: The spectral domain approach naturally decomposes relationships into frequency components. For copper, this could separate long-term macro trend relationships from short-term trading noise, though the static graph assumption is problematic for regime-changing commodity markets.

---

### 1.3 Graph WaveNet (Wu et al., IJCAI 2019)

- **Title**: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
- **Authors**: Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang
- **Venue**: IJCAI 2019 | arXiv: 1906.00121
- **Citations**: ~2,500+

**Architecture**:
- **Adaptive Adjacency Matrix**: Learned through node embedding multiplication, similar to MTGNN but earlier. A_adp = softmax(ReLU(E1 * E2^T)).
- **Stacked Dilated Causal Convolutions**: Exponentially expanding receptive field across layers for long-range temporal dependencies. Follows WaveNet architecture.
- **Graph Convolution**: Diffusion-based graph convolution on both predefined and adaptive adjacency matrices.

**Key Innovations**:
1. Introduced the adaptive adjacency matrix concept for GNN time series (predecessor to MTGNN).
2. Combined WaveNet temporal backbone with graph spatial processing.
3. Supports both predefined and learned graph structures simultaneously.

**Datasets**: METR-LA, PEMS-BAY traffic datasets.
**Limitations**: Originally designed for traffic; adaptive matrix is static post-training.

---

### 1.4 AGCRN -- Adaptive Graph Convolutional Recurrent Network (Bai et al., NeurIPS 2020)

- **Title**: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
- **Authors**: Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang
- **Venue**: NeurIPS 2020 | arXiv: 2007.02842

**Architecture**:
- **Node Adaptive Parameter Learning (NAPL)**: Generates node-specific parameters for GCN, allowing different transformation weights per node.
- **Data Adaptive Graph Generation (DAGG)**: Infers graph structure from learned node embeddings; the adjacency matrix A = softmax(ReLU(E * E^T)) is derived from a single embedding dictionary E.

**Key Innovations**:
1. Node-specific parameters capture heterogeneous node behaviors (critical for commodities where each asset has distinct characteristics).
2. Data-driven graph generation without any prior knowledge.
3. Integrates graph convolution within GRU cells for tight spatial-temporal coupling.

**Relevance to copper**: NAPL concept could model how copper responds differently to macro shocks compared to, say, aluminum -- each commodity node gets its own learned transformation.

---

### 1.5 FourierGNN (Yi et al., NeurIPS 2023)

- **Title**: "FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective"
- **Authors**: Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu
- **Venue**: NeurIPS 2023 | arXiv: 2311.06190

**Architecture**:
- **Hypervariate Graph**: Each individual time step value of each variable becomes a node. For N variables with T time steps, the graph has N*T nodes. This treats spatial and temporal dimensions uniformly.
- **Fourier Graph Operator (FGO)**: Matrix multiplications in Fourier space that are mathematically equivalent to graph convolutions in the time domain but more computationally efficient.
- **Stacked FGO layers**: Multiple FGO layers capture increasingly complex spatial-temporal interactions.

**Key Innovations**:
1. Unifies spatial and temporal modeling into a single graph framework (no separate temporal and spatial modules).
2. Fourier-domain operations reduce computational cost.
3. Theoretically grounded equivalence between Fourier and graph operations.

**Datasets**: 7 benchmark datasets.
**Limitations**: Hypervariate graph can be very large for long sequences; interpretation more difficult than separate spatial/temporal modules.

---

### 1.6 TimeGNN (Xu et al., 2023)

- **Title**: "TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting"
- **Authors**: Nancy Xu, Chrysoula Kosma, Michalis Vazirgiannis
- **Venue**: arXiv: 2307.14680 (2023)
- **Citations**: ~39

**Architecture**:
- Learns dynamic temporal graph representations that capture the evolution of inter-series patterns over time.
- Graph structure changes at each time step based on the data.

**Key Innovations**:
1. Dynamic graph that evolves with data (unlike static graphs in MTGNN/StemGNN).
2. Extremely efficient: 4x to 80x faster inference than other SOTA graph-based methods.
3. Maintains comparable forecasting performance despite dramatic speed improvement.

**Relevance to copper**: Dynamic graph learning is essential for commodity markets where correlation structures change across market regimes. The efficiency gain is important for practical deployment.

---

### 1.7 Multi-Scale Adaptive Graph Neural Network (Chen et al., IEEE 2023)

- **Title**: "Multi-scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting"
- **Authors**: L. Chen, D. Chen, Z. Shang, B. Wu
- **Venue**: IEEE 2023
- **Citations**: ~256

**Architecture**:
- Constructs multiple graph structures at different scales from both proximity and network topology perspectives.
- Adaptive graph generation at each scale.
- Multi-scale fusion for final prediction.

**Relevance to copper**: Multi-scale approach directly maps to the idea that commodity relationships manifest at different frequencies (daily trading, weekly cycles, monthly macro, annual structural).

---

## 2. GNN Architectures for Financial Forecasting

### 2.1 MAGNN -- Multi-Modality Graph Neural Network (Cheng et al., 2022)

- **Title**: "Financial Time Series Forecasting with Multi-modality Graph Neural Network"
- **Authors**: Dawei Cheng, Fangzhou Yang, Sheng Xiang, Jin Liu
- **Venue**: Pattern Recognition, 2022
- **Citations**: ~513

**Architecture**:
- **Multi-modal input**: Combines numerical price data, textual news/sentiment, and relational knowledge graph data.
- **Heterogeneous graph construction**: Builds a financial knowledge graph with different node types (companies, sectors, events) and different edge types (ownership, supply chain, sector membership).
- **Graph Attention**: Applies attention across modalities and graph edges to weight information from different sources.
- **Temporal module**: LSTM-based encoding of sequential features before graph processing.

**Graph Construction**:
- Knowledge graph from financial databases (sector relationships, supply chains, ownership).
- Sentiment graphs from news co-occurrence.
- Price correlation graphs from historical data.
- All merged into a heterogeneous graph.

**Results**: Demonstrated significant improvement over single-modality baselines on Chinese and US stock markets.
**Limitations**: Requires extensive financial knowledge graph construction; knowledge graph may not transfer across markets; computationally expensive multi-modal fusion.

---

### 2.2 THGNN -- Temporal and Heterogeneous GNN (Xiang et al., 2023)

- **Title**: "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction"
- **Authors**: Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, Yuqi Liang
- **Venue**: ACM 2022/2023 | arXiv: 2305.08740

**Architecture**:
- **Dynamic relation graph**: Generates company relation graphs for each trading day based on historical price similarity.
- **Transformer encoder**: Temporal representation learning for each stock's time series.
- **Heterogeneous Graph Attention Network**: Processes the daily company relation graph to optimize stock embeddings by propagating information across related stocks.

**Graph Construction**: Dynamic -- graph structure changes daily based on rolling window price correlations. Different edge types capture different relationship strengths.

**Results**: Outperformed baselines on both US (NYSE, NASDAQ) and Chinese (CSI) markets. Deployed in a real-world trading system with significantly better portfolio returns.
**Key insight**: The dynamic daily graph allows the model to adapt to changing market regimes.

---

### 2.3 FSTGAT -- Financial Spatio-Temporal Graph Attention Network (Wei et al., 2025)

- **Title**: "FSTGAT: Financial Spatio-Temporal Graph Attention Network for Non-Stationary Financial Systems"
- **Authors**: Z.L. Wei, H.Y. An, Y. Yao, W.C. Su, G. Li, Saifullah, B.F. Sun et al.
- **Venue**: Symmetry, 2025

**Architecture**:
- Multi-scale spatio-temporal feature extraction using GAT.
- Specifically addresses non-stationarity in financial systems.
- Captures both spatial (inter-asset) and temporal (across time) dependencies simultaneously.

**Relevance**: Explicitly designed for the non-stationary nature of financial data, which is a key challenge for copper markets.

---

### 2.4 MDGNN -- Multi-Relational Dynamic Graph Neural Network (Qian et al., AAAI 2024)

- **Title**: "MDGNN: Multi-relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction"
- **Authors**: H. Qian, H. Zhou, Q. Zhao, H. Chen, H. Yao et al.
- **Venue**: AAAI 2024
- **Citations**: ~85

**Architecture**:
- Multi-relational: Different edge types capture different relationship categories (sector, supply chain, investor overlap, price correlation).
- Dynamic: Graph structure evolves over time.
- Comprehensive: Integrates multiple information sources.

**Results**: Achieved best performance on public stock datasets.
**Key innovation**: The multi-relational approach is highly relevant for commodities where relationships have different natures (trade flow, substitution, co-production, financial correlation).

---

### 2.5 DeltaLag -- Dynamic Lead-Lag Discovery (Zhou et al., 2025)

- **Title**: "DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets"
- **Authors**: Wanyun Zhou, Saizhuo Wang, Mihai Cucuringu, Zihao Zhang, Xiang Li, Jian Guo, Chao Zhang, Xiaowen Chu
- **Venue**: arXiv: 2511.00390 (2025)

**Architecture**:
- **Sparsified cross-attention mechanism**: Identifies relevant lead-lag pairs dynamically.
- Extracts lag-aligned features from leading assets to forecast lagging asset returns.
- Enables dynamic portfolio construction based on discovered lead-lag relationships.

**Key Innovations**:
1. First end-to-end deep learning method that discovers and exploits dynamic lead-lag structures.
2. Outperforms fixed-lag baselines and precomputed statistical lead-lag graphs.
3. Improved interpretability -- can visualize which assets lead which.

**Relevance to copper**: Lead-lag relationships are well-documented in commodities (e.g., oil leads copper, Chinese macro data leads metal prices). DeltaLag's approach of learning these dynamically rather than assuming fixed lags is highly relevant.

---

### 2.6 S3G -- Stock State Space Graph (Lu et al., 2026)

- **Title**: "S3G: Stock State Space Graph for Enhanced Stock Trend Prediction"
- **Authors**: Yao Lu, Kaiyi Hu, Luyan Zhang
- **Venue**: arXiv, 2026

**Architecture**:
- Constructs data-dependent graphs at each time point.
- State space models characterize evolutionary dynamics of these graphs.
- Wavelet transforms for denoising.

**Results**: Superior performance on CSI 500 historical data.
**Key innovation**: Combines state space models (for temporal evolution) with dynamic graphs and wavelet denoising.

---

### 2.7 Financial Adaptive Graph Attention Network (Jiang et al., 2025)

- **Title**: "Financial Adaptive Graph Attention Network for Cross-Stock Forecasting"
- **Authors**: Z.P. Jiang, H. Zou, D. Zhang, Q. Cai, Y. Li, X. Luo
- **Venue**: Springer, 2025

**Architecture**:
- DGSA (Dynamic Graph Structure Attention) that follows stock market rules.
- Cross-stock information propagation for improved prediction.
- Adaptive attention weights for different market conditions.

---

### 2.8 BiGAT -- Bi-Graph Attention Network for Energy (Liu et al., 2023)

- **Title**: "BiGAT: Bi-Graph Attention Network for Energy Price Forecasting"
- **Authors**: Y. Liu, W. Xiao, T. Chu
- **Venue**: Neural Computing and Applications, 2023

**Architecture**:
- Dual graph attention mechanism for energy product price forecasting.
- One graph captures supply-side relationships, another captures demand-side relationships.
- Attention-weighted fusion of both graph perspectives.

**Relevance**: Directly applicable concept for copper -- separate graphs for supply chain relationships vs. financial correlation relationships.

---

## 3. Graph Construction Methods in Finance

This section catalogs the specific methods used to construct graphs for GNN-based financial forecasting.

### 3.1 Correlation-Based Graphs

**Pearson Correlation Graphs**:
- Most common approach. Compute pairwise Pearson correlations over rolling windows; threshold to create adjacency matrix.
- Used by: Yin et al. (2021) -- stock correlation graphs for GCN; Lazcano et al. (2023) -- BiLSTM-GCN for oil prices.
- Pros: Simple, fast, well-understood. Cons: Linear only, symmetric, sensitive to window size, does not capture causality.

**Dynamic Conditional Correlation (DCC)**:
- Ma & Han (2026) -- BOHB-Optimized GNN combined with DCC for carbon-energy-stock correlations.
- Fanshawe et al. (2026) -- Hybrid Transformer-GNN for forecasting equity correlations using DCC.
- Pros: Captures time-varying correlation; well-established in finance. Cons: Parametric assumption (GARCH); computationally expensive for many assets.

**Partial Correlation Graphs**:
- Remove confounding effects by conditioning on other variables.
- Used in financial network analysis (e.g., Kenett et al., 2010 concept applied in recent GNN work).
- More informative than full correlation for identifying direct relationships.
- Not yet widely adopted in GNN financial forecasting -- a potential gap.

### 3.2 Causality-Based Graphs

**Granger Causality Graphs**:
- Tank et al. (2021) -- "Neural Granger Causality" (IEEE TPAMI, 634 citations). Foundational work on using neural networks to discover Granger causal structure in both linear and nonlinear settings.
- He et al. (2023) -- STGC-GNNs: Spatial-temporal Granger causality graph for traffic (Physica A, 93 citations). Applied Granger causality to construct the spatial graph for a GNN.
- Wu & Kang (2024) -- Granger Causality Test Dynamic Graph Attention Transformer. Combines Granger testing with GAT and Transformer architectures.
- Pros: Directional; economically interpretable; can test statistical significance. Cons: Assumes linearity (standard version); stationarity required; sensitive to lag selection.

**Transfer Entropy Graphs**:
- Lee & Cho (2025) -- H-ETE-GNN: Transfer Entropy GNN with Hurst-based regime adaptation for stock volatility. Uses Effective Transfer Entropy (ETE) to measure directional, asymmetric information flows between assets. Adapts graph structure based on Hurst exponent to detect market regime changes.
- Duan et al. (2022) -- CauGNN: Uses transfer entropy to construct causal graphs for multivariate time series forecasting (Tsinghua Science and Technology / IEEE).
- Xu et al. (2020) -- TEGNN: Transfer entropy graph neural network for multivariate time series forecasting. Proposes transfer entropy for accurate variable selection in graph construction.
- Li & Tang (2024) -- ETEGF: Combines ESMD decomposition with transfer entropy graph for multivariate sequence prediction.
- Pros: Non-linear; directional; model-free; information-theoretic. Cons: Computationally expensive; requires careful estimation of probability distributions; sensitive to bin size / embedding dimension.

### 3.3 Knowledge Graphs from Economic Relationships

- Cheng et al. (2022) -- MAGNN: Financial knowledge graph with heterogeneous nodes (companies, sectors, events) and edges (ownership, supply chain, sector membership).
- Ibrahim et al. (2022) -- Knowledge graph guided simultaneous forecasting and network learning for multivariate financial time series (ACM). Knowledge graphs guide the error residual learning process.
- Pan (2025) -- Integration of deep learning and knowledge graphs for agricultural commodity price risk early warning (IEEE Access).
- Chen et al. (2026) -- Knowledge graph-driven financial market forecasting with LLMs.

**For copper specifically**, a knowledge graph could encode:
- Copper supply chain: mines -> smelters -> refineries -> fabricators -> end-use sectors (construction, electronics, EV).
- Trade relationships: Chile/Peru (producers) -> China (consumer) flow.
- Substitution relationships: Copper-aluminum substitution in electrical applications.
- Co-production: Copper-gold, copper-molybdenum from same mines.
- Economic indicators: PMI -> industrial production -> copper demand.

### 3.4 Sector / Supply Chain Graphs

- Wu et al. (2023) -- Industry classification using supply chain network GNNs.
- Han (2024) -- SupplyGraph: GNN on supply chain graph outperforms conventional deep learning for demand forecasting.
- Kosasih & Brintrup (2022) -- GNN for predicting hidden links in supply chains.
- Tu et al. (2024) -- Large-scale knowledge graph for automotive supply chain with GNN-based supplier recommendation.
- Sieretti et al. (2025) -- GNN analysis of Indonesia's mineral downstream industry, tracing copper ore to cathodes and wire.

**Gap identified**: No paper has constructed a commodity-specific supply chain graph for GNN-based price forecasting. The supply chain literature uses GNNs for demand forecasting and link prediction, not price prediction.

### 3.5 Attention-Learned Graphs (No Predefined Structure)

- MTGNN (Wu et al., 2020) -- Learned adjacency from node embeddings.
- Graph WaveNet (Wu et al., 2019) -- Adaptive adjacency matrix from node embeddings.
- AGCRN (Bai et al., 2020) -- Data Adaptive Graph Generation.
- MDGNN (Qian et al., 2024) -- Multi-relational learned graph.
- DeltaLag (Zhou et al., 2025) -- Sparsified cross-attention for lead-lag discovery.

This family of methods is particularly promising for commodity markets where the true dependency structure is unknown and time-varying.

### 3.6 Visibility Graphs

- Mari & Mari (2026) -- Visibility graph analysis for Italian energy market gas and power price dynamics. Converts time series into graphs using the visibility algorithm, then applies GNN.
- Price Graphs (Wu et al., 2022) -- Converts price time series into graph structures using temporal associations.

Novel approach but limited adoption in commodity forecasting.

---

## 4. GNN for Commodity Markets

This section covers all identified papers applying GNNs to commodity markets specifically.

### 4.1 Copper-Specific GNN Papers

**Paper 1: MVGT for Copper (Sun et al., 2025)**
- **Title**: "Forecasting Copper Price with Multi-view Graph Transformer and Fractional Brownian Motion-Based Data Augmentation"
- **Authors**: Q. Sun, X. Yang, M. Zhong
- **Venue**: Natural Resources Research (Springer), 2025
- **Architecture**: Multi-View Graph Transformer (MVGT)
  - 5 distinct graph generation methods to capture non-Euclidean feature relationships.
  - Multi-view graph transformers generate embeddings from each graph view.
  - Attention-based fusion identifies which market views are most influential.
  - Fractional Brownian Motion (fBM) data augmentation for limited training data.
- **Graph Construction**: 5 different graph construction methods (details: likely correlation, sector, supply-chain, learned, and domain-knowledge based -- specifics require full paper access).
- **Dataset**: COMEX and LME copper prices, monthly frequency, with macro features.
- **Forecast horizon**: 1 month ahead.
- **Results**: Outperforms ANN, LSTM, and standard GNN models in training efficiency, forecasting accuracy, and generalization.
- **Limitations**: Monthly frequency only; single-target (copper only); does not use temporal decomposition; graph construction methods are not fully dynamic.
- **THIS IS THE ONLY PUBLISHED PAPER APPLYING GNN SPECIFICALLY TO COPPER PRICE FORECASTING**.

**Paper 2: VMD Hybrid for Copper (Li & Liu, 2026)**
- **Title**: "A Hybrid Model for Copper Futures Price Forecasting Utilizing Complexity-Aware Variational Mode Decomposition"
- **Authors**: Y. Li, D. Liu
- **Venue**: Entropy, 2026
- **Note**: Uses VMD but NOT GNN. Multi-view projection methods for heterogeneous copper market data. Included here as it represents the VMD side of the copper literature.

### 4.2 Non-Ferrous Metal / Base Metal Papers

**Shang et al. (2023)**:
- "Building public market opinion indices for electricity consumption prediction of Chinese nonferrous metal industry"
- Uses self-learning GNN for aluminum, copper, and other non-ferrous metal markets.
- Focus is on sentiment/opinion indices, not direct price forecasting.
- Published in Procedia Computer Science.

**Sieretti et al. (2025)**:
- Comprehensive GNN analysis of Indonesia's mineral downstream industry.
- Traces copper ore to copper cathodes and wire production.
- Focus is on policy analysis, not price forecasting.

### 4.3 Energy Commodity GNN Papers

**Lazcano et al. (2023)**:
- "A Combined Model Based on Recurrent Neural Networks and Graph Convolutional Networks for Financial Time Series Forecasting"
- Mathematics (MDPI), 124 citations.
- BiLSTM-GCN model for oil price data.
- Graph construction: Correlation-based on spatial and temporal characteristics.

**Liu et al. (2023) -- BiGAT**:
- Bi-graph attention network for energy price forecasting of oil products.
- Dual graph structure (supply-side and demand-side).

**Cao et al. (2026)**:
- Crude oil price prediction via LLM and heterogeneous GNNs (IEEE Access).
- Combines multimodal GNN with Temporal Fusion Transformer.
- Graph attention adaptively learns importance weights.

### 4.4 Agricultural Commodity GNN Papers

**Ozden & Bulut (2023)**:
- "Spectral Temporal Graph Neural Network for Multivariate Agricultural Price Forecasting"
- Ciencia Rural. Applies StemGNN to agricultural commodities.

**Min et al. (2025)**:
- "RNN and GNN Based Prediction of Agricultural Prices with Multivariate Time Series"
- Scientific Reports (Nature), 12 citations.
- Compares RNN and GNN approaches for agricultural commodities.

**Zhang et al. (2025)**:
- GCN-BiGRU-TPE for agricultural product futures prices.
- Computational Economics. Multi-graph construction with tree-structured Parzen estimator optimization.

### 4.5 Cross-Commodity Relationship Papers

**Cui et al. (2026)**:
- "Higher-order Moment Spillovers and Interpretable Prediction in Commodity Markets using ARCD, TVP-VAR-EJC, and Graph Neural Networks"
- Risk Management (Springer).
- Integrates kurtosis into GNN framework to capture complex nonlinear relationships and higher-order spillover effects across commodities.
- First to use GNN topology explicitly for inter-commodity spillover modeling.

**Son et al. (2023)**:
- "Forecasting Global Stock Market Volatility"
- Journal of Forecasting (Wiley). Spatial-temporal GNN incorporating financial connectedness of global markets.

**Ren et al. (2025)**:
- Industry stock index forecasting from risk connectedness perspective using multivariate time series GNN.
- China Journal of Econometrics.

### 4.6 Carbon Price GNN Papers

**Zhang et al. (2025)**:
- "A Text-Based Framework for Carbon Price Forecasting via Multivariate Temporal Graph Neural Network"
- Journal of Supercomputing. Integrates text/news data with temporal GNN for carbon prices.

**Ma & Han (2026)**:
- BOHB-Optimized Multivariable GNN with DCC for carbon-energy-stock correlation forecasting.
- Mathematics (MDPI).

---

## 5. Spectral vs Spatial Graph Convolutions in Finance

### 5.1 Spectral Approaches

Spectral methods operate in the graph frequency domain using eigendecomposition of the graph Laplacian.

**StemGNN (Cao et al., 2020)**: Combines GFT (graph spectral) with DFT (temporal spectral). The spectral representation provides natural frequency decomposition of inter-series relationships. Applied to financial data including exchange rates.

**Uygun & Sefer (2025)**: "Financial Asset Price Prediction with Graph Neural Network-Based Temporal Deep Learning Models." Explicitly uses Spectral Graph Convolution (GConv) blocks to capture spatial dependencies through graph convolution filtering. Applied to cryptocurrency markets.

**Jin et al. (2025)**: "Towards Expressive Spectral-Temporal Graph Neural Networks for Time Series Forecasting." Establishes theoretical framework for spectral-temporal GNN expressiveness. Proposes Temporal Graph Gegenbauer Convolution (TGGC) -- linear-complexity spectral convolution with proven expressiveness guarantees. Published in IEEE Transactions.

**Zheng et al. (2025)**: Stiefel Graph Spectral Convolution (IJCAI 2025). Constrains spectral graph learning to Stiefel manifolds for computational efficiency while maintaining expressiveness.

**Pros of spectral approaches for commodities**:
- Natural frequency interpretation (important for economic cycle analysis).
- Can separate trend, cyclical, and noise components through graph frequency decomposition.
- Strong theoretical foundations.

**Cons**:
- Eigendecomposition is O(n^3) for n nodes.
- Assumes fixed graph structure (graph Laplacian must be stable).
- Less intuitive for domain experts.

### 5.2 Spatial Approaches

Spatial methods operate directly on the graph topology through message passing / neighborhood aggregation.

**GAT-based models**: FSTGAT (Wei et al., 2025), STGAT (Feng et al., 2025), BiGAT (Liu et al., 2023). Attention-weighted message passing where each node attends to its neighbors. Naturally handles dynamic edge weights.

**GCN-based models**: Lazcano et al. (2023) BiLSTM-GCN for oil prices. Standard spatial convolution with fixed or learned adjacency.

**Diffusion-based**: Graph WaveNet uses diffusion convolution (random walk based) which captures multi-hop information propagation.

**Pros of spatial approaches for commodities**:
- More flexible with dynamic graphs.
- Attention weights are directly interpretable as relationship strengths.
- Scale better to larger graphs.
- Can incorporate edge features (e.g., trade flow volumes as edge attributes).

**Cons**:
- May miss global spectral patterns.
- Over-smoothing with many layers.
- No explicit frequency decomposition.

### 5.3 Hybrid / Emerging Approaches

**FourierGNN (Yi et al., 2023)**: Bridges spectral and spatial by performing graph convolutions in Fourier space.

**Graph Fourier Transformer (Zhai et al., 2025)**: "Mining Price Relationships of Futures for Investment Decisions." Extends ARCH models with graph Fourier transform layer. Directly targets futures markets.

**Recommendation for copper**: A spatial approach (GAT) with learned attention is most practical for the initial paper, given its flexibility with dynamic graphs and interpretability. Spectral approaches could be explored as a second contribution, particularly if combined with VMD (which provides an alternative frequency decomposition).

---

## 6. Dynamic Graph Construction Methods

Dynamic graphs -- where the graph structure changes over time -- are critical for financial applications where asset relationships evolve.

### 6.1 Correlation-Based Dynamic Graphs

**Rolling window correlations**: Compute Pearson/Spearman correlation over sliding windows. Simplest approach.
- THGNN (Xiang et al., 2023): Daily company relation graphs from rolling window price correlations.
- Common practice but suffers from window size sensitivity and lag.

**DCC-GARCH dynamic correlations**:
- Ma & Han (2026): BOHB-GNN with DCC for dynamic carbon-energy correlations.
- Fanshawe et al. (2026): Hybrid Transformer-GNN for equity correlation forecasting.
- Advantage: Econometrically principled; captures volatility clustering. Disadvantage: Parametric assumptions.

### 6.2 Causality-Based Dynamic Graphs

**Time-varying Granger causality**:
- Qian et al. (2026) -- MSDG: Multiscale dynamic graph neural network for inferring dynamic Granger causality. Addresses how causal relationships change across time scales.
- Liu (2024) -- Graph attention network with Granger causality map for fault detection (61 citations). Conditional causal mapping approach.
- STGC-GNNs (He et al., 2023): Spatial-temporal Granger causality graph.

**Dynamic transfer entropy**:
- Lee & Cho (2025) -- H-ETE-GNN: Hurst exponent-based regime detection determines when to recompute transfer entropy graphs. Most sophisticated approach to regime-adaptive causal graph construction.
- CauGNN (Duan et al., 2022): Transfer entropy for measuring information flow direction and magnitude between financial entities.

### 6.3 Attention-Learned Dynamic Graphs

**Per-sample adaptive graphs**:
- MDGNN (Qian et al., AAAI 2024): Multi-relational dynamic graph that evolves at each time step.
- TimeGNN (Xu et al., 2023): Temporal dynamic graph representations that capture evolution of inter-series patterns.
- S3G (Lu et al., 2026): Data-dependent graphs at each time point with state space model evolution.

**Cross-attention based**:
- DeltaLag (Zhou et al., 2025): Sparsified cross-attention for lead-lag pair identification. The graph is implicitly defined by attention weights.

### 6.4 Regime-Adaptive Graphs

- **Lee & Cho (2025)**: Hurst exponent-based dynamic graph updating. When Hurst exponent changes (indicating regime shift), the transfer entropy graph is recomputed. This is the most relevant approach for commodity markets with known regime dynamics.
- **Kumar et al. (2024)**: Dynamic GNN for enhanced volatility prediction. Edges represent correlations between indices; graph adapts to market conditions.
- **Islam et al. (2024)**: GNN for systemic financial risk. Models time-varying financial networks and cross-market contagion.

### 6.5 Evaluation: Which Dynamic Graph Method for Commodities?

| Method | Directional? | Nonlinear? | Regime-Aware? | Interpretable? | Computational Cost |
|--------|-------------|-----------|---------------|---------------|-------------------|
| Rolling correlation | No | No | Partial (via window) | High | Low |
| DCC-GARCH | No | No | Yes (built-in) | High | Medium |
| Granger causality | Yes | No (standard) | No (requires re-estimation) | High | Medium |
| Transfer entropy | Yes | Yes | No (requires re-estimation) | Medium | High |
| Neural Granger | Yes | Yes | Yes (if time-varying) | Medium | Medium-High |
| Learned attention | Yes | Yes | Yes (inherent) | Low-Medium | Low (at inference) |
| Hurst-adaptive TE | Yes | Yes | Yes | High | High |

**Recommendation**: For a copper forecasting paper, a hybrid approach is optimal -- use learned attention graphs for the main GNN (flexible, efficient) but validate against Granger causality / transfer entropy graphs for interpretability. Report both for publication.

---

## 7. Temporal Decomposition Combined with GNN

### 7.1 TDG4MSF (Miao et al., 2023)

- **Title**: "TDG4MSF: A Temporal Decomposition Enhanced Graph Neural Network for Multivariate Time Series Forecasting"
- **Authors**: H. Miao, Y. Zhang, Z. Ning, Z. Jiang, L. Wang
- **Venue**: Applied Intelligence (Springer), 2023
- **Citations**: ~10

**Architecture**: Integrates temporal decomposition (trend-seasonal-residual) with GNN for multivariate forecasting. The decomposition provides cleaner signals for the GNN to process.
**Key insight**: Separating trend from seasonal before graph processing improves GNN's ability to capture inter-series relationships.

### 7.2 STDNet -- Spatio-Temporal Decomposition (Jiang et al., 2024)

- **Title**: "STDNet: A Spatio-Temporal Decomposition Neural Network for Multivariate Time Series Forecasting"
- **Authors**: Z. Jiang, Z. Ning, H. Miao, L. Wang
- **Venue**: Tsinghua Science and Technology, 2024

**Architecture**: Applies decomposition strategy alongside GNN for multivariate forecasting. Separates spatial and temporal decomposition pathways.

### 7.3 VMD + GNN Combinations (Cross-Domain)

As documented in the existing gap analysis (`literature_gap_analysis.md`), 9 papers combine VMD with GNN, all outside finance:

1. **Pei et al. (2022)** -- PMNet: AVMD -> MtemGNN for PM2.5 forecasting.
2. **Ahmad et al. (2024)** -- VMGCN: VMD -> GCN for traffic; learns intra-mode and cross-mode dependencies.
3. **Ahmad et al. (2025)** -- Deep-unfolded VMD -> Mode Adaptive Graph Network (MAGN).
4. **Bao et al. (2025)** -- VMD -> GraphSAGE for wind turbine damage detection.
5. **Huang & Liu (2025)** -- Adaptive decomposition -> GNN for green supply chain.
6. **Yuan (2025)** -- Multivariate VMD -> cross-graph forecasting for mine gas.
7. **Yuan et al. (2025)** -- VMD smoothing -> multiview weighted GNN for tracking.
8. **Cheng & Tsai (2026)** -- MVMD -> scale-aware GNN for wind speed.
9. **Ahmad et al. (2025)** -- VMD -> VMGCN with 3D attention for noisy spatiotemporal data.

**Critical observation**: NONE of these apply VMD+GNN to financial/commodity data. NONE construct graph topology from decomposed modes. See Section 10 for detailed gap analysis.

### 7.4 Wavelet + GNN Combinations

**MAGNET (Hong et al., 2024)**: "Multilevel Dynamic Wavelet Graph Neural Network for Multivariate Time Series Classification" (ACM TKDD). Dynamic wavelet GNN with hierarchical approach for temporal-frequency pattern capture.

**Yang et al. (2024)**: VMD + spatio-temporal GNN for long-term wind power forecasting (International Journal of Green Energy).

### 7.5 Fourier / DCT + GNN Combinations

**Zheng & Zhufu (2025)**: DCT (Discrete Cosine Transform) decomposition with cross-scale GNN for stock prices. Closest to the proposed VMD-GNN approach but uses DCT instead of VMD and focuses on single-asset (not inter-commodity).

**FourierGNN (Yi et al., 2023)**: Uses Fourier operators within the graph framework itself rather than as a preprocessing decomposition step.

**Graph Fourier Transformer (Zhai et al., 2025)**: Graph Fourier transform layer optimized for futures investment analysis.

---

## 8. Interpretability Methods for GNN in Finance

### 8.1 GNNExplainer (Ying et al., 2019)

- **Title**: "GNNExplainer: Generating Explanations for Graph Neural Networks"
- **Authors**: Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec
- **Venue**: NeurIPS 2019 | arXiv: 1903.03894
- **Citations**: ~3,000+

**Method**: Identifies compact subgraph structures and node features most important for a prediction by maximizing mutual information. Model-agnostic.
**Application to finance**: Can identify which inter-commodity edges and which features drive a specific copper price prediction. Produces interpretable subgraph visualizations.

### 8.2 Attention Weight Visualization

Used extensively in financial GNN papers:
- MAGNN (Cheng et al., 2022): Attention weights reveal which modalities and graph edges matter most.
- FSTGAT (Wei et al., 2025): Spatio-temporal attention weights.
- MVGT for copper (Sun et al., 2025): Attention-based fusion weights show which market views are influential.

**For commodities**: Attention weights can directly answer "which commodities/factors influenced this copper price prediction and how strongly?"

### 8.3 Interpretable GNN Frameworks for Finance

**Dinani (2025)**: "Interpretable Graph Neural Network Framework for Forecasting Systemic Financial Risk." Integrates dynamic financial networks through GAT with built-in interpretability for systemic risk prediction.

**Wang (2025)**: "Enhancing Stock Market Prediction with Temporal Graph Neural Networks and Large Language Model-Based Explainability." Combines temporal GNNs with LLM-generated explanations.

**Arsenault et al. (2025)**: "A Survey of Explainable AI in Financial Time Series Forecasting" (ACM Computing Surveys). Comprehensive survey including graph-based methods.

### 8.4 General GNN Explainability

**Yuan et al. (2022)**: "Explainability in Graph Neural Networks: A Taxonomic Survey" (IEEE TPAMI). Categories:
1. **Gradient-based**: Saliency maps on graph edges.
2. **Perturbation-based**: GNNExplainer, PGExplainer.
3. **Decomposition-based**: LRP (Layer-wise Relevance Propagation) adapted for graphs.
4. **Surrogate-based**: Train interpretable model to approximate GNN locally.

**GraphXAI (Nandan et al., 2025)**: Survey integrating GNNs with XAI methods.

### 8.5 Interpretability Strategy for Copper GNN

For a publication-quality copper forecasting paper, implement multiple interpretability layers:

1. **Graph structure visualization**: Show learned inter-commodity graphs at different time periods (e.g., normal vs. crisis). Compare against known economic relationships.
2. **Attention weight analysis**: Which commodities/factors have highest attention when predicting copper? Does this change across forecast horizons?
3. **Frequency-specific graphs** (if using decomposition): Show how the copper-oil relationship differs at low vs. high frequencies. This provides unique economic insights.
4. **GNNExplainer**: For specific prediction dates (e.g., COVID crash, supply disruptions), identify which subgraph and features drove the prediction.
5. **Ablation as interpretability**: Removing each input variable and measuring prediction degradation reveals variable importance.

---

## 9. Implementation Frameworks and Technical Details

### 9.1 Framework Comparison

**PyTorch Geometric (PyG)**:
- Dominant framework in financial GNN research.
- Used by: Romanova (2024) for financial GNN classification, Zhang et al. (2026) for financial surveillance, Tian et al. (2023) for stock prediction.
- Strengths: Rich library of GNN layers (GCN, GAT, GraphSAGE, etc.); mini-batch support; temporal graph support via PyG-Temporal; strong community.
- Financial-specific: Easy integration with PyTorch-based financial models (Transformers, LSTMs).
- Recommended for: Most financial GNN projects, especially those using attention-based models.

**DGL (Deep Graph Library)**:
- Alternative to PyG with similar capabilities.
- Strengths: Backend-agnostic (supports PyTorch, MXNet, TensorFlow); good for heterogeneous graphs; efficient message passing.
- Less commonly used in recent financial GNN literature.

**PyG-Temporal (torch_geometric_temporal)**:
- Extension of PyG specifically for temporal graph neural networks.
- Pre-built layers: DCRNN, A3TGCN, TGCN, EvolveGCN, MTGNN.
- Ideal for financial time series where graph structure evolves.

**Recommendation**: PyG + PyG-Temporal for the copper forecasting project. The combination provides:
- Pre-built MTGNN, DCRNN layers as baselines.
- Easy custom layer implementation.
- Built-in data loaders for temporal graph data.
- Mini-batch training for scalability.

### 9.2 Handling Temporal Dynamics in Graph Structure

**Approach 1: Snapshot Graphs**
- Create a separate graph for each time window (e.g., daily, weekly).
- GNN processes each snapshot independently; temporal model (LSTM, Transformer) connects across snapshots.
- Used by: THGNN (Xiang et al., 2023), EvolveGCN.
- Pros: Simple, flexible. Cons: No information sharing across snapshots within GNN.

**Approach 2: Temporal Graph Networks (TGN)**
- Graph structure has temporal edges with timestamps.
- Memory module maintains evolving node states.
- Used by: Kim et al. (2024) for financial anomaly detection.
- Pros: Continuous-time modeling. Cons: More complex to implement.

**Approach 3: Recurrent Graph Neural Networks**
- GCN/GAT within RNN cells (DCRNN, AGCRN).
- Graph convolution replaces matrix multiplication in GRU/LSTM equations.
- Tightly couples spatial and temporal processing.
- Pros: Elegant integration. Cons: Sequential training (no parallelism).

**Approach 4: Attention-based temporal aggregation**
- Transformer-style attention across time steps, combined with graph attention across nodes.
- Used by: S3G (Lu et al., 2026) with state space models for temporal evolution.
- Pros: Parallelizable; long-range dependencies. Cons: Quadratic memory in sequence length.

**Recommendation for copper**: Start with Approach 1 (snapshot graphs) for simplicity and interpretability. Each snapshot represents a rolling window (e.g., 60-day) correlation/causality graph. Temporal modeling via separate module. Upgrade to Approach 3 or 4 if needed.

### 9.3 Multi-Scale Temporal Modeling with GNN

**Hierarchical temporal processing**:
- Process raw data at multiple temporal resolutions (daily, weekly, monthly).
- Separate GNN at each resolution level.
- Fuse predictions via attention or gating.
- Related work: MSGformer (Zhu et al., 2025), Multi-Scale AGNN (Chen et al., 2023).

**Decomposition-based multi-scale** (our proposed approach):
- VMD decomposes into frequency modes.
- Each mode captures a different temporal scale.
- GNN operates per mode or on a mode-augmented graph.
- No existing work in finance.

**Dilated convolution multi-scale**:
- MTGNN's dilated inception layer.
- Different dilation factors capture different temporal scales within a single model.
- Simple but effective.

### 9.4 Key Implementation Considerations

**Data preprocessing**:
- Normalize each variable separately (z-score or min-max).
- Handle missing data: Forward fill for market closures; interpolation for data gaps.
- Align different data frequencies (daily prices, monthly macro indicators) -- use last-available-value for low-frequency data.

**Graph construction hyperparameters**:
- Correlation threshold for adjacency: Tune via validation set (common range: 0.3-0.7).
- Rolling window size for dynamic correlation: 60, 120, or 252 trading days.
- Number of VMD modes: Typically K=3 to 7 for financial data (tune via validation).
- VMD penalty parameter alpha: Controls bandwidth of modes.

**Training details**:
- Walk-forward validation (expanding or sliding window).
- Common lookback windows: 20 (1 month), 60 (1 quarter), 252 (1 year) trading days.
- Common forecast horizons: 1, 5, 10, 20 trading days (1 day to 1 month).
- Loss function: MSE for point forecasting; directional accuracy as secondary metric.
- Early stopping on validation loss.
- Learning rate: 1e-3 to 1e-4 with cosine annealing.

**Evaluation metrics**:
- RMSE, MAE, MAPE for magnitude.
- Directional Accuracy (DA) for direction.
- R-squared for explained variance.
- Diebold-Mariano test for statistical significance vs. baselines.
- Information Coefficient (IC) if framed as signal generation.

---

## 10. Key Innovations and Gaps

### 10.1 What Graph Construction Methods Work Best for Commodities?

Based on this review, the following hierarchy emerges:

1. **Attention-learned graphs** appear most effective for pure prediction performance (MTGNN, MDGNN, DeltaLag). They can capture nonlinear, asymmetric, time-varying relationships without imposed assumptions.

2. **Transfer entropy graphs** provide the best interpretability for directional information flow, which matches how commodities interact (oil "leads" copper). Lee & Cho (2025) show combining with regime detection improves robustness.

3. **Knowledge/supply-chain graphs** provide strong inductive bias for commodities where structural relationships are well-known. MAGNN (Cheng et al., 2022) shows this works for stocks; no one has applied it to commodities.

4. **Correlation graphs** are simple baselines that work well when relationships are approximately linear and stationary, which is often insufficient for commodities.

**Untested for commodities**: Granger causality graphs, partial correlation graphs, DCC-GARCH dynamic graphs (all used in stocks/equity but not commodities).

### 10.2 Has Anyone Combined Graph Structure with Temporal Decomposition?

**Answer**: Barely.

- TDG4MSF (Miao et al., 2023) decomposes time series into trend/seasonal/residual before GNN, but uses the same graph for all components.
- VMD+GNN papers (Ahmad et al., 2024/2025; Cheng & Tsai, 2026) use VMD as preprocessing before GNN, but apply it to pre-existing spatial graphs (traffic, wind), not constructing new graphs from the decomposition.
- Zheng & Zhufu (2025) use DCT + cross-scale GNN for stocks, the closest to "constructing graphs from decomposed signals" but uses DCT (not VMD) and does not construct frequency-specific inter-variable graphs.

**CONFIRMED GAP**: No paper constructs graph topology from decomposed modes. No paper builds frequency-specific inter-variable graphs. No paper applies any form of decomposition+GNN to commodity/financial data.

### 10.3 What Has Not Been Tried?

**Major untested combinations for commodity forecasting**:

1. **VMD + GNN for any financial asset** (zero papers).
2. **Frequency-specific inter-commodity graph construction** (zero papers).
3. **Cross-frequency graph attention** between different variables at different frequency bands (only exists in EEG, never in finance).
4. **Supply chain graph + GNN for commodity price forecasting** (supply chain GNN exists for demand forecasting, not price forecasting).
5. **Regime-adaptive graph + temporal decomposition** (two active threads never combined).
6. **Causal discovery on decomposed modes** to build frequency-band-specific causal graphs (zero papers).
7. **GNN for copper/base metals** -- only Sun et al. (2025) exists, using Graph Transformer without decomposition.
8. **Heterogeneous graph with mode-variable node types** (zero papers in any domain).
9. **Transfer entropy computed on VMD modes** for causality at specific frequency bands (zero papers).
10. **Ensemble of spectral and spatial GNN convolutions** with mode decomposition (zero papers).

### 10.4 Dynamic vs Static Graphs for Volatile Commodity Markets

**The evidence strongly favors dynamic graphs**:

- THGNN (Xiang et al., 2023): Dynamic daily graphs outperform static graphs on stock markets.
- Lee & Cho (2025): Hurst-based regime detection with dynamic transfer entropy graphs improves stock volatility forecasting.
- MDGNN (Qian et al., AAAI 2024): Dynamic multi-relational graphs achieve best performance.
- TimeGNN (Xu et al., 2023): Dynamic temporal graphs match static graph performance with 4-80x speed gain.

**For commodity markets specifically**: Commodity correlations are notoriously unstable (e.g., copper-oil correlation was ~0.7 pre-COVID, dropped during COVID, recovered differently post-COVID). Static graphs would miss these structural breaks. Dynamic graphs that recompute per time step or per regime are essential.

**Recommended approach**: Learned adaptive graph (like MTGNN) that evolves implicitly, validated against explicitly dynamic correlation or causality graphs at key regime change points.

### 10.5 Positioning for Maximum Novelty

The single strongest novelty claim for a copper GNN paper combines three confirmed gaps:

1. **First GNN for copper/base metals price forecasting** (only Sun et al. 2025 exists, and they use Graph Transformer, not GNN+decomposition).
2. **First to construct frequency-specific inter-commodity graphs from VMD decomposition**.
3. **First to combine temporal decomposition with graph neural networks for financial data**.

This triple gap creates a highly defensible contribution. The approach is technically feasible (all components are individually well-studied) and provides clear ablation paths for rigorous evaluation.

---

## 11. Consolidated Paper Table

### Foundational GNN for Time Series

| Paper | Year | Venue | Architecture | Graph Construction | Dataset | Citations |
|-------|------|-------|--------------|-------------------|---------|-----------|
| MTGNN (Wu et al.) | 2020 | KDD | Mix-hop GCN + Dilated Inception | Learned adaptive (node embeddings) | Traffic, Solar, Electricity | ~2,804 |
| StemGNN (Cao et al.) | 2020 | NeurIPS | GFT + DFT + Conv + GRU | Learned (spectral) | 10 datasets incl. exchange rates | ~979 |
| Graph WaveNet (Wu et al.) | 2019 | IJCAI | Diffusion GCN + WaveNet | Adaptive adjacency + predefined | Traffic (METR-LA, PEMS-BAY) | ~2,500+ |
| AGCRN (Bai et al.) | 2020 | NeurIPS | Adaptive GCN + GRU | Data Adaptive Graph Generation | Traffic | N/A |
| FourierGNN (Yi et al.) | 2023 | NeurIPS | Fourier Graph Operator | Hypervariate graph (all values as nodes) | 7 benchmarks | N/A |
| TimeGNN (Xu et al.) | 2023 | arXiv | Dynamic temporal GNN | Dynamic per-timestep | Multiple MTS benchmarks | ~39 |
| Multi-Scale AGNN (Chen et al.) | 2023 | IEEE | Multi-scale adaptive GNN | Multiple scale-specific graphs | Multiple MTS benchmarks | ~256 |

### Financial Forecasting with GNN

| Paper | Year | Venue | Architecture | Graph Construction | Market | Citations |
|-------|------|-------|--------------|-------------------|--------|-----------|
| MAGNN (Cheng et al.) | 2022 | Pattern Recognition | Multi-modal GNN + LSTM | Knowledge graph + correlation | US & Chinese stocks | ~513 |
| THGNN (Xiang et al.) | 2023 | ACM | Transformer + Heterogeneous GAT | Daily dynamic correlation | US & Chinese stocks | N/A |
| FSTGAT (Wei et al.) | 2025 | Symmetry | Spatio-temporal GAT | Multi-scale financial graph | Stocks | N/A |
| MDGNN (Qian et al.) | 2024 | AAAI | Multi-relational dynamic GNN | Multi-relational dynamic | Stocks | ~85 |
| DeltaLag (Zhou et al.) | 2025 | arXiv | Sparsified cross-attention | Lead-lag attention | Stocks | N/A |
| ML-GAT (Huang et al.) | 2022 | IEEE Access | Multilevel GAT | Multiple levels | Stocks | N/A |
| DHGAT (Ao & Ma) | N/A | SSRN | Dynamic heterogeneous GAT | Heterogeneous dynamic | Stocks | N/A |
| Uygun & Sefer | 2025 | Neural Comp & App | Spectral GConv temporal DL | Spectral | Crypto | ~20 |
| BiLSTM-GCN (Lazcano et al.) | 2023 | Mathematics | BiLSTM + GCN | Correlation | Oil prices | ~124 |
| BiGAT (Liu et al.) | 2023 | Neural Comp & App | Bi-graph attention | Supply + demand dual graph | Energy prices | N/A |

### Commodity-Specific GNN

| Paper | Year | Venue | Architecture | Graph Construction | Commodity | Citations |
|-------|------|-------|--------------|-------------------|-----------|-----------|
| MVGT (Sun et al.) | 2025 | Natural Resources Research | Multi-view Graph Transformer + fBM augmentation | 5 graph generation methods | Copper (COMEX, LME) | N/A |
| Li & Liu | 2026 | Entropy | VMD hybrid (no GNN) | N/A | Copper futures | N/A |
| Cui et al. | 2026 | Risk Management | GNN with higher-order moments | Spillover topology | Cross-commodity | N/A |
| Ozden & Bulut | 2023 | Ciencia Rural | StemGNN | Spectral learned | Agricultural | ~7 |
| Min et al. | 2025 | Scientific Reports | GNN vs RNN comparison | Correlation | Agricultural | ~12 |
| Zhang et al. | 2025 | Computational Economics | GCN-BiGRU-TPE | Multi-graph | Agricultural futures | N/A |
| Zhang et al. | 2025 | J. Supercomputing | Temporal GNN + text | Text-enhanced graph | Carbon price | ~6 |

### Causality + GNN

| Paper | Year | Venue | Method | Graph Type | Application | Citations |
|-------|------|-------|--------|-----------|-------------|-----------|
| Neural Granger (Tank et al.) | 2021 | IEEE TPAMI | Neural network Granger causality | Granger causal graph | General MTS | ~634 |
| CauGNN (Duan et al.) | 2022 | Tsinghua S&T / IEEE | Transfer entropy + GNN | Transfer entropy graph | Financial MTS | N/A |
| H-ETE-GNN (Lee & Cho) | 2025 | Fractal & Fractional | Transfer entropy + Hurst regime | Dynamic TE graph | Stock volatility | N/A |
| STGC-GNNs (He et al.) | 2023 | Physica A | Granger causality + GNN | Spatio-temporal causal | Traffic | ~93 |
| CausalGNN (Wang et al.) | 2022 | AAAI | Causal GNN | Causal epidemic graph | Epidemiology | ~177 |
| MSDG (Qian et al.) | 2026 | Information Sciences | Multiscale dynamic GNN | Dynamic Granger | General MTS | N/A |

### Decomposition + GNN

| Paper | Year | Venue | Decomposition | GNN Type | Domain | Key Innovation |
|-------|------|-------|---------------|----------|--------|---------------|
| TDG4MSF (Miao et al.) | 2023 | Applied Intelligence | Trend-seasonal-residual | GNN | General MTS | First explicit decomposition+GNN |
| STDNet (Jiang et al.) | 2024 | Tsinghua S&T | Spatio-temporal decomposition | GNN | General MTS | Parallel decomposition pathways |
| VMGCN (Ahmad et al.) | 2024 | Various | VMD | GCN | Traffic | Intra-mode + cross-mode learning |
| MAGN (Ahmad et al.) | 2025 | Various | Deep-unfolded VMD | Adaptive GNN | Spatiotemporal | Trainable VMD within GNN |
| MVMD-GNN (Cheng & Tsai) | 2026 | Various | MVMD | Scale-aware GNN | Wind speed | Multivariate VMD + scale-specific GNN |
| DCT-GNN (Zheng & Zhufu) | 2025 | ACM | DCT | Cross-scale GNN | Stocks | Closest to proposed approach |

---

## Key References for Implementation

### Must-Read Papers (Priority Order for Copper GNN Project)

1. **MTGNN** (Wu et al., KDD 2020) -- Foundational adaptive graph + temporal architecture. Start here for base implementation.
2. **MVGT for Copper** (Sun et al., 2025) -- Only existing copper GNN paper. Must cite and differentiate.
3. **StemGNN** (Cao et al., NeurIPS 2020) -- Spectral approach baseline.
4. **THGNN** (Xiang et al., 2023) -- Dynamic graph construction for finance.
5. **H-ETE-GNN** (Lee & Cho, 2025) -- Transfer entropy + regime detection for financial graphs.
6. **CauGNN** (Duan et al., 2022) -- Transfer entropy graph construction.
7. **MDGNN** (Qian et al., AAAI 2024) -- Multi-relational dynamic graph for finance.
8. **FourierGNN** (Yi et al., NeurIPS 2023) -- Unified spatial-temporal via Fourier.
9. **TDG4MSF** (Miao et al., 2023) -- Temporal decomposition + GNN baseline.
10. **GNNExplainer** (Ying et al., 2019) -- Interpretability tool.

### Code Repositories

- MTGNN: github.com/nnzhan/MTGNN
- StemGNN: github.com/microsoft/StemGNN
- Graph WaveNet: github.com/nnzhan/Graph-WaveNet
- FourierGNN: github.com/aikunyi/FourierGNN
- PyG-Temporal: github.com/benedekrozemberczki/pytorch_geometric_temporal
- GNNExplainer: github.com/RexYing/gnn-model-explainer

---

*End of literature review. Last updated April 27, 2026.*
