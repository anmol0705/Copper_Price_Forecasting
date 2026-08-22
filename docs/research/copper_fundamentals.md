# Copper Market Fundamentals for Price Forecasting
## Comprehensive Research Report (April 2026)

---

## Table of Contents

1. [Copper Market Structure](#1-copper-market-structure)
2. [Supply-Demand Dynamics](#2-supply-demand-dynamics)
3. [Macro-Economic Relationships](#3-macro-economic-relationships)
4. [Green Transition and Emerging Demand](#4-green-transition-and-emerging-demand)
5. [Data Sources for Quantitative Research](#5-data-sources-for-quantitative-research)
6. [Underexplored Features for Price Forecasting](#6-underexplored-features-for-price-forecasting)

---

## 1. Copper Market Structure

### 1.1 Major Exchanges

Copper is traded on three primary exchanges, each serving different geographic and functional roles:

**London Metal Exchange (LME)**
- The world's largest market for standardized forward contracts, futures, and options on base metals.
- **Contract unit**: 25 tonnes of Grade A copper cathode (min 99.99% purity per BS EN 1978:1998).
- **Unique date structure**: Unlike other commodity exchanges, LME trades daily prompt dates out to 3 months, weekly dates to 6 months, and monthly dates to 123 months (over 10 years). This granular forward curve is critical for physical hedging.
- **Official settlement price**: Determined daily via open-outcry ring trading at 12:30 London time. The "cash seller" settlement price is the global benchmark for physical copper contracts.
- **Contract types**: Futures, options, TAPOs (Traded Average Price Options -- monthly average settled), LMEminis (cash-settled, smaller lots).
- **Warehouse system**: LME operates a global network of approved warehouses. Warrant holders can take physical delivery. Warehouse stocks are a closely watched indicator of physical market tightness (see Section 2.4).
- **Trading hours**: Ring (open outcry) 11:40--17:00 London; electronic (LMEselect) nearly 24 hours.
- URL: https://www.lme.com/en/metals/non-ferrous/lme-copper

**COMEX (CME Group)**
- The primary US copper futures exchange, part of CME Group's metals complex.
- **Contract unit**: 25,000 pounds (~11.34 tonnes) of electrolytic copper cathode (Grade 1).
- **Price quotation**: US cents per pound.
- **Settlement**: Physical delivery against COMEX-approved warehouses, predominantly in the US.
- **Tick size**: 0.0005 USD/lb ($12.50 per contract).
- **Options**: Liquid options on copper futures with American-style exercise. Monthly and weekly expirations available. Options data (skew, put-call ratio) is a valuable but underused forecasting signal (see Section 6).
- **Current price** (April 2026): ~$6.07/lb (~$13,380/tonne); 52-week range: $4.33--$6.58/lb.
- URL: https://www.cmegroup.com/markets/metals/base/copper.html

**Shanghai Futures Exchange (SHFE)**
- China's primary copper futures market. China consumes over 50% of global copper, making SHFE prices increasingly important.
- **Contract unit**: 5 tonnes per lot.
- **Trading hours**: Daytime session (9:00--15:00 CST) and night session (21:00--01:00 CST).
- **Role**: Bridges the time gap between LME close and COMEX open. SHFE night session overlaps with LME daytime trading.
- **Bonded copper stocks**: Copper held in bonded warehouses (customs-exempt zones) in Shanghai is a separate, closely watched inventory metric reflecting Chinese import pipeline.
- SHFE-LME price spreads (the "arb window") drive physical copper flows into/out of China and are a key signal for physical market tightness.
- URL: https://www.shfe.com.cn/en/

### 1.2 Spot vs. Futures and the Forward Curve

The LME forward curve is the richest source of term-structure information in base metals:

- **Cash price**: Price for immediate delivery (T+2 settlement).
- **3-month price**: The most liquid and widely quoted LME benchmark.
- **Cash-3M spread**: The premium or discount of cash over 3-month forward. This is the most important structural indicator.

**Contango** (futures > spot):
- Normal market state reflecting cost of carry (storage, insurance, financing).
- Indicates adequate supply availability and willingness to hold inventory.
- Deep contango signals perception of current supply surplus.
- Supports "cash-and-carry" arbitrage strategies.

**Backwardation** (spot > futures):
- Signals immediate physical tightness -- spot metal commands a premium because it is scarce.
- Steep backwardation indicates perception of current shortage.
- Historically associated with drawdowns in LME warehouse stocks.
- "Convenience yield" theory: holders value immediate possession more than future delivery.
- Backwardation tends to be associated with price rallies but can also precede mean-reversion.

**Forecasting relevance**: The shape of the forward curve (contango/backwardation), its slope, and changes in slope are potentially powerful predictive features. The spread between cash and 3-month LME copper encodes market expectations about near-term supply-demand balance.

### 1.3 Cross-Exchange Arbitrage

- **LME-COMEX spread**: Differences arise from regional supply-demand, tariffs (e.g., US Section 232 tariff risks in 2024--2025 caused COMEX premiums to spike over LME), and basis differentials. Historically co-integrated but can diverge sharply.
- **SHFE-LME arbitrage ("the arb")**: When SHFE prices exceed LME + shipping + insurance + duties, the "arb window" is open, attracting copper imports into China. When it closes, copper may flow out or imports slow. This arb signal is a direct indicator of Chinese demand strength.
- **SHFE bonded premium**: The premium of bonded copper over SHFE futures indicates strength of physical demand in China's free-trade zones.

---

## 2. Supply-Demand Dynamics

### 2.1 Global Production (Mine Output)

Global copper mine production in 2024 was estimated at approximately 22.8--23.0 million metric tonnes.

**Top producing countries (2022--2024 estimates, thousand tonnes):**

| Rank | Country | Production (kt) | Share |
|------|---------|-----------------|-------|
| 1 | Chile | 5,300 | ~23% |
| 2 | DR Congo | 3,300 | ~14% |
| 3 | Peru | 2,600 | ~11% |
| 4 | China | 1,800 | ~8% |
| 5 | Indonesia | 1,100 | ~5% |
| 6 | United States | 1,100 | ~5% |
| 7 | Russia | 930 | ~4% |
| 8 | Australia | 800 | ~3% |
| 9 | Kazakhstan | 740 | ~3% |
| 10 | Zambia | 680 | ~3% |
| -- | **World Total** | **~23,000** | **100%** |

Source: USGS Mineral Commodity Summaries / Wikipedia compilation of USGS data.

**Key observations for forecasting:**
- Top 3 countries (Chile, DRC, Peru) account for ~48% of global mine output. Supply disruption in any of these is a major price catalyst.
- Chile's output has been flat/declining due to water scarcity, aging mines (Escondida, Chuquicamata), and declining ore grades.
- DRC has been the fastest-growing producer, rising from minor player to #2 globally in a decade, driven by Kamoa-Kakula and other projects. Political/regulatory risk is significant.
- Peru faces recurring social/political disruptions (Las Bambas, Antapaccay blockades).
- China is a large miner but an even larger consumer; its net deficit drives massive imports.

**Major mining companies**: Freeport-McMoRan (world's largest publicly traded copper producer; Grasberg, Indonesia), Codelco (state-owned, Chile; Chuquicamata, El Teniente), BHP (Escondida, Chile; Olympic Dam, Australia), Glencore, Southern Copper, First Quantum, Teck Resources, Ivanhoe Mines (Kamoa-Kakula, DRC).

**Reserves**: Estimated at 25--60 years of current production, depending on demand growth assumptions. More than 95% of all copper ever mined has been extracted since 1900.

### 2.2 Treatment Charges and Refining Charges (TC/RC)

TC/RC are the fees that copper smelters charge miners to convert concentrate into refined cathode:
- **TC (Treatment Charge)**: USD per dry metric tonne of concentrate.
- **RC (Refining Charge)**: US cents per pound of payable copper.
- **Benchmark**: Set annually through negotiations between major Japanese/Chinese smelters and large miners. The annual benchmark TC/RC is a barometer of concentrate market tightness.

**How TC/RC signal supply-demand:**
- **High TC/RC** = abundant concentrate supply relative to smelter capacity; miners have less bargaining power.
- **Low TC/RC** = tight concentrate market; smelters compete for feed and accept lower margins.
- **2024--2025 context**: TC/RC collapsed to near-zero or even negative levels in late 2024/early 2025 as smelter capacity (especially in China) expanded faster than concentrate supply. This forced some smelters to cut production.
- Chinese smelters reached record refined copper output of 1.33 million tonnes in March 2026, despite low TC/RC, indicating aggressive capacity expansion.

**Forecasting relevance**: TC/RC levels are a leading indicator of refined copper supply availability. When TC/RC are very low, smelter cutbacks eventually tighten refined supply and support prices.

### 2.3 Copper Scrap and Recycling

- Recycled copper supplies approximately one-third of global demand (~8 Mt).
- An estimated 80% of all copper ever mined is still in use, making copper the third most recycled metal (after iron and aluminium).
- **Scrap spread**: The discount of scrap (No. 2 copper) to cathode (Grade A) is a key physical market indicator. When the spread narrows, scrap substitution increases and relieves primary demand; when it widens, scrap supply is tight.
- Recycling requires significantly less energy than primary extraction (~85% energy savings), making it economically sensitive to copper price levels.
- **China scrap import policy**: China's evolving restrictions on scrap imports (reclassification of "waste" vs. "resource") significantly affect global scrap flows and prices.

### 2.4 Inventory Levels

Exchange-reported inventories are the most closely watched physical market indicator:

- **LME warehouse stocks**: Reported daily. Historically ranged from ~100,000 to 1,000,000+ tonnes. Drawdowns signal physical tightness and often precede backwardation and price rallies.
- **SHFE warehouse stocks**: Reported weekly. Reflect Chinese domestic supply conditions.
- **COMEX warehouse stocks**: Reported daily. US-focused inventory.
- **Bonded warehouse stocks (China)**: Not reported by exchanges but estimated by market intelligence services. Represent copper in China's free-trade zones awaiting customs clearance.

**Total visible inventory** (LME + SHFE + COMEX) is typically 300,000--800,000 tonnes, representing only 5--15 days of global consumption. This thin inventory buffer makes copper prices highly sensitive to supply disruptions.

**Forecasting relevance**: Inventory levels, changes in inventory, and days-of-supply ratios are among the most powerful physical market features for price prediction. Inventory draws tend to be coincident with or slightly leading price increases.

### 2.5 Supply-Demand Balance (IEA Projections)

Based on IEA Global Critical Minerals Outlook 2024:

| Metric | 2021 | 2023 | 2030 (APS) | 2040 (APS) |
|--------|------|------|------------|------------|
| Clean tech demand | 5,380 kt | 6,311 kt | 12,001 kt | 16,343 kt |
| Other uses | 19,548 kt | 19,543 kt | 19,127 kt | 20,036 kt |
| **Total demand** | **24,928 kt** | **25,855 kt** | **31,128 kt** | **36,379 kt** |
| Secondary supply (recycling) | -- | -- | 5,879 kt | 10,006 kt |
| Primary supply requirement | -- | -- | 25,249 kt | 25,373 kt |

**Supply risk**: 31% shortfall in the project pipeline vs. 2035 Announced Pledges Scenario (APS) mining requirements. 52% of mines are located in high water-stress areas. Top 3 mining countries represent 54% of supply by 2040.

---

## 3. Macro-Economic Relationships

### 3.1 "Doctor Copper" -- Copper as Economic Bellwether

Copper's ubiquitous use in construction, manufacturing, electronics, and infrastructure gives it a reputation as a barometer of global economic health. The nickname "Doctor Copper" reflects the metal's perceived ability to diagnose the economy's health:

- **Rationale**: Copper has no viable large-scale substitute in most applications. Demand closely tracks industrial production, construction activity, and manufacturing output.
- **Empirical support**: Copper prices tend to lead or coincide with turning points in global GDP and industrial production. Sharp copper sell-offs have preceded recessions (2008, 2020).
- **Caveats**: The relationship has weakened somewhat post-2020 as speculative flows and green transition demand have introduced structural demand shifts not tied to cyclical GDP.

### 3.2 Copper-Dollar Relationship

- Copper is priced in USD globally (LME, COMEX). A stronger dollar mechanically makes copper more expensive for non-USD buyers, reducing demand at the margin.
- **Inverse correlation**: Copper and DXY (US Dollar Index) exhibit a well-documented negative correlation, typically -0.3 to -0.6 depending on the time period.
- **DXY composition**: Euro (57.6%), Japanese Yen (13.6%), British Pound (11.9%), Canadian Dollar (9.1%), Swedish Krona (4.2%), Swiss Franc (3.6%). Heavily Euro-weighted.
- **Mechanism**: Dollar weakness often reflects loose US monetary policy, which stimulates global growth and commodity demand. Dollar strength reflects tightening and risk aversion.
- **Forecasting relevance**: DXY is one of the most reliable macro features for copper models. The relationship may be frequency-dependent: strong at low frequencies (macro trends) but noisy at daily frequencies.

### 3.3 Copper-PMI Relationship

The Purchasing Managers' Index (PMI) is a leading indicator of manufacturing activity:

- **PMI > 50**: Manufacturing expansion; **PMI < 50**: Contraction.
- **Key PMI indices for copper**:
  - **Caixin China Manufacturing PMI**: Most important single PMI for copper, given China's >50% share of global consumption.
  - **ISM US Manufacturing PMI**: Second most important; reflects the world's largest economy.
  - **S&P Global / Markit Eurozone PMI**: Reflects European industrial demand.
  - **Global Composite PMI**: Broadest measure.
- PMI data is released monthly, on the first business day after the reference month, making it among the timeliest macro indicators.
- **Correlation**: Copper prices have historically shown positive correlation with global manufacturing PMI, particularly at 1--3 month horizons.

### 3.4 Interest Rates and Monetary Policy

- **Cost of carry**: Higher interest rates increase the cost of holding physical copper inventory (financing cost), reducing incentive to stockpile and potentially increasing contango.
- **Demand channel**: Higher rates slow construction and manufacturing investment, reducing copper demand with a lag of 6--18 months.
- **Dollar channel**: Rate hikes strengthen the dollar, creating headwinds for copper via the copper-dollar mechanism.
- **Key rates**: US Federal Funds Rate, PBOC Loan Prime Rate (China), ECB deposit rate.
- **Forecasting relevance**: Interest rate differentials (US vs. China, US vs. EM) may be more informative than absolute rate levels. The Fed Funds rate and 10Y Treasury yield are standard model inputs.

### 3.5 Copper-to-Gold Ratio

- The copper/gold ratio is used as a macro signal: copper represents industrial demand (risk-on) while gold represents safe-haven demand (risk-off).
- Rising copper/gold ratio signals improving economic outlook; falling ratio signals deterioration.
- Some analysts use the copper/gold ratio as a proxy for real interest rates and bond yields.
- **Forecasting relevance**: May be a useful feature capturing risk sentiment regime shifts.

### 3.6 Other Macro Relationships

- **S&P 500 / MSCI World**: Copper correlates with equity markets through the shared "risk appetite" channel. Copper is sometimes called a "risk-on" commodity.
- **VIX (Volatility Index)**: Spikes in VIX (fear gauge) tend to coincide with copper sell-offs. Negative correlation.
- **Chinese fixed asset investment**: A direct demand driver given China's infrastructure-heavy copper consumption.
- **Real estate indicators (China)**: New housing starts, property investment, and developer credit conditions directly affect copper demand in China's construction sector (the single largest end-use).

---

## 4. Green Transition and Emerging Demand

### 4.1 Copper Intensity by Technology

Renewable energy and electrification technologies are fundamentally more copper-intensive than their fossil fuel counterparts:

**Copper per unit of generation capacity (approximate):**

| Technology | Copper Intensity | Multiple vs. Conventional |
|------------|-----------------|--------------------------|
| Conventional (gas/coal) | ~1 tonne/MW | Baseline |
| Solar PV | 4--5.5 tonnes/MW | 4--6x |
| Onshore Wind | 2.5--7.5 tonnes/MW | 3--8x |
| Offshore Wind | 8--10.5 tonnes/MW | 8--11x |
| Concentrating Solar (CSP) | 2.6--4.4 tonnes/MW | 3--4x |

Sources: Wikipedia (Copper in Renewable Energy), IEA, Copper Alliance.

**Wind generator technologies vary substantially:**
- Double-fed asynchronous: ~650 kg copper/MW
- Permanent magnet synchronous: 600--2,150 kg copper/MW
- High-temperature superconductor: ~325 kg copper/MW (emerging)

**Offshore wind** is particularly copper-intensive due to submarine power cables (21,000 lbs/MW including cabling), which can represent 50%+ of total copper content.

### 4.2 Electric Vehicles

EVs use substantially more copper than conventional internal combustion engine (ICE) vehicles:

- **Battery Electric Vehicle (BEV)**: ~53--83 kg of copper (motor windings, battery connections, wiring harness, inverter, onboard charger)
- **Plug-in Hybrid (PHEV)**: ~40--60 kg
- **Hybrid (HEV)**: ~28--39 kg
- **Conventional ICE vehicle**: ~15--23 kg
- **EV charging station (Level 2)**: ~1--2 kg copper
- **DC fast charger**: ~8--25 kg copper

The 3--4x copper multiplier per vehicle is a structural demand driver. With global EV sales exceeding 15 million units in 2024 and projected to reach 40+ million by 2030, this represents incremental copper demand of 1.5--3.0 Mt by 2030.

**Key copper components in EVs**: Electric motor (copper wound stator), high-voltage wiring harness, battery bus bars and connectors, power electronics (inverter, DC-DC converter), onboard charger, and the charging infrastructure.

### 4.3 Grid Infrastructure and Electricity Networks

The IEA identifies electricity networks as the single largest source of clean-energy copper demand:
- Grid expansion and modernization to support renewable integration requires massive copper investment.
- Transformers, underground cables, switchgear, and substations are all copper-intensive.
- Emerging markets electrification further adds demand.
- Grid copper demand is projected to grow from ~5 Mt (2023) to 7--9 Mt by 2030 under energy transition scenarios.

### 4.4 Data Centers and AI Infrastructure

This is the newest and potentially most significant emerging demand driver (2024--2026):

- **Power infrastructure**: Each MW of data center capacity requires significant copper for power distribution (busbars, cables, transformers, backup generators, UPS systems).
- **AI-specific demand**: AI training clusters require substantially more power than traditional data centers. A single large AI training cluster can consume 100+ MW. The hyperscaler buildout (Microsoft, Google, Amazon, Meta) is driving unprecedented data center construction.
- **Estimated copper demand**: Industry estimates (Wood Mackenzie, Goldman Sachs) suggest data center copper demand could reach 500 kt--1 Mt annually by 2030, up from ~200 kt in 2023.
- **Indirect demand**: AI-driven electricity demand growth requires generation capacity expansion (often gas + renewables), transmission infrastructure, and grid reinforcement -- all copper-intensive.
- **China context**: Chinese smelters reached record refined copper production of 1.33 million tonnes in March 2026, partly reflecting data center-driven demand.

### 4.5 Total Green Transition Demand Projections

Based on IEA data (Announced Pledges Scenario):
- **2021**: Clean tech copper demand ~5,380 kt (22% of total)
- **2023**: Clean tech copper demand ~6,311 kt (24% of total)
- **2030**: Clean tech copper demand ~12,001 kt (39% of total)
- **2040**: Clean tech copper demand ~16,343 kt (45% of total)

This implies clean tech copper demand roughly doubles from 2023 to 2030, while other uses remain roughly flat. The structural supply gap (31% shortfall in project pipeline) suggests sustained upward price pressure.

---

## 5. Data Sources for Quantitative Research

### 5.1 Copper Price Data

| Source | Ticker/Series | Frequency | Access | URL |
|--------|--------------|-----------|--------|-----|
| **Yahoo Finance** | HG=F (COMEX front-month) | Daily OHLCV | Free (yfinance Python library: `pip install yfinance`) | https://finance.yahoo.com/quote/HG=F/ |
| **FRED** | PCOPPUSDM | Monthly | Free (FRED API) | https://fred.stlouisfed.org/series/PCOPPUSDM |
| **World Bank "Pink Sheet"** | Copper, grade A cathode, LME | Monthly | Free (Excel download) | https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx |
| **World Bank API** | PCOPPUSDM | Monthly | Free (REST API) | `https://api.worldbank.org/v2/en/indicator/PCOPPUSDM?downloadformat=csv` |
| **Nasdaq Data Link (Quandl)** | CHRIS/CME_HG1 | Daily | Free tier available | https://data.nasdaq.com/ |
| **Trading Economics** | Copper | Daily | Subscription (API available) | https://tradingeconomics.com/commodity/copper |
| **LME** | Official settlement prices | Daily | Subscription (LME DataStore) | https://www.lme.com/en/market-data |
| **Investing.com** | Copper | Intraday to monthly | Free (delayed); API for premium | https://www.investing.com/commodities/copper |

**Python access (yfinance example):**
```python
import yfinance as yf

# Download COMEX copper futures daily data
copper = yf.download('HG=F', start='2005-01-01', end='2026-04-27')
# Fields: Open, High, Low, Close, Adj Close, Volume
```

**FRED API access:**
```
# Base URL pattern:
https://api.stlouisfed.org/fred/series/observations?series_id=PCOPPUSDM&api_key=YOUR_KEY&file_type=json
```
Register for a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html

### 5.2 LME Inventory and Forward Curve Data

| Data | Source | Access |
|------|--------|--------|
| LME warehouse stocks (daily) | LME DataStore | Subscription |
| LME forward curve (cash to 3-year) | LME DataStore | Subscription |
| LME cash-3M spread | LME / Bloomberg / Reuters | Subscription |
| SHFE warehouse stocks (weekly) | SHFE website | Free (delayed) |
| COMEX warehouse stocks | CME Group | Free (delayed) |

Note: Historical LME inventory data is sometimes available via academic data providers (Refinitiv/LSEG Datastream, Bloomberg) through university licenses.

### 5.3 Macro-Economic Data

| Series | Source | Series ID | Frequency | URL |
|--------|--------|-----------|-----------|-----|
| **US Dollar Index (DXY)** | Yahoo Finance (yfinance) | DX-Y.NYB | Daily | https://finance.yahoo.com/quote/DX-Y.NYB/ |
| **US Dollar Index** | FRED | DTWEXBGS (broad trade-weighted) | Daily | https://fred.stlouisfed.org/series/DTWEXBGS |
| **ISM Manufacturing PMI** | FRED | MANEMP, NAPM | Monthly | https://fred.stlouisfed.org/ |
| **Caixin China Manufacturing PMI** | S&P Global / Caixin | -- | Monthly | https://www.pmi.spglobal.com/ (subscription) |
| **US Industrial Production** | FRED | INDPRO | Monthly | https://fred.stlouisfed.org/series/INDPRO |
| **China Industrial Production** | NBS China / FRED | -- | Monthly | https://data.stats.gov.cn/ |
| **Fed Funds Rate** | FRED | FEDFUNDS | Monthly | https://fred.stlouisfed.org/series/FEDFUNDS |
| **10-Year Treasury Yield** | FRED | DGS10 | Daily | https://fred.stlouisfed.org/series/DGS10 |
| **VIX** | Yahoo Finance | ^VIX | Daily | https://finance.yahoo.com/quote/%5EVIX/ |
| **S&P 500** | Yahoo Finance | ^GSPC | Daily | https://finance.yahoo.com/quote/%5EGSPC/ |
| **Crude Oil (WTI)** | Yahoo Finance / FRED | CL=F / DCOILWTICO | Daily | https://fred.stlouisfed.org/series/DCOILWTICO |
| **Baltic Dry Index** | Various | -- | Daily | Subscription (Baltic Exchange); some free historical via investing.com |
| **BIS Effective Exchange Rates** | BIS | NEER/REER (64 economies) | Daily/Monthly | https://data.bis.org/topics/EER |
| **CPI / Inflation** | FRED | CPIAUCSL | Monthly | https://fred.stlouisfed.org/series/CPIAUCSL |

### 5.4 Chinese Economic Data

China accounts for >50% of global copper consumption. Key sources:

| Data | Source | URL |
|------|--------|-----|
| China PMI (Official NBS) | National Bureau of Statistics | https://data.stats.gov.cn/ |
| Caixin Manufacturing PMI | S&P Global / Caixin | Subscription; press releases free |
| Fixed Asset Investment | NBS | https://data.stats.gov.cn/ |
| New Housing Starts | NBS | https://data.stats.gov.cn/ |
| Credit data (TSF, M2) | PBOC | http://www.pbc.gov.cn/en/ |
| Copper imports (customs) | China Customs (GACC) | Subscription; monthly summaries free via news |
| SHFE copper inventory | SHFE | https://www.shfe.com.cn/en/ |

**Free aggregators of Chinese data:**
- CEIC (limited free access): https://www.ceicdata.com/
- Trading Economics China page: https://tradingeconomics.com/china/indicators
- FRED China series: search "China" at https://fred.stlouisfed.org/

### 5.5 Commodity and Cross-Market Data

| Series | Ticker (yfinance) | Relevance to Copper |
|--------|-------------------|---------------------|
| Gold | GC=F | Safe-haven counterpart; copper/gold ratio = macro signal |
| Silver | SI=F | Industrial + precious metal hybrid |
| Aluminum | ALI=F | Co-moves with copper (base metal complex) |
| Zinc | -- | Base metal peer |
| Nickel | -- | Base metal peer; EV battery metal |
| Crude Oil (WTI) | CL=F | Energy cost input to mining/smelting; macro co-movement |
| Crude Oil (Brent) | BZ=F | Global oil benchmark |
| Natural Gas | NG=F | Energy cost for smelting |
| Iron Ore | -- | China construction proxy |
| US 10Y Bond | ^TNX | Interest rate proxy |

**Note**: LME metals (aluminum, zinc, nickel, tin, lead) are available via Bloomberg/Reuters or Nasdaq Data Link with subscription. Free daily LME prices are sometimes published by metals news sites.

### 5.6 Institutional Reports (Free / Partially Free)

| Organization | What They Publish | URL |
|-------------|-------------------|-----|
| **ICSG** (International Copper Study Group) | Copper Market Forecast, Statistical Yearbook, Monthly Bulletin, Mine/Smelter/Refinery maps | https://www.icsg.org/ |
| **USGS** | Mineral Commodity Summaries (annual, free PDF) | https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-copper.pdf |
| **World Bank** | Commodity Markets Outlook (quarterly, free) | https://www.worldbank.org/en/research/commodity-markets |
| **IEA** | Global Critical Minerals Outlook, Global EV Outlook | https://www.iea.org/reports/copper |
| **CRU** | TC/RC benchmarks, supply-demand balance | Subscription |
| **Wood Mackenzie** | Mine-level production, cost curves, demand forecasts | Subscription |
| **S&P Global** | Metals news, analysis, TC/RC tracking | Subscription |

---

## 6. Underexplored Features for Price Forecasting

This section identifies signals that are rarely used in academic copper forecasting models but may provide significant predictive power.

### 6.1 Cross-Market Signals

Most copper forecasting papers use copper's own price history plus a few macro variables. The following cross-market relationships are under-exploited:

- **LME-COMEX spread**: Divergences signal regional supply-demand imbalances. The 2024--2025 episode where COMEX premiums spiked on tariff fears is an example of a regime shift that cross-exchange spread data would capture.
- **SHFE-LME arbitrage spread**: Directly signals Chinese import appetite. When the arb is open (SHFE > LME adjusted), Chinese demand is pulling copper eastward.
- **Copper-aluminum spread**: Aluminum is a partial substitute for copper in some electrical applications. When copper becomes too expensive relative to aluminum, substitution occurs (e.g., in power cables, heat exchangers).
- **Base metals complex co-movement**: Aluminum, zinc, nickel, tin, lead all share macro drivers with copper. Multi-metal GNN/graph models can capture spillover effects.
- **Copper scrap spread (No. 2 vs. cathode)**: Encodes physical market tightness and recycling economics. Rarely used in academic models.

### 6.2 Options Market Signals

Copper options on COMEX futures contain rich information about market expectations:

- **Implied volatility (IV)**: Forward-looking measure of expected price uncertainty. Rising IV often precedes large price moves.
- **IV term structure**: Relationship between short-dated and long-dated IV. Inversion (short > long) signals near-term stress.
- **Volatility skew**: The difference in IV between out-of-the-money puts and calls. A steep put skew signals asymmetric downside risk pricing; a call skew signals upside risk pricing.
- **Put-call ratio**: Ratio of put volume (or open interest) to call volume. Elevated ratios signal bearish sentiment or hedging demand.
- **Risk reversal (25-delta)**: The IV of 25-delta calls minus 25-delta puts. Positive = market pricing more upside risk; negative = more downside risk.

**Data access**: CME Group provides options data (delayed) at https://www.cmegroup.com/markets/metals/base/copper.quotes.options.html. Historical options data via CME DataMine (paid) or academic subscriptions (OptionMetrics, via WRDS).

### 6.3 CFTC Commitments of Traders (COT) Positioning Data

The COT report reveals the positions of major market participant categories in COMEX copper futures:

**Report types:**
- **Legacy report**: Commercial vs. Non-Commercial vs. Non-Reportable.
- **Disaggregated report** (since 2006): Producer/Merchant/Processor/User, Swap Dealers, Managed Money, Other Reportables.

**Key signals:**
- **Managed Money net positioning**: Represents speculative/hedge fund positions. Extreme net long positions may signal crowded trades and potential mean-reversion. Extreme net short positions may signal bearish consensus near potential bottoms.
- **Commercial net positioning**: Represents producer/consumer hedging. Unusually large commercial short positions (producer hedging) may signal expected price weakness.
- **Changes in positioning**: Weekly changes in net positions can anticipate price moves. Rapid unwinding of speculative longs often precedes sell-offs.
- **Concentration ratios**: The share of open interest held by the top 4 or top 8 traders. High concentration signals potential for outsized impact from individual position changes.

**Data access:**
- CFTC website (free, weekly): https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- Public Reporting Environment: https://publicreporting.cftc.gov/ (CSV, XML, TSV exports)
- Historical data from 1986 (Legacy) and 2006 (Disaggregated): https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- Python library: `cot_reports` or direct download and parsing of text files.

**Copper identification in COT data:** Look for "COPPER-GRADE #1" under COMEX (CME Group exchange) in the disaggregated futures report. The contract code is typically "HG" with 25,000 lbs per contract.

### 6.4 Satellite and Alternative Data

Emerging data sources that could provide edge in copper forecasting:

- **Satellite imagery of mine sites**: Track mine activity levels, stockpile sizes, truck movements. Companies like Orbital Insight, Planet Labs, and Ursa Space provide these services.
- **Satellite imagery of smelter activity**: Nighttime light emissions and thermal signatures from smelters can indicate capacity utilization.
- **Shipping/AIS data**: Track copper cargo vessel movements between major ports (Chilean ports to China, DRC copper belt to Dar es Salaam). MarineTraffic, VesselFinder provide AIS data. Vessel loading data at key copper ports (Antofagasta, Mejillones, Callao) can be a leading indicator.
- **Chinese port copper arrivals**: Track copper cathode and concentrate unloading at major Chinese ports.
- **Power consumption at mining regions**: Electricity demand in Chilean mining regions can proxy for mine output ahead of official production data.
- **Google Trends**: Search interest for "copper price" or related terms has shown marginal predictive value in some studies.
- **News sentiment (NLP)**: Automated extraction of copper-related news sentiment from Reuters, Bloomberg, FT, and metals trade publications (Metal Bulletin, Fastmarkets). Few academic copper models incorporate NLP-based sentiment.
- **Chinese social media sentiment**: Weibo/WeChat mining industry sentiment could signal Chinese demand expectations.

### 6.5 Forward Curve Features

The shape and dynamics of the LME forward curve encode substantial information:

- **Cash-3M spread (level and change)**: Direct measure of physical tightness.
- **3M-15M spread**: Captures medium-term supply expectations.
- **Forward curve curvature**: Second derivative of the term structure. Unusual curvature patterns can signal anticipated structural shifts.
- **Roll yield**: The return from rolling futures contracts. Persistent backwardation generates positive roll yield, attracting speculative inflows.
- **Term structure momentum**: Changes in the shape of the forward curve over time.

### 6.6 Cross-Frequency Signals (Relevant to VMD-GNN Framework)

Building on the literature gap analysis, these cross-frequency relationships are economically motivated but have not been modeled:

- **Low-frequency DXY vs. high-frequency copper**: The macro trend in the dollar may predict short-term copper dynamics at a lag.
- **Low-frequency Chinese PMI vs. medium-frequency copper**: China's manufacturing cycle operates at multi-month frequencies but affects copper at shorter horizons.
- **High-frequency inventory changes vs. medium-frequency price trends**: Rapid inventory draws at LME warehouses may signal the start of multi-week price rallies.
- **Cross-metal frequency alignment**: When low-frequency modes of copper, aluminum, and zinc align (e.g., all trending up), it signals a broad base metals rally driven by macro forces. Divergence may signal copper-specific factors.

These cross-frequency relationships provide a direct motivation for the VMD-MFGNN framework described in the literature gap analysis: decomposing variables into frequency modes and building frequency-specific graphs captures exactly these relationships.

---

## 7. Summary: Key Takeaways for Forecasting Model Design

### Feature Categories (Ordered by Likely Importance)

1. **Price dynamics**: Copper's own lagged returns, volatility, momentum, mean-reversion signals
2. **Physical market**: LME/SHFE/COMEX inventory levels and changes, cash-3M spread (contango/backwardation), TC/RC levels
3. **Macro drivers**: DXY, Chinese PMI (Caixin Manufacturing), US PMI (ISM), Fed Funds rate, 10Y yield
4. **Cross-market**: Other base metals (Al, Zn, Ni), crude oil, gold, S&P 500, VIX, Baltic Dry Index
5. **Positioning**: COT managed money net long, changes in positioning, concentration
6. **Options-derived**: Implied volatility, skew, put-call ratio, risk reversal
7. **Alternative**: Satellite/AIS shipping data, news sentiment, Google Trends

### Recommended Data Stack (Freely Accessible)

For a research paper, the following data can be assembled entirely from free sources:

| Variable | Source | Access Method |
|----------|--------|---------------|
| Copper price (daily) | Yahoo Finance | `yfinance` Python library, ticker HG=F |
| Copper price (monthly, long history) | FRED | API, series PCOPPUSDM |
| Gold, Silver, Oil, S&P 500, VIX | Yahoo Finance | `yfinance`, tickers GC=F, SI=F, CL=F, ^GSPC, ^VIX |
| DXY (dollar index) | Yahoo Finance | `yfinance`, ticker DX-Y.NYB |
| US Industrial Production | FRED | API, series INDPRO |
| ISM Manufacturing PMI | FRED | API, series NAPM |
| Fed Funds Rate | FRED | API, series FEDFUNDS |
| 10Y Treasury Yield | FRED | API, series DGS10 |
| CPI | FRED | API, series CPIAUCSL |
| Trade-weighted USD | FRED | API, series DTWEXBGS |
| COT positioning | CFTC | Weekly CSV download |
| World Bank copper price (long history) | World Bank | Excel/API download |

### Current Market Snapshot (April 2026)

- **Price**: ~$6.07/lb ($13,380/tonne); near all-time highs
- **52-week range**: $4.33--$6.58/lb
- **1-year return**: +24--25%
- **All-time high**: $6.58/lb (January 2026)
- **TC/RC**: Near historic lows, signaling tight concentrate market
- **Chinese smelter output**: Record 1.33 Mt in March 2026
- **Market structure**: Fundamentally tight; green transition demand accelerating
- **Key risks**: US tariff uncertainty, China property sector weakness, global recession risk, US dollar strength

---

## References and Key URLs

### Exchange Websites
- LME: https://www.lme.com/en/metals/non-ferrous/lme-copper
- CME Group (COMEX): https://www.cmegroup.com/markets/metals/base/copper.html
- SHFE: https://www.shfe.com.cn/en/

### Free Data Sources
- FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org/
- FRED API documentation: https://fred.stlouisfed.org/docs/api/fred/
- World Bank Commodity Data: https://www.worldbank.org/en/research/commodity-markets
- World Bank Monthly Data (Excel): https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx
- World Bank Annual Data (Excel): https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Annual.xlsx
- CFTC COT Reports: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- CFTC Public Reporting: https://publicreporting.cftc.gov/
- Yahoo Finance (yfinance): https://pypi.org/project/yfinance/
- USGS Mineral Commodity Summaries: https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-copper.pdf
- EIA Open Data API: https://www.eia.gov/opendata/
- BIS Exchange Rates: https://data.bis.org/topics/EER

### Industry and Research
- ICSG: https://www.icsg.org/
- IEA Copper Report: https://www.iea.org/reports/copper
- IEA Global Critical Minerals Outlook: https://www.iea.org/reports/global-critical-minerals-outlook-2024
- International Copper Association: https://internationalcopper.org/
- Trading Economics Copper: https://tradingeconomics.com/commodity/copper
- Investing.com Copper: https://www.investing.com/commodities/copper
