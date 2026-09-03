# Copper Forecasting Project — Complete Teaching Primer

> **How to use this document:** This is written to be self-contained — it assumes you know nothing about this project going in. You can read it top to bottom yourself, or paste the whole thing into ChatGPT/Claude/any LLM with the instruction "teach me this project interactively, quiz me on each section before moving to the next, and let me ask questions" for a guided walkthrough instead of a passive read. Either way, by the end you should understand not just *what* was built but *why* every major decision was made, and be able to explain it to your professor or a reviewer without hand-waving.

---

## Part 0: What this project is, in one paragraph

This is a B.Tech Capstone research project (NIT Rourkela, CS4095) about **forecasting copper prices** using machine learning. It went through two major phases: first, an ambitious deep-learning architecture (called VMD-MFGNN) that was built, trained on 15 years of real data, and found — through rigorous testing — to fundamentally not work, for reasons that were precisely diagnosed rather than just observed. Second, a pivot to a simpler, more rigorously-evaluated approach (called CuBench) built after extensive research into what actually works in this field, currently mid-execution. The throughline connecting both phases is **rigor**: at every stage, the project caught real bugs and false claims by insisting on independently verifying results rather than trusting that code "should" work.

---

## Part 1: Why copper, and why is forecasting it hard?

**Why copper matters**: Copper is used in almost everything electrical — wiring, motors, and especially the energy transition (electric vehicles use 3-4x more copper than gas cars; wind and solar power need far more copper per unit of energy than fossil fuels). China consumes over half the world's copper. Because of this, copper price movements are watched as a signal of industrial/economic health (nicknamed "Dr. Copper").

**Why it's hard to forecast, specifically**: This is the single most important fact underlying every methodological decision in this project. A formal statistical test (a "variance-ratio test") was applied to copper prices on the London Metal Exchange, and it **failed to reject the hypothesis that copper's daily price movements are a random walk** — meaning, statistically, there's no strong evidence that tomorrow's direction (up or down) is predictable from past prices. This isn't unique to copper — it's consistent with the "weak-form efficient market hypothesis," which says liquid, widely-traded markets tend to absorb information quickly, leaving little exploitable pattern in price history alone.

**But volatility (how *much* prices move) is a different story.** Even if you can't predict *direction*, there's strong, well-established evidence that volatility is more predictable — it "clusters" (calm periods tend to stay calm, turbulent periods tend to stay turbulent). This distinction — direction is hard, volatility is more tractable — becomes the central design choice of the whole second phase of this project.

**Key term: "random walk"** — a mathematical model where the next value is unpredictable from the current one, like a drunk person's steps. If a market is a random walk, no amount of clever modeling of past prices alone will reliably predict the next move.

---

## Part 2: Phase One — VMD-MFGNN (the deep learning attempt)

### 2.1 What the model was trying to do

The idea: combine two techniques.

1. **VMD (Variational Mode Decomposition)** — a signal-processing technique that splits a noisy time series into several cleaner "frequency bands" (think of it like separating a messy audio recording into bass, mid, and treble). The intuition: copper's price is a mix of slow macro trends and fast noisy fluctuations, and separating them might make each easier to model.

2. **GNN (Graph Neural Network)** — a type of neural network that operates on data with *relationships* between entities (a "graph" of nodes and edges), rather than a flat list of numbers. The idea here: build a graph where each node is a market variable (copper, gold, oil, the dollar index, etc.), and let the network *learn* which variables influence each other, separately at each frequency band from step 1.

**Key term: "graph" (in ML)** — not a chart/plot. A data structure of *nodes* (things) connected by *edges* (relationships), often with edge "weights" representing how strong a relationship is. A GNN learns to use this structure when making predictions — e.g., "copper's price is influenced 70% by aluminum and 30% by oil at this frequency."

**Key term: "learned adjacency"** — instead of a human deciding the graph's connections in advance, the model learns them from data during training (via trainable parameters), theoretically discovering real relationships automatically.

### 2.2 What went wrong (in order of discovery — teaching moment: layered bugs)

This is worth understanding in detail because the debugging story itself is a lesson in rigor.

1. **The VMD decomposition was "stale."** Due to a coding shortcut (to avoid rerunning an expensive computation every single day), the decomposition was recomputed only every 21 days and held constant in between — so instead of smooth daily-varying signal, the model was fed a "staircase" that barely changed within each 21-day block. This threw away most of the temporal detail the whole VMD idea was supposed to provide.

2. **A baseline model (ARIMA) had a unit-conversion bug** making its errors look artificially huge — inflating how good the *other* models looked by comparison.

3. **Reported model-size numbers didn't match the actual configuration** used to produce them — a factual inconsistency in early paper drafts, caught by independent review.

4. **An ablation study** (a controlled experiment that removes one part of a model to see if it matters) claimed two model variants had *equal* capacity when they actually didn't — undermining the fairness of that comparison.

5. **The single worst bug, found last and independently verified**: the GNN's core mechanism — the "learned graph" — was **structurally incapable of learning**. A specific coding detail (an argument passed positionally instead of by name to a library function) caused the model to silently *discard* the very information (edge weights) that was supposed to encode "how strongly do these two variables relate." The graph-learning layers received **zero gradient** during training — meaning, in the language of neural network training, they never updated from their random starting point, ever, across the entire training run. This was verified directly: the code checked the gradient values after a training pass and found them literally `None` for every single one of these layers.

**Key term: "gradient"** — in neural network training, a gradient tells each parameter which direction and how much to change in order to reduce prediction error. If a parameter never receives a gradient, it never learns anything — it's frozen at its random initial value forever, regardless of how much data you feed the model.

**Consequence**: the model's headline claim — "we built a graph that learns meaningful relationships between market variables at different frequencies" — had literally never happened. The "interpretability" figures showing which variables the model supposedly learned to weight heavily were, in fact, visualizing random noise from initialization, not learned structure.

A fix was engineered (properly wiring the edge weights through) and tested. It **mechanically worked** — the crash/discard bug was gone — but the graph *still* didn't learn anything meaningful, because the deeper problem was that the actual prediction task's signal reaching those parameters was too weak to move them meaningfully even once the plumbing was fixed. This is an important nuance: **fixing an obvious bug doesn't guarantee the underlying idea works** — sometimes the bug was hiding a more fundamental limitation, not causing the whole problem by itself.

### 2.3 A second, separate class of bug: fake accuracy

Independent of the graph bug, the project found that several models — including a comparison "Transformer" baseline — were reporting seemingly good "directional accuracy" (correctly guessing up/down) that turned out to be an illusion. **These models were just predicting the same direction almost every single time**, and that constant guess happened to match how often the market actually moved that direction during the test period. This looks identical to real skill on a simple accuracy scorecard, but it's actually zero skill — like a weighted coin that always lands heads, correctly "predicting" heads 70% of the time in a period where the coin genuinely landed heads 70% of the time, with no actual foresight involved.

**Key term: "base-rate" / "majority class"** — the frequency of the more common outcome in your test data. If 70% of days in your test period were "up" days, a model that always says "up" gets 70% accuracy — indistinguishable, on the accuracy metric alone, from a model with real 70%-confidence skill.

A diagnostic was built to catch this: check whether a model's predictions are suspiciously one-sided (a "sign-concentration" check plus a formal statistical test), and flag any result whose accuracy exactly matches the base rate as a suspected artifact rather than real skill. Applying this diagnostic to the project's *own* earlier results found six contaminated result cells, two of which had been bolded as "best" in a table before the check was applied.

### 2.4 The honest conclusion of Phase One

Rather than hide these findings, the project wrote them up as the actual paper: "we built a sophisticated architecture, rigorously diagnosed exactly why its core mechanism doesn't work, and built a reusable diagnostic for catching a specific kind of fake-accuracy result along the way." This is a legitimate, if less glamorous, contribution — a negative result with a precise, verified explanation is more valuable to the field than a positive result nobody stress-tested. This paper (call it "Paper 1") is complete, reviewed, and ready for submission — but it is not what the rest of this document is about.

---

## Part 3: The Pivot — why the project didn't stop at Paper 1

After Paper 1's honest-but-negative result, there was a natural temptation: can we find something that actually *works*, not just something that rigorously *doesn't*? Rather than guess, the project ran **six parallel research tracks**, each investigating the question from a different angle, using web search and literature review:

1. **Copper market fundamentals** — supply/demand structure, inventories, historical price shocks.
2. **Practitioner/trading-desk view** — what real commodity trading firms actually rely on.
3. **Rigorous econometrics** — what the serious academic finance literature says is actually predictable, and by how much.
4. **Modern ML benchmarks** — which model families (not just deep learning) actually win in large, careful studies.
5. **Macro/geopolitical indicators** — which broader economic signals genuinely relate to copper.
6. **Publication standards** — what makes a forecasting claim credible enough to survive peer review, versus what gets rejected as likely-leakage.

### 3.1 What these tracks converged on (independently, which is the important part)

Several tracks reached the *same* conclusions without seeing each other's work — this convergence is itself evidence the conclusions are sound, not cherry-picked:

- **Gradient-boosted decision trees beat deep learning on this kind of data.** A large study (18 million data points across 10,000+ securities) found tree-based models (like LightGBM, XGBoost, CatBoost) decisively outperform both deep neural networks and even specialized "foundation models" on daily financial prediction tasks. The intuition: deep learning shines with huge amounts of data and complex patterns; financial daily data is comparatively small and noisy, which favors simpler, more regularized methods.

  **Key term: "gradient boosting"** — an ML technique that builds many small decision trees in sequence, where each new tree focuses on correcting the errors of the trees before it. Distinct from deep learning (neural networks) — no "layers" of artificial neurons, no backpropagation through a network architecture.

- **Volatility is the more forecastable target, confirmed from multiple independent angles** — the market-efficiency test (Part 1), the practitioner literature (real trading desks forecast volatility and size bets accordingly, rather than betting on raw direction), and formal academic surveys all agreed.

- **A credible, defensible target range exists, and it's much lower than intuition suggests.** A referee experienced in this field would find a model claiming **50s-percent directional accuracy** believable; a model claiming **70%+ accuracy** would be treated as a red flag suggesting a data leak, not a breakthrough. This directly informs how any future result must be presented: modest, statistically well-supported claims are more credible (and more publishable) than flashy ones.

- **A specific, useful reframe for reputation-building**: papers that become influential in this field are often remembered for a *rigor tool* (a statistical test, a diagnostic, a benchmark methodology) rather than for a specific model's accuracy number. The base-rate diagnostic built in Phase One (Part 2.3) is exactly this kind of contribution — though it was later found that a very similar idea had already been published by someone else in 2026, so the project's framing shifted from "we invented this" to "we independently discovered and extended this, and we cite the prior work properly." (This is itself a lesson: check the literature before claiming novelty, and correct course honestly when you find you weren't first.)

### 3.2 The decision that followed

Given all this, the project made a deliberate pivot: **build a new, second research effort — code-named "CuBench" — targeting realized volatility forecasting with gradient-boosted trees, evaluated with much more statistical rigor than is typical in this field.** This does not discard Paper 1; it becomes a second, connected paper that cites the first as the reason a graph-based deep-learning approach was ruled out.

---

## Part 4: CuBench — the actual architecture being built

### 4.1 The prediction targets

Rather than one target ("predict tomorrow's price"), CuBench predicts three different things, in order of how confident the research says we should be in each:

1. **T1 — Realized volatility** (primary target): how much will the price move over the next 1, 5, or 22 trading days, regardless of direction? This is the "we expect this to actually show real, defensible predictive skill" target.
2. **T2 — Return quantiles**: instead of a single number, predict a *range* of plausible outcomes (e.g., "there's a 10% chance the return is below X, a 50% chance it's below Y," etc.). This is "quantile regression" / probabilistic forecasting — more honest than a single point guess, since it communicates uncertainty.
3. **T3 — Direction** (secondary/diagnostic target): predicted purely for completeness and honest reporting, with the expectation, going in, that it will look close to a coin flip — and if it doesn't, that's treated as a reason for suspicion (possible leakage), not celebration.

**Key term: "quantile regression"** — instead of predicting one number, you predict specific percentile cutoffs of the outcome distribution (e.g., the 10th percentile, 50th percentile/median, 90th percentile). Useful for expressing a *range* of confidence rather than false precision.

### 4.2 The features (inputs to the model)

About 71 engineered input variables, grouped into four "blocks":

- **Block A — Trend/momentum**: things like "how much has the price moved over the last 5/20/60/250 days" — capturing whether the market is trending.
- **Block B′ — Curve proxy**: originally intended to capture "contango vs. backwardation" (a futures-market concept about whether future-dated contracts are priced above or below the current price, which can signal supply tightness) — but the real data needed for this (individual futures contract prices going back to 2010) turned out not to be freely available after checking 192 possible data sources, so this was demoted to a small, clearly-labeled "proxy" block used only in side experiments, not the main model. **This is a good example of a plan adapting to a real, checked constraint rather than faking data.**
- **Block C — Macro/cross-market**: other assets (gold, silver, oil, the dollar index, interest rates, credit spreads) and broader economic indicators (industrial production, producer prices). Each of these was empirically tested against real data before being trusted — several literature claims about these relationships turned out to be overstated or unstable when checked (see Part 4.4 below).
- **Block D — Volatility/regime**: statistical measures of how volatile the market currently is, including outputs from classical models like GARCH (a well-established statistical volatility model), and a simple 3-way "regime" label (is volatility currently low, medium, or high compared to recent history).

**Key term: "GARCH"** — Generalized Autoregressive Conditional Heteroskedasticity. A classical statistical model (not machine learning) specifically designed to forecast volatility by modeling how today's volatility depends on recent volatility and recent shocks. It's the closest thing to a gold-standard baseline for the T1 target — the project's research found that a well-tuned classical GARCH/HAR model is genuinely hard for fancier ML to beat, and treats it as the number to beat, not a strawman to easily defeat.

**Key term: "HAR-RV"** (Heterogeneous Autoregressive Realized Volatility) — a simple, well-regarded statistical model for forecasting volatility using volatility measured over different recent windows (yesterday, last week, last month). Despite its simplicity, real published research on this exact asset (COMEX copper) found it hard to beat, so the project treats it as the primary benchmark.

### 4.3 The models being trained and compared (the full roster)

- **"Null" baselines** — trivially simple reference points (e.g., "predict tomorrow will look like today," "always predict the majority outcome"). Every real model must beat these to mean anything.
- **Classical statistical models** — HAR-RV, GARCH, ARIMA. These are the "old but respected" methods.
- **Linear ML** — ElasticNet, a regularized linear regression.
- **Gradient-boosted trees** — LightGBM (the primary model, chosen based on the research), plus XGBoost, CatBoost, RandomForest as comparisons.
- **Deep learning** — LSTM and Transformer models, included as comparison points (not the star of the show this time — carried over/retrained from Paper 1's codebase, on a reduced subset of targets/horizons for practical time reasons).

### 4.4 Verifying the literature with real data (a key rigor step)

Rather than trusting research-paper claims blindly, the project **downloaded 16 years of real market data and directly tested every claimed relationship** before building features around them. Some interesting, real findings:

- The "US Dollar Index is copper's most reliable macro signal" claim was only partially true — the correlation was real but swung wildly over different time periods, not stable as claimed.
- "Gold and silver have great influence on copper" was found to be overstated — real correlations were modest, no more than several other, less-hyped variables.
- **VIX (the market "fear index") turned out to be the single most consistent, stable relationship found** — more reliable than the flashier claims from the literature.
- No candidate variable was found to meaningfully "lead" copper (predict it in advance) at a daily frequency — everything moves roughly at the same time, which has a direct practical consequence: features should be built same-day, not artificially lagged.

**Lesson embedded here**: literature claims are a starting point for hypotheses, not ground truth — always check against your own data before building a pipeline around a claim.

### 4.5 The evaluation rigor (this is what makes CuBench a credible research contribution, not just "another model")

This is arguably the most important part of the whole project to understand, because it's the actual *contribution* — not the specific model choice.

- **Walk-forward validation across 11 years (2015–2025)**: instead of one train/test split, the model is retrained repeatedly, always only using data from before the test period (never peeking at the future) — "expanding window." This tests whether a method holds up across genuinely different market regimes (2015-16 commodity bust, 2020 COVID crash, 2022 inflation shock, etc.), not just one lucky period.

  **Key term: "leakage"** — when information from the future (or from the test set) accidentally influences a model during training, making results look artificially better than they'd be in real, honest use. A huge portion of this project's rigor work exists specifically to prevent leakage — e.g., a monthly economic indicator must only be used starting from the date it was *actually published*, not the date it *describes*, or the model would be silently cheating by seeing data before it existed in the real world.

- **Statistical significance testing, not just "which number is bigger."** Two models' results are compared using a formal test (the Diebold-Mariano test, with a small-sample correction) rather than just eyeballing which one has a lower error. This test was independently validated against a trusted reference implementation to make sure it was coded correctly (agreement to a tiny fraction of a percent).

- **Correcting for testing many things at once.** If you compare 30+ model/horizon combinations, some will look "significant" purely by chance, the same way flipping enough coins will eventually produce a suspicious-looking streak. A correction (Holm-Bonferroni) is applied to account for this, so a reported "win" is genuinely trustworthy, not a statistical fluke.

- **Checking whether a good-looking backtest is actually overfit.** A model can look great on paper by accident — if you tried enough variations, one will eventually look good by luck alone, the same issue as above but applied to trading-strategy backtests specifically. A specific check (Probability of Backtest Overfitting, PBO) is applied. **A real, important finding already surfaced from this**: LightGBM's trading-strategy backtest looked realistic on the surface (a plausible-looking risk-adjusted return) but got flagged by this deeper check as *likely overfit* — exactly the kind of catch this whole rigor apparatus exists to make. This will be an honest, reportable finding in the eventual paper, not something to hide.

- **The base-rate/fake-accuracy diagnostic from Part 2.3**, now applied systematically to every single result cell in the new project too, not just retroactively to catch mistakes.

### 4.6 Why "gradient boosting on CPU" instead of "deep learning on GPU"

An important practical point that surprised the user earlier in the project: **none of CuBench needs a GPU.** This was a deliberate design decision, not a limitation worked around — the research found that gradient-boosted trees genuinely outperform deep learning on this kind of data, so there was no reason to need GPU-heavy training. Everything runs on ordinary CPU. The reason things still took a long time locally wasn't a GPU shortfall — it was CPU/memory contention from training multiple models at once on a single laptop, which is why the work eventually moved to Google Colab (more computing headroom, can run for hours without needing the laptop to stay on).

---

## Part 5: The build process — six phases, and the pattern of catching real bugs

The actual implementation was broken into six phases, each built by an AI coding agent and (where feasible) independently checked. The recurring, important pattern: **at nearly every single phase, something that looked correct on first pass turned out to have a real bug, caught only because the project insisted on actually running things and checking real output, rather than trusting plausible-looking code.**

1. **Phase 2 — Data & features**: built the 71 features and the safety tests that check for leakage. These tests caught two real bugs — a subtle look-ahead leak in a volatility calculation, and a data-alignment bug that had silently zeroed out several features.
2. **Phase 3 — Models**: built all the models listed in 4.3. Caught a units-scale bug and a metric-reporting bug during testing. Local computer hit a genuine hardware throughput limit partway through — not a bug, just not enough CPU power to finish everything quickly.
3. **Phase 4 — Statistical evaluation**: built and validated all the significance tests, the overfitting checks, and the base-rate diagnostic (validated against reference implementations, per section 4.5). This is where the LightGBM "likely overfit" finding was caught.
4. **Phase 5 — Figures**: built the visualizations for the eventual paper, from real (partial, at the time) data. Caught a display bug where one extreme, real outlier value would have visually distorted every other bar in a chart if not handled carefully.
5. **Phase 6 — Colab packaging**: wrapped the entire pipeline into a single notebook so the full, much larger computation could run on Google Colab instead of the local laptop.
6. **A resume-logic bug, found after the first real Colab run**: the first full Colab run *did* successfully train every single model (2,838 total result combinations, fully complete) — but a session restart during the run wiped the raw prediction files while the summary statistics survived (only the summaries had been backed up along the way). Worse, the code that decides "is this already done, or do I need to retrain it?" only checked for the existence of a summary line, not the actual prediction file — meaning if left unfixed, the project would have permanently believed the work was done and never regenerated the missing files. This was fixed at the root: a result now only counts as "done" if both the summary *and* the actual prediction file genuinely exist, and the backup process now also saves the prediction files, not just the summaries.

**The teaching point of this whole section**: rigor isn't a one-time checklist you complete and move past — it's a discipline applied continuously, and it keeps paying off. Every single phase found something real. That's not this project being unusually buggy; it's what actually happens whenever anyone builds something complex and *looks carefully*. Most projects don't look carefully enough to notice.

---

## Part 6: Where things stand right now

- **Paper 1** (VMD-MFGNN, the negative result with the precise diagnosis) is complete, reviewed, and ready for submission — currently paused, not abandoned.
- **CuBench**'s entire pipeline (data → features → models → statistics → figures → Colab packaging) is built and was already proven to work correctly, phase by phase, on real local data.
- **The full-scale run on Google Colab has completed model training successfully once already** (all 2,838 result combinations produced), but that specific run's raw prediction files were lost to a session restart before download — the underlying numbers/statistics survived, but the ability to independently re-verify them from scratch did not.
- **A fix has been made and pushed to the project's GitHub repository** so that this can't happen again, and the next Colab run will regenerate the full grid correctly, with the raw prediction files properly preserved this time.
- **Next concrete step**: re-run the (now-fixed) Colab notebook from a fresh session, following the project's runbook document, to get a complete and independently-verifiable result set — after which the real findings (including the "LightGBM looks good but is actually likely overfit" result) get written up into the CuBench manuscript.

---

## Part 7: Glossary (quick reference)

| Term | Plain-language meaning |
|---|---|
| Random walk | A process where the next value can't be predicted from the current one |
| Weak-form market efficiency | The idea that a market's own price history contains little exploitable predictive information |
| VMD | Splits a noisy time series into cleaner frequency "bands" |
| Graph Neural Network (GNN) | A neural network that operates on data with relationships (a graph), not just flat numbers |
| Gradient | The signal that tells a neural network parameter how to update during training; zero gradient = never learns |
| Gradient boosting | An ML technique building many small decision trees in sequence, each correcting the last's errors |
| GARCH | A classical statistical model for forecasting volatility |
| HAR-RV | A simple, respected statistical volatility-forecasting model used as the benchmark to beat |
| Quantile regression | Predicting a range/distribution of outcomes instead of one number |
| Leakage | Future or test information accidentally influencing training, making results falsely look better |
| Walk-forward validation | Testing a model repeatedly across time, always training only on the past |
| Diebold-Mariano test | A statistical test for whether one model's errors are significantly smaller than another's |
| Holm-Bonferroni correction | A statistical safeguard against false "wins" when testing many comparisons at once |
| Probability of Backtest Overfitting (PBO) | A check for whether a good-looking trading strategy backtest is actually a statistical fluke |
| Base-rate / constant-forecast artifact | Fake-looking "accuracy" caused by a model always guessing the more common outcome |
| Realized volatility | A measure of how much an asset's price actually moved over a given period |

---

## Part 8: Suggested self-test questions

If you're using this document to actually learn (not just skim), try answering these before checking the sections referenced:

1. Why does this project target volatility instead of price direction? (Part 1, Part 3.1)
2. What specifically caused the GNN's core mechanism to never learn anything, and how was this proven rather than just suspected? (Part 2.2, item 5)
3. What is a base-rate/constant-forecast artifact, and why doesn't a simple "check if predictions vary a lot" catch it? (Part 2.3)
4. Why is HAR-RV treated as a serious benchmark to beat rather than an easy strawman? (Part 4.2, Part 4.5)
5. What real data-driven finding overturned a literature claim about the US Dollar Index? (Part 4.4)
6. What was the actual root cause of the Colab prediction-files-lost incident, and how was it fixed (not just patched around)? (Part 5, item 6)
7. Why doesn't this project need a GPU anywhere? (Part 4.6)
