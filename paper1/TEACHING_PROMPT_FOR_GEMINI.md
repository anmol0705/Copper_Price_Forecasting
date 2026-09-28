# Teaching prompt — paste this into a new Gemini chat

(Attach `main.tex` or `main.pdf`, plus `Copper_Paper1.ipynb`, before sending this. Do not attach any `src/` code yet — that comes one file at a time in Phase 8.)

---

I am the student author of this paper (Anmol Jain, supervisor Dr. Sibarama Panigrahi). Teach me the whole thing from scratch, as if I've never seen it, even though it's mine. I need to defend every number and every decision to my professor without hesitation.

**Rules for how you teach, follow these strictly:**
- Short, direct sentences. No buzzwords, no filler, no "it is worth noting."
- Every idea gets a plain-English picture first, then the real numbers/equations, in that order.
- After each topic, ask me one short question. Don't move on until I answer it in my own words.
- Ground everything in the actual paper — name the table, section, or number you mean.
- If I ask about something not yet given to you, say so and ask for it. Never guess or fill a gap with something plausible-sounding.

**One sentence of context before you start:** this paper builds a graph neural network to forecast copper prices, and the main finding is that the graph part doesn't work — the paper spends most of its length figuring out exactly why, catches its own mistakes along the way (including mistakes in the paper's own text, not just the code), and says so plainly instead of hiding them.

Work through the plan below **in order, one numbered topic at a time.** Announce the phase when you enter a new one.

---

## The plan

**Phase 1 — Big picture.** Why copper forecasting matters, what idea the model is testing, the one-line punchline.

**Phase 2 — Dataset.** What data, what time range, how it's split, what went wrong with the data and how it was fixed.

**Phase 3 — Processing.** How raw price turns into what the model actually sees (VMD decomposition).

**Phase 4 — Model structure.** How the model turns that processed data into a prediction, piece by piece.

**Phase 5 — Training.** How the model learns, and a real bug found in how "best" was decided.

**Phase 6 — Output / Results.** What the model actually produced, and what beats what.

**Phase 7 — Ablation study.** What happens when you remove or change one piece at a time — this is where the paper's central finding lives.

**Phase 8 — Code walkthrough.** Same material, now in the real files, one at a time.

**Phase 9 — Mock defense.** You play a skeptical professor. I answer.

---

## Phase 1 — Big picture

1. What copper price forecasting is and why it matters to anyone.
2. The core idea: split the price signal into "moods" (fast wiggles vs. slow wiggles — like bass vs. treble in a song), and let each mood build its own map of which other variables matter to it, instead of one map for everything.
3. The one-line punchline: that per-mood map never actually learns anything — it stays uniform, like it's shrugging at every connection equally — and the paper proves this carefully instead of just reporting a bad number.

Check-in: make me say the punchline back before moving on.

---

## Phase 2 — Dataset

1. **15 variables**, not a small handful: copper (the target), gold, oil, the dollar index, S&P 500, VIX, the 10-year yield, silver, the Chilean peso, two mining-related stocks (FCX, FXI), and four economic indicators from FRED (two daily, two monthly).
2. **Aluminum was removed.** It looked fine at first, but its real trading history only starts in 2014, not 2010 like everything else — so keeping it was silently cutting a quarter of the whole dataset without anyone noticing. Removing it fixed that.
3. **Two bad data points were found and fixed** in the Chilean peso series — single days where the price was off by roughly 100x, clearly a typo in the source data, not a real market move. Both were caught by looking at neighboring days and fixed the same way: replace with a number between its neighbors.
4. **One variable (a producer price index) has a real, disclosed weakness**: the world's inflation behavior after 2022 looks nothing like anything in the training years, so this one variable is feeding the model something it's never seen before — a real, worth-stating limitation, not a bug. This specific weakness is also the reason two of the baseline models (a plain LSTM and a Transformer) fall apart badly at longer horizons — trace that connection.
5. **The split**: train on 2010–2019, tune on 2020–2021, test on 2022–2025 — chronological, never mixed, so the model never sees the future while training on the past.
6. **Leakage**, in one line: making sure nothing about tomorrow's price sneaks into today's inputs.

Check-in: ask me to explain in my own words why the aluminum removal mattered.

---

## Phase 3 — Processing (VMD)

1. VMD, intuitively: take one wiggly price line and split it into several simpler wiggly lines that add back up to the original — each one capturing a different speed of wiggle, from slow trend to fast noise.
2. Why 5 bands, and what "leakage-safe" means here specifically: the split is redone using only data available up to each day, never peeking ahead.

Check-in: one question on why a leaky version of this decomposition would be a real problem, not just sloppy.

---

## Phase 4 — Model structure

Go piece by piece, in the order data actually flows through the model:

1. **Per-band graph**: each of the 5 bands builds its own small map of which of the 15 variables matter to it, keeping only its top few connections (top 7 of the 15, sparsified).
2. **GATv2** reads that map and decides how much attention to pay to each connection — "weighted listening." This is a specific, deliberate upgrade over the plain, older version of this idea (GAT) — GATv2 fixes a proven limitation where the older version's attention ranking couldn't actually depend on the node asking the question. Worth knowing this by name: it's genuinely what trained every current result, not an approximation.
3. This graph-building step is **shared across all 4 forecast horizons** (1, 5, 10, 22 days out) — one shared foundation.
4. From there, **each horizon gets its own separate LSTM path** (5 bands × 4 horizons = 20 small LSTMs total) — so the 1-day forecast and the 22-day forecast aren't forced to share the exact same internal reasoning.
5. Each horizon's 5 band-outputs get combined ("fused") and turned into one number: the price prediction for that horizon.

Check-in: ask me to draw (in words) the path from raw price to one prediction.

---

## Phase 5 — Training

1. **Weight decay**, intuitively: a constant tiny tax pulling every number in the model toward zero, meant to prevent overfitting.
2. **Early stopping**, intuitively: give up once validation performance stops improving.
3. **The real bug**: for a while, a model was allowed to be declared "done training" and "best" using the exact same rule — so a handful of models got judged and reported using weights from very early in training, before they'd learned anything real. Fixed by requiring a minimum number of real training epochs before anything can be called "best."
4. This fix was applied **uniformly** across the whole project this time — same floor for every model, no exceptions.
5. Two more training details worth knowing plainly, since they weren't in the paper's text for a while and had to be added once found missing: the four horizons' loss terms aren't just added up equally — they're weighted so the longer, noisier horizon doesn't dominate training by scale alone — and the learning rate isn't constant, it follows a smooth decreasing schedule over training.

Check-in: ask me why judging a model too early is dangerous specifically because it fails silently.

---

## Phase 6 — Output / Results

1. **7 baselines** to compare against: ARIMA, LSTM, Transformer, two graph-based models (one simplified, one a faithful rebuild of a real published method — MTGNN), XGBoost, and a VMD+LSTM combo.
2. **The anchor finding**: almost nothing beats the dumbest possible forecast — "tomorrow's price equals today's." State this first, before any fancier comparison, so nothing else gets over-read.
3. **The model's own weak spot**: at two of the four forecast horizons, the proposed model's predictions turn out to always point the same direction — meaning its "accuracy" number is really just how often that one direction happened to be right, not real judgment. This is checked for directly, not assumed.
4. **Metrics reported**: RMSE, MAE, two different percentage-error metrics (one of them, SMAPE, exists specifically because the more common one, MAPE, breaks down when the true value is near zero — which happens constantly with daily returns), a scale-free metric (MASE, the standard one used in real forecasting competitions, which asks "how much better or worse than the dumbest naive guess is this, exactly") and directional accuracy.
5. Statistical testing, intuitively: one test asks "are these two models' mistakes meaningfully different, or just noise" (Diebold-Mariano); a second one raises the bar because so many comparisons were run at once (Holm-Bonferroni correction).
6. The real pattern in the results: the proposed model's real wins cluster at the shortest horizon and against the weakest opponents; its losses cluster against the cleaner competitors.

Check-in: ask me to explain, without looking, why "beats a baseline" needs a follow-up question before it means anything.

---

## Phase 7 — Ablation study (the heart of the paper)

1. The idea: change or remove one ingredient at a time, retrain, and see what actually mattered.
2. **8 versions tested**: the full model, a version with no VMD at all, a version with one shared graph instead of one per band, a version with a graph that's frozen and never learned, and two versions with a specific anti-collapse fix applied.
3. **The full model loses to almost everything**, including the version with a completely frozen, untrained graph. That's the clearest possible sign the "learn your own graph" idea isn't earning its keep.
4. **The core finding, built up slowly**: the numbers that decide "which connections matter" shrink toward zero during training and end up nearly identical for every possible connection — the model ends up treating every relationship as equally (un)important. This was measured directly in the trained weights, not guessed at.
5. **A special check ("null control")**: feed the model real data, scrambled data, and pure random noise, and see if it collapses the same way for all three. It does. That rules out "maybe it's something specific about the real data" as the explanation — the collapse happens regardless of what's put in.
6. **The twist**: a targeted fix exists that genuinely stops the collapse — and the resulting graph does show real, non-uniform structure. But models with that fix applied predict *worse*, not better. This is reported honestly as an open question, not swept away.
7. **One extra layer of honesty worth knowing about**: a separate set of robustness checks (does the finding hold across different settings — band count, decomposition method, loss function, random seed) was run earlier in the project, before the dataset was expanded and the architecture was updated. Those results are real and still reported, but the paper now says plainly that this specific section reflects an earlier phase of the project, not the current setup — rather than quietly leaving a reader to assume everything in the paper is from one single, consistent run.

Check-in: ask me to state, in one sentence, why "we fixed the collapse and it got worse" is a more interesting result than a clean win would have been.

---

## Phase 8 — Code walkthrough

Same order as Phases 2–7, but now in real files. Ask me for one file at a time; do not proceed until I paste it.

1. `paper1/src/data_pipeline.py` — data download, the aluminum/CLP fixes, VMD decomposition (Phases 2–3).
2. `paper1/src/models/vmd_mfgnn.py` — the full architecture: graph construction, GATv2, the per-horizon LSTM split (Phase 4).
3. `paper1/src/models/baselines.py` — the 7 baselines (Phase 6).
4. `paper1/src/utils.py` — `EarlyStopper`, and exactly how the early-judging bug happened and got fixed (Phase 5); also `compute_metrics()`, where every reported metric (including SMAPE and MASE) is actually computed.
5. `paper1/src/trainer.py` — the training loop and the ablation-study driver (Phases 5, 7).
6. `paper1/src/diagnostics.py` — how the collapse was actually measured in the weights (Phase 7).
7. `paper1/src/experiments.py` — the null-control experiment specifically (Phase 7).

For each file: one-sentence purpose, then walk the real functions in execution order, tying each part back to the specific phase above. One question per file before moving on.

---

## Phase 9 — Mock defense

Switch to a skeptical professor. Ask one at a time, wait for my answer, then give feedback and the ideal answer.

- "Your model doesn't beat a coin-flip baseline. Why should anyone care?"
- "How do you know the collapse is real and not just a bug in your code?"
- "You fixed the collapse and performance got worse. Doesn't that mean your diagnosis was wrong?"
- "Why are two of your baselines your own rebuilds instead of the original authors' code?"
- "You only tuned your own model's settings, not the baselines'. Isn't that unfair?"
- "What's actually new here, versus just combining existing ideas?"
- "You removed a variable (aluminum) mid-project. Isn't that moving the goalposts?"
- "Your paper's own text had factual errors in it — a wrong reference number repeated eight times, an undisclosed change in which attention mechanism you actually used. How do I know there aren't more, and why should I trust anything else in the paper?"
- "What would you do differently starting over?"

Don't let me move on from a question until I can answer it confidently, in my own words, grounded in the real paper.

---

Begin with Phase 1, topic 1. Wait for me before continuing.
