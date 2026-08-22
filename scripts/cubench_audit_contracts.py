"""S2 / Week-A1 spike: COMEX contract-month coverage audit.

Probes all 192 tickers of the form HG{month}{YY}.CMX (month codes F,G,H,J,K,M,N,Q,U,V,X,Z
x years 10..25) via the Yahoo Finance chart HTTP endpoint directly (requests works fine
against query1.finance.yahoo.com on this machine -- only fred.stlouisfed.org has the
TLS/curl-workaround issue, per docs/cubench_implementation_plan.md Section 1.1).

For each ticker records: HTTP status, first obs date, last obs date, obs count.
Writes results/cubench/data_audit/comex_contract_month_coverage.csv.

Spec: docs/cubench_implementation_plan.md Section 1.5 / Section 6 (Week A1, spike S2).
"""
import time
import json
import datetime as dt
from pathlib import Path

import requests
import pandas as pd

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
YEARS = list(range(10, 26))  # 10..25 inclusive -> 2010..2025

OUT_CSV = Path("results/cubench/data_audit/comex_contract_month_coverage.csv")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Full window, wide enough to catch any contract's full life.
PERIOD1 = int(dt.datetime(2005, 1, 1).timestamp())
PERIOD2 = int(dt.datetime(2026, 1, 1).timestamp())


def probe(ticker: str, retries: int = 3) -> dict:
    params = {"period1": PERIOD1, "period2": PERIOD2, "interval": "1d"}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(
                CHART_URL.format(ticker=ticker), params=params, headers=HEADERS, timeout=15
            )
            status = r.status_code
            if status != 200:
                return {
                    "ticker": ticker, "http_status": status,
                    "first_obs_date": None, "last_obs_date": None, "obs_count": 0,
                    "note": f"non-200 status",
                }
            payload = r.json()
            result = payload.get("chart", {}).get("result")
            error = payload.get("chart", {}).get("error")
            if error is not None:
                return {
                    "ticker": ticker, "http_status": status,
                    "first_obs_date": None, "last_obs_date": None, "obs_count": 0,
                    "note": f"api_error: {error.get('description', error)}",
                }
            if not result:
                return {
                    "ticker": ticker, "http_status": status,
                    "first_obs_date": None, "last_obs_date": None, "obs_count": 0,
                    "note": "empty_result",
                }
            r0 = result[0]
            timestamps = r0.get("timestamp") or []
            closes = (
                r0.get("indicators", {}).get("quote", [{}])[0].get("close") or []
            )
            valid_ts = [
                t for t, c in zip(timestamps, closes) if c is not None
            ]
            if not valid_ts:
                return {
                    "ticker": ticker, "http_status": status,
                    "first_obs_date": None, "last_obs_date": None, "obs_count": 0,
                    "note": "no_valid_close_obs",
                }
            first_date = dt.datetime.utcfromtimestamp(min(valid_ts)).strftime("%Y-%m-%d")
            last_date = dt.datetime.utcfromtimestamp(max(valid_ts)).strftime("%Y-%m-%d")
            return {
                "ticker": ticker, "http_status": status,
                "first_obs_date": first_date, "last_obs_date": last_date,
                "obs_count": len(valid_ts), "note": "ok",
            }
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    return {
        "ticker": ticker, "http_status": None,
        "first_obs_date": None, "last_obs_date": None, "obs_count": 0,
        "note": f"exception: {type(last_exc).__name__}: {last_exc}",
    }


def main():
    tickers = [f"HG{m}{y:02d}.CMX" for m in MONTH_CODES for y in YEARS]
    assert len(tickers) == 192, f"expected 192 tickers, got {len(tickers)}"

    rows = []
    for i, t in enumerate(tickers):
        row = probe(t)
        rows.append(row)
        print(f"[{i+1:3d}/192] {t:16s} status={row['http_status']} "
              f"first={row['first_obs_date']} last={row['last_obs_date']} "
              f"n={row['obs_count']} note={row['note']}")
        time.sleep(0.15)  # be polite to the endpoint

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    # Summary
    live = df[df["obs_count"] > 0]
    print(f"\nTickers with >0 valid obs: {len(live)} / {len(df)}")
    print(f"Tickers with HTTP 200: {(df['http_status']==200).sum()}")
    print(f"Tickers with non-200 or error: {len(df) - (df['http_status']==200).sum()}")
    if len(live):
        print(live[["ticker", "first_obs_date", "last_obs_date", "obs_count"]].to_string(index=False))

    # Promotion-rule check: >=2 simultaneously-live contract months on >=60% of trading
    # days across full 2010-2025 window. We approximate "simultaneously live" via date
    # range overlap count per calendar day using first/last obs date (coarse, sufficient
    # to confirm/refute the plan's prediction given the expected near-total absence of data).
    if len(live) >= 2:
        calendar = pd.bdate_range("2010-01-04", "2025-12-31")
        live2 = live.dropna(subset=["first_obs_date", "last_obs_date"]).copy()
        live2["first_obs_date"] = pd.to_datetime(live2["first_obs_date"])
        live2["last_obs_date"] = pd.to_datetime(live2["last_obs_date"])
        counts = pd.Series(0, index=calendar)
        for _, r in live2.iterrows():
            mask = (calendar >= r["first_obs_date"]) & (calendar <= r["last_obs_date"])
            counts[mask] += 1
        frac_ge2 = (counts >= 2).mean()
        print(f"\nFraction of 2010-2025 trading days with >=2 simultaneously-live "
              f"contract months (coverage-range approximation): {frac_ge2:.4f}")
        promote = frac_ge2 >= 0.60
    else:
        frac_ge2 = 0.0
        promote = False
        print("\nFewer than 2 tickers returned any data -- promotion rule trivially fails.")

    summary = {
        "n_tickers_probed": len(df),
        "n_tickers_with_data": int(len(live)),
        "n_http_200": int((df["http_status"] == 200).sum()),
        "frac_days_with_ge2_live_contracts": float(frac_ge2),
        "promotion_rule_threshold": 0.60,
        "block_b_promoted_to_tier1": bool(promote),
    }
    summary_path = OUT_CSV.parent / "comex_contract_month_coverage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
