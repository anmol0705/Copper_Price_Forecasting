import yfinance as yf
import pandas as pd

for ticker in ["CLP=X", "QC=F"]:
    print(f"=== {ticker} ===")
    try:
        df = yf.download(ticker, start="2010-01-01", end="2025-12-31", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        n = len(df)
        print("rows:", n)
        if n > 0:
            print("first:", df.index.min(), "last:", df.index.max())
            # gap analysis vs business days in range
            bdays = pd.bdate_range(df.index.min(), df.index.max())
            gap_frac = 1 - n / len(bdays)
            print(f"business days in span: {len(bdays)}, obs: {n}, gap_frac: {gap_frac:.4f}")
            print("meets rule (>=3800 obs 2010-2025, <2% gaps):",
                  n >= 3800 and gap_frac < 0.02)
        df.to_csv(f"scratchpad/s4_{ticker.replace('=','_')}.csv")
    except Exception as e:
        print("FAILED:", type(e).__name__, e)
    print()
