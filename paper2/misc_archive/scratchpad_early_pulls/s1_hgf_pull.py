import sys
import yfinance as yf
import pandas as pd

print("yfinance version:", yf.__version__)
try:
    df = yf.download("HG=F", start="2010-01-04", end="2025-12-31", auto_adjust=False, progress=False)
    print("Download shape:", df.shape)
    print(df.columns.tolist())
    print(df.head())
    print(df.tail())
    n = len(df)
    ok_rows = n >= 4000
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    nonnull = df[["Open","High","Low","Close"]].notnull().all(axis=None)
    print("rows:", n, "assert>=4000:", ok_rows, "all OHLC non-null:", bool(nonnull))
    df.to_csv("scratchpad/s1_hgf_raw.csv")
except Exception as e:
    print("FAILED:", type(e).__name__, e)
    sys.exit(1)
