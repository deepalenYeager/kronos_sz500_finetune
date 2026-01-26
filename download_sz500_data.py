import argparse
import os
import time
from datetime import datetime

import pandas as pd

import akshare as ak


def _get_code_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "代码" in col:
            return col
    raise ValueError("Cannot find code column in index constituents.")


def fetch_sz500_symbols() -> list[str]:
    cons_df = ak.index_stock_cons(symbol="000905")
    code_col = _get_code_column(cons_df)
    symbols = cons_df[code_col].astype(str).tolist()
    # Convert to akshare format with exchange prefix
    return [ak.stock_a_code_to_symbol(code) for code in symbols]


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "amount" not in df.columns:
        if "成交额" in df.columns:
            df = df.rename(columns={"成交额": "amount"})
        elif "close" in df.columns and "volume" in df.columns:
            df["amount"] = df["close"] * df["volume"]
        else:
            df["amount"] = 0.0

    if "outstanding_share" not in df.columns and "流通股本" in df.columns:
        df = df.rename(columns={"流通股本": "outstanding_share"})
    if "turnover" not in df.columns and "换手率" in df.columns:
        df = df.rename(columns={"换手率": "turnover"})

    cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "outstanding_share",
        "turnover",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def download_one(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    df = ak.stock_zh_a_daily(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    return normalize_daily_df(df)


def main():
    parser = argparse.ArgumentParser(description="Download SZ500 daily data via AKShare.")
    parser.add_argument("--out-dir", default="sz500_2y/sz500_2y")
    parser.add_argument("--start-date", default="20240101", help="YYYYMMDD")
    parser.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"), help="YYYYMMDD")
    parser.add_argument("--adjust", default="", help="''|qfq|hfq")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    symbols = fetch_sz500_symbols()
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    failures = []
    for i, sym in enumerate(symbols, start=1):
        code = sym[2:] if sym[:2] in {"sh", "sz"} else sym
        out_path = os.path.join(args.out_dir, f"{code}.csv")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{len(symbols)}] Skip {sym}: exists")
            continue
        try:
            df = download_one(sym, args.start_date, args.end_date, args.adjust)
            if df.empty:
                raise ValueError("empty dataframe")
            df.to_csv(out_path, index=False)
            print(f"[{i}/{len(symbols)}] Saved {sym} -> {out_path} ({len(df)} rows)")
        except Exception as exc:
            failures.append((sym, str(exc)))
            print(f"[{i}/{len(symbols)}] Failed {sym}: {exc}")
        time.sleep(args.sleep)

    if failures:
        print("Failures:")
        for sym, err in failures:
            print(f"- {sym}: {err}")


if __name__ == "__main__":
    main()
