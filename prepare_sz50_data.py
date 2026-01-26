import argparse
import glob
import os
from datetime import datetime

import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
OUTPUT_COLUMNS = ["open", "high", "low", "close", "vol", "amt"]


def _find_time_column(df):
    for col in ["timestamps", "timestamp", "date", "datetime"]:
        if col in df.columns:
            return col
    return None


def _load_symbol_csv(path):
    df = pd.read_csv(path)
    time_col = _find_time_column(df)
    if time_col is None:
        raise ValueError(f"{os.path.basename(path)} missing time column (date/timestamp/timestamps).")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    df = df.rename(columns={"volume": "vol", "amount": "amt"})
    time_index = pd.to_datetime(df[time_col])
    df = df[["open", "high", "low", "close", "vol", "amt"]]

    df.index = time_index
    df.index.name = "datetime"

    if df.isnull().any().any():
        df = df.dropna()

    return df


def _split_by_date(df, train_end, val_end):
    train_mask = df.index <= train_end
    val_mask = (df.index > train_end) & (df.index <= val_end)
    test_mask = df.index > val_end
    return df.loc[train_mask], df.loc[val_mask], df.loc[test_mask]


def prepare_sz50_dataset(
    data_dir,
    output_dir,
    train_end="2025-03-31",
    val_end="2025-08-31",
    min_window=91,
):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        p for p in glob.glob(os.path.join(data_dir, "*.csv"))
        if not p.endswith("constituents.csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in {data_dir}")

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train_data, val_data, test_data = {}, {}, {}
    skipped = []

    for path in csv_files:
        symbol = os.path.splitext(os.path.basename(path))[0]
        df = _load_symbol_csv(path)
        tr, va, te = _split_by_date(df, train_end, val_end)

        if len(tr) >= min_window:
            train_data[symbol] = tr
        if len(va) >= min_window:
            val_data[symbol] = va
        if len(te) >= min_window:
            test_data[symbol] = te

        if len(tr) < min_window and len(va) < min_window and len(te) < min_window:
            skipped.append(symbol)

    train_path = os.path.join(output_dir, "train_data.pkl")
    val_path = os.path.join(output_dir, "val_data.pkl")
    test_path = os.path.join(output_dir, "test_data.pkl")

    pd.to_pickle(train_data, train_path)
    pd.to_pickle(val_data, val_path)
    pd.to_pickle(test_data, test_path)

    summary = {
        "symbols_total": len(csv_files),
        "symbols_train": len(train_data),
        "symbols_val": len(val_data),
        "symbols_test": len(test_data),
        "skipped": skipped,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare SZ50 multi-symbol dataset for Kronos finetuning.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sz50_2y/sz50_2y",
        help="Directory containing SZ50 csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetune/data/sz50_processed",
        help="Output directory for train/val/test pickle files.",
    )
    parser.add_argument("--train-end", type=str, default="2025-03-31")
    parser.add_argument("--val-end", type=str, default="2025-08-31")
    parser.add_argument("--min-window", type=int, default=91)
    args = parser.parse_args()

    summary = prepare_sz50_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        min_window=args.min_window,
    )

    print("SZ50 dataset prepared:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
