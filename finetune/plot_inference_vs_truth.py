import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.config_utils import get_config_class
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference, calc_time_stamps


def main():
    parser = argparse.ArgumentParser(description="Plot SZ50 inference vs ground truth.")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to plot (single)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol list")
    parser.add_argument("--num-symbols", type=int, default=1, help="Number of symbols to plot if --symbols not set")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in test series")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Config = get_config_class()
    config = Config()

    test_path = os.path.join(config.dataset_path, "test_data.pkl")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test data: {test_path}")

    test_data = pd.read_pickle(test_path)
    if not test_data:
        raise ValueError("Test dataset is empty.")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = sorted(test_data.keys())[: max(1, args.num_symbols)]

    missing = [s for s in symbols if s not in test_data]
    if missing:
        raise ValueError(f"Symbols not found in test set: {missing}")

    out_dir = os.path.join(config.save_path, config.predictor_save_folder_name, "plots")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer_path = config.finetuned_tokenizer_path if os.path.exists(config.finetuned_tokenizer_path) else config.pretrained_tokenizer_path
    predictor_path = config.finetuned_predictor_path if os.path.exists(config.finetuned_predictor_path) else config.pretrained_predictor_path

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(args.device).eval()
    model = Kronos.from_pretrained(predictor_path).to(args.device).eval()

    lookback = config.lookback_window
    pred_len = config.predict_window

    for symbol in symbols:
        df = test_data[symbol].sort_index()
        start_idx = args.start_index
        end_idx = start_idx + lookback + pred_len
        if end_idx > len(df):
            print(f"Skipping {symbol}: not enough data for start_index={start_idx}.")
            continue

        context_df = df.iloc[start_idx:start_idx + lookback]
        future_df = df.iloc[start_idx + lookback:end_idx]

        x = context_df[["open", "high", "low", "close", "vol", "amt"]].values.astype(np.float32)
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -config.clip, config.clip)

        x_timestamp = context_df.index.to_series()
        y_timestamp = future_df.index.to_series()

        x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

        x_tensor = torch.from_numpy(x_norm[None, :, :]).to(args.device)
        x_stamp_tensor = torch.from_numpy(x_stamp[None, :, :]).to(args.device)
        y_stamp_tensor = torch.from_numpy(y_stamp[None, :, :]).to(args.device)

        preds = auto_regressive_inference(
            tokenizer=tokenizer,
            model=model,
            x=x_tensor,
            x_stamp=x_stamp_tensor,
            y_stamp=y_stamp_tensor,
            max_context=config.max_context,
            pred_len=pred_len,
            clip=config.clip,
            T=config.inference_T,
            top_k=config.inference_top_k,
            top_p=config.inference_top_p,
            sample_count=args.sample_count,
        )

        preds = preds[0]
        preds = preds[-pred_len:, :]
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(
            preds,
            columns=["open", "high", "low", "close", "vol", "amt"],
            index=future_df.index,
        )

        out_csv = os.path.join(out_dir, f"{symbol}_predictions.csv")
        pred_df.to_csv(out_csv, index=True)

        plt.figure(figsize=(10, 6))
        plt.plot(
            future_df.index,
            future_df["close"],
            label="truth_close",
            color="#d62728",
            linewidth=1.8,
        )
        plt.plot(
            pred_df.index,
            pred_df["close"],
            label="pred_close",
            color="#1f77b4",
            linestyle="--",
            linewidth=1.6,
        )

        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.title(f"{symbol} Test Prediction vs Ground Truth (close)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{symbol}_pred_vs_truth_close.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved prediction plot to: {out_png}")
        print(f"Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
