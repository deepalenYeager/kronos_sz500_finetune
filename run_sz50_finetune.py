import argparse
import os
import subprocess
import sys

from prepare_sz50_data import prepare_sz50_dataset


def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="SZ50 finetune pipeline: data prep + training.")
    parser.add_argument("--data-dir", default="sz50_2y/sz50_2y")
    parser.add_argument("--output-dir", default="finetune/data/sz50_processed")
    parser.add_argument("--train-end", default="2025-03-31")
    parser.add_argument("--val-end", default="2025-08-31")
    parser.add_argument("--min-window", type=int, default=91)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-predictor", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-num-symbols", type=int, default=5)
    parser.add_argument("--config-module", default="finetune.config_sz50")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = prepare_sz50_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        min_window=args.min_window,
    )
    print("Dataset prepared:", summary)

    env = os.environ.copy()
    env["KRONOS_CONFIG_MODULE"] = args.config_module
    env["KRONOS_DATASET_PATH"] = args.output_dir
    env.setdefault("KRONOS_PRETRAINED_TOKENIZER", "./pretrained/Kronos-Tokenizer-base")
    env.setdefault("KRONOS_PRETRAINED_PREDICTOR", "./pretrained/Kronos-small")
    root_dir = os.path.abspath(os.path.dirname(__file__))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root_dir + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    if args.dry_run:
        print("Dry run: skipping training.")
        return

    if not args.skip_tokenizer:
        run_cmd(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.gpus}",
                "finetune/train_tokenizer.py",
            ],
            env=env,
        )

    if not args.skip_predictor:
        run_cmd(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.gpus}",
                "finetune/train_predictor.py",
            ],
            env=env,
        )

    if not args.no_plots:
        run_cmd([sys.executable, "finetune/plot_training_losses.py"], env=env)
        run_cmd(
            [
                sys.executable,
                "finetune/plot_inference_vs_truth.py",
                "--num-symbols",
                str(args.plot_num_symbols),
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
