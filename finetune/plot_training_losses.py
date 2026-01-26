import json
import os

import matplotlib.pyplot as plt

from utils.config_utils import get_config_class


def main():
    Config = get_config_class()
    config = Config()

    save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
    history_path = os.path.join(save_dir, "loss_history.json")
    summary_path = os.path.join(save_dir, "summary.json")

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    elif os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        history = summary.get("final_result", {}).get("loss_history", {})
    else:
        raise FileNotFoundError("No loss history found. Run training first.")

    train = history.get("train", [])
    val = history.get("val", [])
    test = history.get("test", [])

    if not train and not val and not test:
        raise ValueError("Loss history is empty.")

    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    out_path = os.path.join(save_dir, "plots", "loss_curves.png")

    plt.figure(figsize=(10, 6))
    if train:
        plt.plot(train, label="train")
    if val:
        plt.plot(val, label="val")
    if test:
        plt.plot(test, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation/Test Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve to: {out_path}")


if __name__ == "__main__":
    main()
