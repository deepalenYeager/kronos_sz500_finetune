import argparse
import os
import pickle
import sys


REQUIRED_MODEL_FILES = ["config.json"]
MODEL_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


def _check_model_dir(path):
    if not os.path.isdir(path):
        return False, f"missing dir: {path}"
    missing = [f for f in REQUIRED_MODEL_FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        return False, f"missing files in {path}: {missing}"
    if not any(os.path.exists(os.path.join(path, f)) for f in MODEL_WEIGHT_FILES):
        return False, f"missing weights in {path} (need one of {MODEL_WEIGHT_FILES})"
    return True, "ok"


def _check_pickle(path, min_symbols=1):
    if not os.path.exists(path):
        return False, f"missing: {path}"
    try:
        data = pickle.load(open(path, "rb"))
    except Exception as exc:
        return False, f"failed to load {path}: {exc}"
    if not isinstance(data, dict):
        return False, f"invalid format in {path}: expected dict, got {type(data)}"
    if len(data) < min_symbols:
        return False, f"too few symbols in {path}: {len(data)}"
    return True, f"symbols={len(data)}"


def main():
    parser = argparse.ArgumentParser(description="Verify SZ50 finetune setup.")
    parser.add_argument("--model-root", default="pretrained")
    parser.add_argument("--dataset-dir", default="finetune/data/sz50_processed")
    parser.add_argument("--min-symbols", type=int, default=10)
    args = parser.parse_args()

    ok = True

    tokenizer_dir = os.path.join(args.model_root, "Kronos-Tokenizer-base")
    predictor_dir = os.path.join(args.model_root, "Kronos-small")

    status, msg = _check_model_dir(tokenizer_dir)
    print(f"Tokenizer: {msg}")
    ok = ok and status

    status, msg = _check_model_dir(predictor_dir)
    print(f"Predictor: {msg}")
    ok = ok and status

    train_pkl = os.path.join(args.dataset_dir, "train_data.pkl")
    val_pkl = os.path.join(args.dataset_dir, "val_data.pkl")
    test_pkl = os.path.join(args.dataset_dir, "test_data.pkl")

    status, msg = _check_pickle(train_pkl, min_symbols=args.min_symbols)
    print(f"Train data: {msg}")
    ok = ok and status

    status, msg = _check_pickle(val_pkl, min_symbols=args.min_symbols)
    print(f"Val data: {msg}")
    ok = ok and status

    status, msg = _check_pickle(test_pkl, min_symbols=args.min_symbols)
    print(f"Test data: {msg}")
    ok = ok and status

    if not ok:
        print("Verification failed.")
        sys.exit(1)

    print("Verification passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
