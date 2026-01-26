import os


class Config:
    """
    SZ50 fine-tuning configuration (daily bars, multi-symbol).
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.instrument = "sz50"
        # Dataset date range (for reference)
        self.dataset_begin_time = "2024-01-26"
        self.dataset_end_time = "2026-01-23"

        # Sliding window parameters
        self.lookback_window = 80
        self.predict_window = 10
        self.max_context = 512

        # Features used in training data
        self.feature_list = ["open", "high", "low", "close", "vol", "amt"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # These are used by the data preparation script.
        self.train_time_range = ["2024-01-26", "2025-03-31"]
        self.val_time_range = ["2025-04-01", "2025-08-31"]
        self.test_time_range = ["2025-09-01", "2026-01-23"]

        # Processed pickle datasets (train/val/test)
        self.dataset_path = os.getenv(
            "KRONOS_DATASET_PATH",
            "./finetune/data/sz50_processed"
        )

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0
        self.epochs = 8
        self.log_interval = 100
        self.batch_size = 64  # per GPU

        # Number of samples per epoch (cap on random sampling)
        self.n_train_iter = 10000
        self.n_val_iter = 2000
        self.n_test_iter = 2000

        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 2e-5

        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        self.seed = 100

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-SZ50",
            "workspace": "your_comet_workspace",
        }
        self.comet_tag = "sz50"
        self.comet_name = "sz50_finetune"

        self.save_path = os.getenv(
            "KRONOS_SAVE_PATH",
            "./outputs/models_sz50"
        )
        self.tokenizer_save_folder_name = "sz50_tokenizer"
        self.predictor_save_folder_name = "sz50_predictor"
        self.backtest_save_folder_name = "sz50_backtest"
        self.backtest_result_path = "./outputs/backtest_results_sz50"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # Can be local paths or HF model IDs
        self.pretrained_tokenizer_path = os.getenv(
            "KRONOS_PRETRAINED_TOKENIZER",
            "./pretrained/Kronos-Tokenizer-base"
        )
        self.pretrained_predictor_path = os.getenv(
            "KRONOS_PRETRAINED_PREDICTOR",
            "./pretrained/Kronos-small"
        )

        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # =================================================================
        # Backtesting Parameters (kept for compatibility)
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            "sz50": "SH000016",
            "csi300": "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        raise ValueError(f"Benchmark not defined for instrument: {instrument}")
