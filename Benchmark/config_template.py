from dataclasses import dataclass, field
from pathlib import Path


FEATURE_FOLDER_TO_MODE = {
    "mfbe": "mfbe_only",
    "pitch": "pitch_only",
    "mfcc": "mfcc_only",
    "mfbe_pitch": "mfbe_pitch",
    "mfcc_pitch": "mfcc_pitch",
    "fbank": "fbank_only",
    "fbank_pitch": "fbank_pitch",
}

HC_FILENAME_MAP = {
    "mfbe_only": "Only_MFBE.pt",
    "pitch_only": "Only_Pitch.pt",
    "mfcc_only": "Only_MFCC.pt",
    "mfbe_pitch": "MFBE_Pitch.pt",
    "mfcc_pitch": "MFCC_Pitch.pt",
    "fbank_only": "FBank.pt",
    "fbank_pitch": "FBank_Pitch.pt",
}


@dataclass
class BenchmarkTemplateConfig:
    # Fill these 2 fields after your own research.
    target_duration: str = "train_raw"
    target_feature_folder: str = "fbank"

    # Default: 4 benchmark models only (ESPnet excluded by request).
    benchmark_archs: list[str] = field(
        default_factory=lambda: ["campp", "eres2netv2", "mfa_conformer", "redimnet"]
    )

    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    features_root: Path = Path("D:/extracted_features")
    test_base_dir: Path = Path("D:/test_o")
    list_gt_dirname: str = "list_gt"

    # Output kept under Benchmark folder
    benchmark_output_rootname: str = "outputs"
    experiments_dirname: str = "experiments_benchmark"
    reports_dirname: str = "reports"

    # Train defaults (can be edited in notebook)
    num_epochs: int = 50
    batch_size: int = 128
    batch_size_overrides: dict[str, int] = field(default_factory=lambda: {"eres2netv2": 16})
    learning_rate: float = 1e-4
    learning_rate_overrides: dict[str, float] = field(default_factory=lambda: {"eres2netv2": 3e-5})
    weight_decay: float = 1e-3
    early_stop_patience: int = 4
    lr_scheduler: str = "plateau"
    mode: int = 2
    mixed_precision: bool = True
    mixed_precision_overrides: dict[str, bool] = field(default_factory=lambda: {"eres2netv2": False})
    amp_scaler_init_scale: float = 1024.0
    amp_scaler_growth_interval: int = 2000
    amp_grad_warn_every: int = 50
    amp_nonfinite_grad_fallback_ratio: float = 0.30
    amp_nonfinite_grad_fallback_min_batches: int = 300
    use_augment: bool = True
    max_frames: int = 350
    max_frames_overrides: dict[str, int] = field(default_factory=lambda: {"eres2netv2": 200})
    val_batch_size: int = 128
    val_batch_size_overrides: dict[str, int] = field(default_factory=lambda: {"eres2netv2": 16})

    # Runtime/data-loader knobs for large shard training on Windows.
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking_transfer: bool = True
    validate_every: int = 1
    val_max_pos_pairs: int = 5000
    max_train_steps_per_epoch: int = 0
    train_subset_fraction: float = 1.0
    max_train_samples: int = 0
    cudnn_benchmark: bool = True

    # IMPORTANT: keep RAM stable by default in benchmark runs.
    preload_all_ram: bool = False
    preload_ptm_ram: bool = False
    preload_hc_ram: bool = False

    # Runtime
    device: str = "cuda"
    seed: int = 42

    # Optional override; when None, each architecture uses its native embedding dim.
    embedding_dim_override: int | None = None

    # Inference
    p_target: float = 0.05
    test_set_configs: list[dict] = field(
        default_factory=lambda: [
            {
                "name": "test_celeb_e",
                "test_dir": "D:/test_celeb_e",
                "trial_file": "D:/test_celeb_e/list_gt.txt",
                "trial_in_test_dir": True,
            },
            {
                "name": "test_celeb_h",
                "test_dir": "D:/test_celeb_h",
                "trial_file": "D:/test_celeb_h/list_gt.txt",
                "trial_in_test_dir": True,
            },
            {"name": "full_length", "trial_file": "test_list_gt.csv", "duration": "full_length"},
            {"name": "test_vi", "trial_file": "test_list_gt_vi.csv", "duration": "test_vi"},
            {"name": "test_3s", "trial_file": "test_list_gt_3s.csv", "duration": "test_3s"},
            {"name": "test_5s", "trial_file": "test_list_gt_5s.csv", "duration": "test_5s"},
            {"name": "test_7s", "trial_file": "test_list_gt_7s.csv", "duration": "test_7s"},
        ]
    )

    def validate_placeholders(self) -> None:
        if self.target_duration.startswith("__FILL"):
            raise ValueError("Please set target_duration in BenchmarkTemplateConfig.")
        if self.target_feature_folder.startswith("__FILL"):
            raise ValueError("Please set target_feature_folder in BenchmarkTemplateConfig.")
        if self.target_feature_folder not in FEATURE_FOLDER_TO_MODE:
            valid = ", ".join(sorted(FEATURE_FOLDER_TO_MODE.keys()))
            raise ValueError(f"Unknown target_feature_folder={self.target_feature_folder}. Valid: {valid}")

    @property
    def target_feature_mode(self) -> str:
        return FEATURE_FOLDER_TO_MODE[self.target_feature_folder]

    @property
    def benchmark_output_root(self) -> Path:
        return self.repo_root / "Benchmark" / self.benchmark_output_rootname

    @property
    def experiments_root(self) -> Path:
        return self.benchmark_output_root / self.experiments_dirname

    @property
    def reports_root(self) -> Path:
        return self.benchmark_output_root / self.reports_dirname

    @property
    def list_gt_dir(self) -> Path:
        return self.test_base_dir / self.list_gt_dirname

    @property
    def train_feature_path(self) -> Path:
        return self.features_root / self.target_duration / f"{self.target_feature_folder}_shards"
