import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark_models import ARCH_DEFAULT_EMBEDDING_DIM, build_benchmark_get_model


def _setup_train_import_paths(repo_root: Path) -> None:
    train_dir = repo_root / "train"
    for path in (train_dir, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _import_train_module(repo_root: Path):
    _setup_train_import_paths(repo_root)
    return importlib.import_module("train")


def _build_args(cfg, arch_name: str) -> SimpleNamespace:
    native_dim = int(ARCH_DEFAULT_EMBEDDING_DIM[arch_name])
    embed_dim = int(cfg.embedding_dim_override) if cfg.embedding_dim_override is not None else native_dim

    batch_size = int(getattr(cfg, "batch_size_overrides", {}).get(arch_name, cfg.batch_size))
    val_batch_size = int(getattr(cfg, "val_batch_size_overrides", {}).get(arch_name, cfg.val_batch_size))
    max_frames = int(getattr(cfg, "max_frames_overrides", {}).get(arch_name, cfg.max_frames))
    learning_rate = float(getattr(cfg, "learning_rate_overrides", {}).get(arch_name, cfg.learning_rate))
    mixed_precision = bool(getattr(cfg, "mixed_precision_overrides", {}).get(arch_name, cfg.mixed_precision))

    exp_name = f"Bench_{arch_name}_{cfg.target_duration}_{cfg.target_feature_mode}"

    return SimpleNamespace(
        output_dir=str(cfg.benchmark_output_root),
        experiments_dirname=str(cfg.experiments_dirname),
        mode=int(cfg.mode),
        exp_name=exp_name,
        duration=str(cfg.target_duration),
        pretrained_model="benchmark",
        feature_mode=str(cfg.target_feature_mode),
        feature_path=str(cfg.train_feature_path),
        embedding_path="",
        fusion_method="none",
        use_gating=False,
        use_augment=bool(cfg.use_augment),
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        learning_rate=learning_rate,
        weight_decay=float(cfg.weight_decay),
        lr_scheduler=str(cfg.lr_scheduler),
        num_epochs=int(cfg.num_epochs),
        early_stop_patience=int(cfg.early_stop_patience),
        embedding_dim=int(embed_dim),
        mixed_precision=mixed_precision,
        amp_scaler_init_scale=float(cfg.amp_scaler_init_scale),
        amp_scaler_growth_interval=int(cfg.amp_scaler_growth_interval),
        amp_grad_warn_every=int(cfg.amp_grad_warn_every),
        amp_nonfinite_grad_fallback_ratio=float(cfg.amp_nonfinite_grad_fallback_ratio),
        amp_nonfinite_grad_fallback_min_batches=int(cfg.amp_nonfinite_grad_fallback_min_batches),
        max_frames=max_frames,
        optimizer="adam",
        aam_margin=0.25,
        aam_scale=30.0,
        seed=int(cfg.seed),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers),
        prefetch_factor=int(cfg.prefetch_factor),
        non_blocking_transfer=bool(cfg.non_blocking_transfer),
        validate_every=int(cfg.validate_every),
        val_max_pos_pairs=int(cfg.val_max_pos_pairs),
        max_train_steps_per_epoch=int(cfg.max_train_steps_per_epoch),
        train_subset_fraction=float(cfg.train_subset_fraction),
        max_train_samples=int(cfg.max_train_samples),
        cudnn_benchmark=bool(cfg.cudnn_benchmark),
        preload_all_ram=bool(cfg.preload_all_ram),
        preload_ptm_ram=bool(cfg.preload_ptm_ram),
        preload_hc_ram=bool(cfg.preload_hc_ram),
        augment_prob=0.2,
        feature_noise_std=0.002,
        embedding_noise_std=0.001,
        benchmark_arch=str(arch_name),
        use_mode3_warmstart=False,
        use_mode3_init_mode1=False,
        use_mode3_init_mode2=False,
        init_mode1_ckpt=None,
        init_mode2_ckpt=None,
    )


def _best_model_exists(cfg, exp_name: str) -> bool:
    best_model = cfg.experiments_root / exp_name / "best_model.pth"
    return best_model.exists()


def run_training_template(cfg, skip_if_done: bool = True):
    cfg.validate_placeholders()

    cfg.benchmark_output_root.mkdir(parents=True, exist_ok=True)
    cfg.experiments_root.mkdir(parents=True, exist_ok=True)

    train_module = _import_train_module(cfg.repo_root)
    original_get_model = train_module.get_model

    run_log = []

    for arch_name in cfg.benchmark_archs:
        args = _build_args(cfg, arch_name=arch_name)

        if skip_if_done and _best_model_exists(cfg, args.exp_name):
            print(f"[SKIP] existing checkpoint: {args.exp_name}")
            run_log.append({"arch": arch_name, "exp_name": args.exp_name, "status": "skipped_existing"})
            continue

        print("=" * 90)
        print(f"[TRAIN] arch={arch_name} | exp={args.exp_name}")
        print(f"feature_path={args.feature_path}")
        print("=" * 90)

        train_module.get_model = build_benchmark_get_model(
            arch_name=arch_name,
            feature_mode=cfg.target_feature_mode,
            embedding_dim_override=cfg.embedding_dim_override,
        )

        try:
            train_module.train(args)
            run_log.append({"arch": arch_name, "exp_name": args.exp_name, "status": "done"})
        except Exception as ex:
            print(f"[ERROR] arch={arch_name} | {type(ex).__name__}: {ex}")
            run_log.append(
                {
                    "arch": arch_name,
                    "exp_name": args.exp_name,
                    "status": "error",
                    "error_type": type(ex).__name__,
                    "error": str(ex),
                }
            )
        finally:
            train_module.get_model = original_get_model

    log_path = cfg.benchmark_output_root / "benchmark_train_run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print(f"Saved benchmark train log: {log_path}")
    return run_log
