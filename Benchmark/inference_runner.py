import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark_models import build_benchmark_get_model
from config_template import HC_FILENAME_MAP


def _setup_metric_import_paths(repo_root: Path) -> None:
    import sys

    train_dir = repo_root / "train"
    for path in (train_dir, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _candidate_keys(path_text: str):
    p = str(path_text).replace("\\", "/").strip()
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    rel = p.lstrip("./")

    # Include slash<->underscore variants to align trial paths like
    # "spk/utt.wav" with feature keys like "spk_utt.wav".
    rel_us = rel.replace("/", "_")
    base_us = base.replace("/", "_")
    stem_us = stem.replace("/", "_")
    p_us = p.replace("/", "_")

    candidates = [
        p,
        rel,
        base,
        stem,
        p_us,
        rel_us,
        base_us,
        stem_us,
    ]
    out = []
    seen = set()
    for key in candidates:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _register_lookup(lookup, path_text, value):
    for key in _candidate_keys(path_text):
        if key not in lookup:
            lookup[key] = value


def _get_feature_from_lookup(lookup, path_text):
    for key in _candidate_keys(path_text):
        if key in lookup:
            return lookup[key]
    return None


def _count_params_and_memory(model):
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    trainable_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_param_memory_mb": float(total_param_bytes / (1024 ** 2)),
        "trainable_param_memory_mb": float(trainable_param_bytes / (1024 ** 2)),
    }


def _profile_gflops_per_sample(model, device, sample_inputs):
    if sample_inputs is None:
        return 0.0

    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return 0.0

    batch_size = 0
    for _, value in sample_inputs.items():
        if isinstance(value, torch.Tensor):
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        return 0.0

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                _ = model(**sample_inputs)

        total_flops = 0.0
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", 0) or 0
            total_flops += float(evt_flops)

        return float(max(0.0, (total_flops / 1e9) / float(batch_size)))
    except Exception:
        return 0.0
    finally:
        if model_was_training:
            model.train()


def parse_trial_file(trial_path: Path) -> pd.DataFrame:
    suffix = str(trial_path.suffix).lower()

    # Text format: "<label> <path1> <path2>" on each line.
    if suffix in {".txt", ".trials", ".lst"}:
        rows = []
        with open(trial_path, "r", encoding="utf-8") as f:
            for line in f:
                s = str(line).strip()
                if not s:
                    continue
                m = re.match(r"^\s*([01])\s+(.+?)\s+(.+?)\s*$", s)
                if m is None:
                    continue
                rows.append((int(m.group(1)), m.group(2), m.group(3)))
        if len(rows) == 0:
            raise ValueError(f"Trial txt has no valid rows: {trial_path}")
        return pd.DataFrame(rows, columns=["label", "path1", "path2"])

    raw_df = pd.read_csv(trial_path, header=None)
    if raw_df.shape[1] >= 3:
        df = raw_df.iloc[:, :3].copy()
        df.columns = ["label", "path1", "path2"]
    else:
        df = raw_df.rename(columns={0: "raw"}).copy()
        parsed = df["raw"].astype(str).str.strip('"').str.extract(r"^\s*([01])\s+(.+?)\s+(.+?)\s*$")
        parsed.columns = ["label", "path1", "path2"]
        df = parsed
    df["label"] = df["label"].astype(int)
    return df


def _resolve_trial_path(cfg, test_cfg: dict, test_dir: Path):
    trial_file = str(test_cfg.get("trial_file", "")).strip()
    trial_path = Path(trial_file) if trial_file else None

    if trial_path is not None and trial_path.is_absolute() and trial_path.exists():
        return trial_path

    if bool(test_cfg.get("trial_in_test_dir", False)):
        if trial_path is not None and str(trial_path) != "":
            p = test_dir / str(trial_path)
            if p.exists():
                return p

        # Auto-discover common txt trial names inside test directory.
        for pattern in ["list_gt*.txt", "*list*gt*.txt", "*.txt"]:
            found = sorted(test_dir.glob(pattern))
            if len(found) > 0:
                return found[0]

    if trial_path is not None and str(trial_path) != "":
        p = cfg.list_gt_dir / str(trial_path)
        if p.exists():
            return p

    return None


def resolve_hc_path(feature_mode: str, feature_dir: Path):
    feature_mode = str(feature_mode).lower()
    if feature_mode in HC_FILENAME_MAP:
        exact = feature_dir / HC_FILENAME_MAP[feature_mode]
        if exact.exists():
            return exact
    for cand in [feature_dir / f"{feature_mode}.pt", feature_dir / f"all_features_{feature_mode}.pt"]:
        if cand.exists():
            return cand
    return None


def build_lookup_from_pt_file(obj, value_field: str):
    if isinstance(obj, dict) and "filenames" in obj and value_field in obj:
        filenames = obj["filenames"]
        values = obj[value_field]
        lookup = {}
        for idx, name in enumerate(filenames):
            value = values[idx]
            tensor_value = value.float() if hasattr(value, "float") else torch.tensor(value, dtype=torch.float32)
            _register_lookup(lookup, str(name), tensor_value)
        return lookup

    if isinstance(obj, dict):
        lookup = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            tensor_value = v.float() if hasattr(v, "float") else torch.tensor(v, dtype=torch.float32)
            _register_lookup(lookup, k, tensor_value)
        if len(lookup) > 0:
            return lookup

    raise ValueError("Unsupported .pt format for lookup.")


def discover_benchmark_experiments(experiments_root: Path):
    experiments = []
    if not experiments_root.exists():
        return experiments

    known_arches = ["mfa_conformer", "eres2netv2", "redimnet", "campp"]

    for exp_dir in sorted(glob.glob(str(experiments_root / "*"))):
        exp_dir = Path(exp_dir)
        model_path = exp_dir / "best_model.pth"
        config_path = exp_dir / "config.json"
        if not model_path.exists() or not config_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        arch = str(cfg.get("benchmark_arch", "")).lower().strip()
        if arch == "":
            # Backward-compatible fallback for older benchmark runs that did
            # not persist benchmark_arch in config.json.
            exp_name = str(exp_dir.name)
            for candidate in known_arches:
                if exp_name.startswith(f"Bench_{candidate}_"):
                    arch = candidate
                    break
        if arch == "":
            continue

        experiments.append(
            {
                "exp_name": exp_dir.name,
                "exp_dir": str(exp_dir),
                "model_path": str(model_path),
                "arch": arch,
                "feature_mode": str(cfg.get("feature_mode", "mfcc_only")).lower(),
                "duration": str(cfg.get("duration", "")),
                "seed": int(cfg.get("seed", 42)),
            }
        )

    return experiments


def _score_one_experiment(exp_info, trials_df, feature_dir: Path, device: str, p_target: float, repo_root: Path):
    _setup_metric_import_paths(repo_root)
    from metrics import compute_eer, compute_mindcf

    hc_path = resolve_hc_path(exp_info["feature_mode"], feature_dir)
    if hc_path is None:
        print(f"[SKIP] missing feature file for mode={exp_info['feature_mode']} in {feature_dir}")
        return None

    t0 = time.perf_counter()
    hc_lookup = build_lookup_from_pt_file(torch.load(hc_path, map_location="cpu"), value_field="features")

    get_model = build_benchmark_get_model(
        arch_name=exp_info["arch"],
        feature_mode=exp_info["feature_mode"],
        embedding_dim_override=None,
    )
    model = get_model(num_speakers=1000, device=device, mode=2)
    ckpt = torch.load(exp_info["model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    model_stats = _count_params_and_memory(model)

    unique_paths = sorted(set(trials_df["path1"].astype(str).tolist() + trials_df["path2"].astype(str).tolist()))
    path_to_emb = {}
    missing_paths = 0
    sample_inputs_for_profile = None

    with torch.no_grad():
        for p in unique_paths:
            feat = _get_feature_from_lookup(hc_lookup, p)
            if feat is None:
                missing_paths += 1
                continue
            inputs = {"feature": feat.unsqueeze(0).to(device).float()}
            if sample_inputs_for_profile is None:
                sample_inputs_for_profile = {"feature": inputs["feature"]}
            _, emb = model(**inputs)
            path_to_emb[p] = F.normalize(emb, p=2, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if len(path_to_emb) < 2:
        print(
            f"[SKIP] insufficient matched paths for {exp_info['exp_name']} in {feature_dir}. "
            f"matched={len(path_to_emb)}, missing_paths={missing_paths}"
        )
        return None

    paths = list(path_to_emb.keys())
    emb_matrix = np.stack([path_to_emb[p] for p in paths], axis=0)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    pair_i, pair_j, labels = [], [], []
    missing_pairs = 0
    for _, row in trials_df.iterrows():
        p1, p2 = str(row["path1"]), str(row["path2"])
        if p1 not in path_to_idx or p2 not in path_to_idx:
            missing_pairs += 1
            continue
        pair_i.append(path_to_idx[p1])
        pair_j.append(path_to_idx[p2])
        labels.append(int(row["label"]))

    if len(pair_i) == 0:
        print(
            f"[SKIP] no valid trial pairs after key match for {exp_info['exp_name']}. "
            f"missing_pairs={missing_pairs}, trials={len(trials_df)}"
        )
        return None

    pair_i = np.asarray(pair_i, dtype=np.int32)
    pair_j = np.asarray(pair_j, dtype=np.int32)
    labels_np = np.asarray(labels, dtype=np.int32)
    scores = np.sum(emb_matrix[pair_i] * emb_matrix[pair_j], axis=1).astype(np.float32)

    eer_out = compute_eer(labels_np, scores)
    mindcf_out = compute_mindcf(labels_np, scores, p_target=p_target)
    eer = eer_out[0] if isinstance(eer_out, tuple) else eer_out
    mindcf = mindcf_out[0] if isinstance(mindcf_out, tuple) else mindcf_out

    runtime_total = float(time.perf_counter() - t0)
    gflops_per_sample = _profile_gflops_per_sample(
        model=model,
        device=device,
        sample_inputs=sample_inputs_for_profile,
    )
    total_samples_accounted = int(len(path_to_emb))
    gflops_total = float(gflops_per_sample * total_samples_accounted)

    peak_gpu_memory_allocated_mb = 0.0
    peak_gpu_memory_reserved_mb = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        peak_gpu_memory_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    del model
    del ckpt
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "Experiment": exp_info["exp_name"],
        "Arch": exp_info["arch"],
        "Duration": exp_info["duration"],
        "Feature": exp_info["feature_mode"],
        "EER (%)": float(eer * 100.0),
        "MinDCF": float(mindcf),
        "Used Pairs": int(len(labels_np)),
        "Missing Pairs": int(missing_pairs),
        "Missing Paths": int(missing_paths),
        "Runtime Total (s)": runtime_total,
        "GFLOPs/sample": float(gflops_per_sample),
        "GFLOPs total": float(gflops_total),
        "Total Samples Accounted": total_samples_accounted,
        "Peak GPU Memory Allocated (MB)": float(peak_gpu_memory_allocated_mb),
        "Peak GPU Memory Reserved (MB)": float(peak_gpu_memory_reserved_mb),
        "Total Params": model_stats["total_params"],
        "Trainable Params": model_stats["trainable_params"],
        "Param Memory (MB)": model_stats["total_param_memory_mb"],
        "Trainable Param Memory (MB)": model_stats["trainable_param_memory_mb"],
    }


def run_inference_template(cfg):
    cfg.validate_placeholders()

    reports_root = cfg.reports_root
    reports_root.mkdir(parents=True, exist_ok=True)

    experiments = discover_benchmark_experiments(cfg.experiments_root)
    print(f"Found {len(experiments)} benchmark experiments")
    if len(experiments) == 0:
        return []

    all_rows = []
    for test_cfg in cfg.test_set_configs:
        test_name = test_cfg["name"]
        print("=" * 90)
        print(f"[INFER] test_set={test_name}")
        test_dir_cfg = test_cfg.get("test_dir", None)
        if test_dir_cfg is None or str(test_dir_cfg).strip() == "":
            test_dir = cfg.test_base_dir / test_name
        else:
            test_dir = Path(test_dir_cfg)
            if not test_dir.is_absolute():
                test_dir = cfg.test_base_dir / test_dir

        trial_path = _resolve_trial_path(cfg=cfg, test_cfg=test_cfg, test_dir=test_dir)
        if trial_path is None or (not trial_path.exists()):
            print(f"[SKIP] missing trial csv: {trial_path}")
            continue

        feature_dir_cfg = test_cfg.get("feature_dir", None)
        if feature_dir_cfg is None or str(feature_dir_cfg).strip() == "":
            feature_dir = test_dir / "features"
        else:
            feature_dir = Path(feature_dir_cfg)
            if not feature_dir.is_absolute():
                feature_dir = test_dir / feature_dir

        if not feature_dir.exists():
            alt = test_dir / test_cfg.get("duration", "") / "features"
            feature_dir = alt if alt.exists() else feature_dir

        if not feature_dir.exists():
            print(f"[SKIP] missing feature dir: {feature_dir}")
            continue

        print(f"[INFER] trial_path={trial_path}")
        print(f"[INFER] feature_dir={feature_dir}")

        trials_df = parse_trial_file(trial_path)
        print(f"[INFER] trials={len(trials_df)} | experiments={len(experiments)}")
        rows = []

        for exp_info in experiments:
            try:
                print(
                    f"[RUN] {test_name} | exp={exp_info['exp_name']} "
                    f"| arch={exp_info['arch']} | feature={exp_info['feature_mode']}"
                )
                out = _score_one_experiment(
                    exp_info=exp_info,
                    trials_df=trials_df,
                    feature_dir=feature_dir,
                    device=cfg.device,
                    p_target=cfg.p_target,
                    repo_root=cfg.repo_root,
                )
                if out is not None:
                    out["TestSet"] = test_name
                    rows.append(out)
                    all_rows.append(out)
                    print(
                        f"[OK] {exp_info['exp_name']} | EER={out['EER (%)']:.3f} "
                        f"| MinDCF={out['MinDCF']:.4f} | UsedPairs={out['Used Pairs']}"
                    )
            except Exception as ex:
                print(f"[ERROR] {test_name} | {exp_info['exp_name']} -> {type(ex).__name__}: {ex}")

        per_test_path = reports_root / f"benchmark_inference_{test_name}.csv"
        if len(rows) > 0:
            df = pd.DataFrame(rows).sort_values(["EER (%)", "MinDCF"], ascending=[True, True]).reset_index(drop=True)
            df.to_csv(per_test_path, index=False)
            print(f"Saved: {per_test_path}")

    all_path = reports_root / "benchmark_inference_all_testsets.csv"
    if len(all_rows) > 0:
        all_df = pd.DataFrame(all_rows).sort_values(["TestSet", "EER (%)", "MinDCF"], ascending=[True, True, True]).reset_index(drop=True)
        all_df.to_csv(all_path, index=False)
        print(f"Saved: {all_path}")

    return all_rows
