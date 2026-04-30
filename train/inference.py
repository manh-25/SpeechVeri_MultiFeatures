import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import csv
import json
import os
import time
from datetime import datetime
from metrics import compute_eer, compute_mindcf
from tqdm import tqdm


def _count_params_and_memory(model):
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    trainable_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_param_memory_bytes": total_param_bytes,
        "total_param_memory_mb": float(total_param_bytes / (1024 ** 2)),
        "trainable_param_memory_bytes": trainable_param_bytes,
        "trainable_param_memory_mb": float(trainable_param_bytes / (1024 ** 2)),
    }


def _profile_inference_gflops_per_sample(model, device, sample_inputs):
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


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sanitize_exp_name(exp_name):
    raw = str(exp_name or "").strip()
    if not raw:
        raw = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    return safe or f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _extract_metric_by_prefix(metrics, prefix):
    for key, value in metrics.items():
        if str(key).startswith(prefix):
            return value
    return None


def _build_inference_flat_row(final_results):
    config = final_results.get("config", {}) or {}
    metrics = final_results.get("metrics", {}) or {}
    model_stats = final_results.get("model_stats", {}) or {}
    performance = final_results.get("performance", {}) or {}

    row = {
        "exp_name": final_results.get("exp_name"),
        "exp_dir": final_results.get("exp_dir"),
        "timestamp": final_results.get("timestamp"),
        "mode": config.get("mode"),
        "fusion_method": config.get("fusion_method"),
        "feature_mode": config.get("feature_mode"),
        "duration": config.get("duration"),
        "pretrained_model": config.get("pretrained_model"),
        "trials_path": config.get("trials_path"),
        "num_pairs": config.get("num_pairs"),
        "max_trials": config.get("max_trials"),
        "p_target": config.get("p_target"),
        "eer_percent": metrics.get("EER (%)"),
        "eer_threshold": metrics.get("EER Threshold"),
        "mindcf": _extract_metric_by_prefix(metrics, "MinDCF (p="),
        "mindcf_threshold": metrics.get("MinDCF Threshold"),
        "runtime_total_sec": performance.get("total_runtime_sec"),
        "runtime_extract_sec": performance.get("extract_time_sec"),
        "runtime_score_sec": performance.get("score_time_sec"),
        "gflops_per_sample": performance.get("gflops_per_sample"),
        "gflops_total": performance.get("gflops_total"),
        "total_samples_accounted": performance.get("total_samples_accounted"),
        "peak_gpu_memory_allocated_mb": performance.get("peak_gpu_memory_allocated_mb"),
        "peak_gpu_memory_reserved_mb": performance.get("peak_gpu_memory_reserved_mb"),
        "total_params": model_stats.get("total_params"),
        "trainable_params": model_stats.get("trainable_params"),
        "total_param_memory_mb": model_stats.get("total_param_memory_mb"),
        "trainable_param_memory_mb": model_stats.get("trainable_param_memory_mb"),
    }

    # Keep all raw metrics for full traceability in CSV.
    for key, value in metrics.items():
        csv_key = "metric_" + str(key).strip().lower().replace(" ", "_").replace("%", "percent")
        row[csv_key] = value

    return row


def _write_single_row_csv(csv_path, row):
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _update_inference_summary(experiment_root, row):
    os.makedirs(experiment_root, exist_ok=True)
    summary_json = os.path.join(experiment_root, "summary_all_modes.json")
    summary_csv = os.path.join(experiment_root, "summary_all_modes.csv")

    rows = []
    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r", encoding="utf-8") as file:
                payload = json.load(file)
                if isinstance(payload, list):
                    rows = payload
        except Exception:
            rows = []

    rows = [item for item in rows if item.get("exp_dir") != row.get("exp_dir")]
    rows.append(row)
    rows = sorted(rows, key=lambda x: (str(x.get("mode", "")), str(x.get("exp_name", ""))))

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(_to_serializable(rows), file, indent=2)

    if len(rows) > 0:
        # Union all keys to keep old/new rows compatible if schema evolves.
        fieldnames = []
        seen = set()
        for item in rows:
            for key in item.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(summary_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow(item)

    return summary_json, summary_csv


def save_inference_artifacts(
    output_dir,
    eval_results,
    exp_name=None,
    config_snapshot=None,
    experiments_dirname="inference_experiments",
):
    """Save train-like inference artifacts: config/results JSON+CSV and run summary."""
    if not isinstance(eval_results, dict):
        raise ValueError("eval_results must be a dict returned by evaluate_speaker_verification().")

    exp_name = _sanitize_exp_name(exp_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    experiment_root = os.path.join(str(output_dir), str(experiments_dirname))
    exp_dir = os.path.join(experiment_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config_snapshot = dict(config_snapshot or {})
    config_snapshot.setdefault("exp_name", exp_name)
    config_snapshot.setdefault("timestamp", timestamp)

    model_stats = {
        "total_params": eval_results.get("Total Params"),
        "trainable_params": eval_results.get("Trainable Params"),
        "total_param_memory_mb": eval_results.get("Param Memory (MB)"),
        "trainable_param_memory_mb": eval_results.get("Trainable Param Memory (MB)"),
    }

    performance = {
        "total_runtime_sec": eval_results.get("Runtime Total (s)"),
        "extract_time_sec": eval_results.get("Runtime Extract (s)"),
        "score_time_sec": eval_results.get("Runtime Score (s)"),
        "gflops_per_sample": eval_results.get("GFLOPs/sample"),
        "gflops_total": eval_results.get("GFLOPs total"),
        "total_samples_accounted": eval_results.get("Total Samples Accounted"),
        "peak_gpu_memory_allocated_mb": eval_results.get("Peak GPU Memory Allocated (MB)"),
        "peak_gpu_memory_reserved_mb": eval_results.get("Peak GPU Memory Reserved (MB)"),
    }

    final_results = {
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "timestamp": timestamp,
        "config": config_snapshot,
        "metrics": dict(eval_results),
        "model_stats": model_stats,
        "performance": performance,
    }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(_to_serializable(config_snapshot), file, indent=2)

    results_json_path = os.path.join(exp_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as file:
        json.dump(_to_serializable(final_results), file, indent=2)

    flat_row = _build_inference_flat_row(final_results)
    results_csv_path = os.path.join(exp_dir, "results.csv")
    _write_single_row_csv(results_csv_path, _to_serializable(flat_row))

    summary_json_path, summary_csv_path = _update_inference_summary(
        experiment_root=experiment_root,
        row=_to_serializable(flat_row),
    )

    return {
        "exp_dir": exp_dir,
        "config_path": config_path,
        "results_json_path": results_json_path,
        "results_csv_path": results_csv_path,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
    }


def _safe_test_tag(name):
    text = str(name or "test").strip()
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return text or "test"


def _coerce_report_row(exp_name, test_name, eval_results):
    row = {
        "Experiment": str(exp_name),
        "Test Set": str(test_name),
    }
    for key, value in dict(eval_results or {}).items():
        row[str(key)] = _to_serializable(value)
    return row


def _write_rows_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            file.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_inference_benchmark_reports(
    model,
    test_loaders,
    device,
    output_dir,
    exp_name,
    p_target=0.05,
    trials_by_test=None,
    max_trials=None,
    per_test_filename_template="{exp_name}_{test_tag}.csv",
    all_report_filename="{exp_name}_all.csv",
):
    """
    Run evaluation across test sets and save exactly N+1 CSV files:
    - N per-test files (one per test set)
    - 1 aggregated all-tests file

    Each CSV row includes full inference metrics returned by
    evaluate_speaker_verification (EER, MinDCF, runtime, GFLOPs, params, memory...).
    """
    if not isinstance(test_loaders, dict) or len(test_loaders) == 0:
        raise ValueError("test_loaders must be a non-empty dict: {test_name: data_loader}.")

    exp_name = _sanitize_exp_name(exp_name)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows_by_test = {}
    all_rows = []
    file_map = {}

    for test_name, loader in test_loaders.items():
        test_tag = _safe_test_tag(test_name)
        trials_path = None
        if isinstance(trials_by_test, dict):
            trials_path = trials_by_test.get(test_name)
            if trials_path is None:
                trials_path = trials_by_test.get(test_tag)

        eval_results = evaluate_speaker_verification(
            model=model,
            data_loader=loader,
            device=device,
            p_target=p_target,
            trials_path=trials_path,
            max_trials=max_trials,
        )
        row = _coerce_report_row(exp_name=exp_name, test_name=test_name, eval_results=eval_results)

        rows_by_test.setdefault(test_name, []).append(row)
        all_rows.append(row)

        per_test_filename = per_test_filename_template.format(exp_name=exp_name, test_tag=test_tag)
        per_test_path = os.path.join(output_dir, per_test_filename)
        _write_rows_csv(per_test_path, rows_by_test[test_name])
        file_map[str(test_name)] = per_test_path

    all_report_path = os.path.join(output_dir, all_report_filename.format(exp_name=exp_name))
    _write_rows_csv(all_report_path, all_rows)

    return {
        "exp_name": exp_name,
        "per_test_paths": file_map,
        "all_report_path": all_report_path,
        "num_test_files": len(file_map),
        "total_files": len(file_map) + 1,
        "rows": _to_serializable(all_rows),
    }


def _candidate_keys(path_text):
    p = str(path_text).replace('\\', '/').strip()
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    rel = p.lstrip('./')
    candidates = [p, rel, base, stem]
    seen = set()
    out = []
    for key in candidates:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _load_trials_file(trials_path, max_trials=None):
    trials = []
    with open(trials_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            label = int(parts[0])
            path1 = parts[1]
            path2 = parts[2]
            trials.append((label, path1, path2))
            if max_trials is not None and len(trials) >= max_trials:
                break
    return trials

def evaluate_speaker_verification(
    model,
    data_loader,
    device,
    num_pairs=50000,
    p_target=0.05,
    trials_path=None,
    max_trials=None,
):
    """
    Đánh giá mô hình trên tập Test với thuật toán Lấy mẫu cân bằng (Balanced Sampling)
    Đảm bảo tỷ lệ Positive / Negative là 50/50 để đo EER chuẩn xác nhất.
    """
    model.eval()
    run_start_time = time.perf_counter()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    model_stats = _count_params_and_memory(model)
    all_embeddings = []
    all_labels = []
    all_utt_ids = []

    selected_trials = None
    required_trial_keys = None
    if trials_path is not None and os.path.exists(trials_path):
        selected_trials = _load_trials_file(trials_path, max_trials=max_trials)
        required_trial_keys = set()
        for _, path1, path2 in selected_trials:
            for key in _candidate_keys(path1):
                required_trial_keys.add(key)
            for key in _candidate_keys(path2):
                required_trial_keys.add(key)

    print("\n[Evaluation] Trích xuất Embeddings từ tập Test...")
    sample_inputs_for_profile = None
    extract_start_time = time.perf_counter()
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting"):
            utt_ids = batch_data.get("utt_id")
            labels = batch_data["label"]

            selected_indices = None
            if required_trial_keys is not None and utt_ids is not None:
                selected_indices = []
                for idx, utt in enumerate(utt_ids):
                    utt_key_hit = False
                    for key in _candidate_keys(utt):
                        if key in required_trial_keys:
                            utt_key_hit = True
                            break
                    if utt_key_hit:
                        selected_indices.append(idx)

                if len(selected_indices) == 0:
                    continue

            inputs = {
                k: v.to(device)
                for k, v in batch_data.items()
                if isinstance(v, torch.Tensor) and k != "label"
            }

            if sample_inputs_for_profile is None and len(inputs) > 0:
                sample_inputs_for_profile = {k: v for k, v in inputs.items()}

            if selected_indices is not None:
                idx_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
                labels = labels.index_select(0, idx_tensor)
                inputs = {k: v.index_select(0, idx_tensor).to(device) for k, v in inputs.items()}
                utt_ids = [utt_ids[i] for i in selected_indices]
            
            # Trích xuất embedding (bỏ qua logits)
            _, embeddings = model(**inputs) 
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            if utt_ids is None:
                all_utt_ids.extend([str(len(all_utt_ids) + i) for i in range(len(labels))])
            else:
                all_utt_ids.extend([str(item) for item in utt_ids])
    extract_time_sec = float(time.perf_counter() - extract_start_time)

    gflops_per_sample = _profile_inference_gflops_per_sample(
        model=model,
        device=device,
        sample_inputs=sample_inputs_for_profile,
    )

    if len(all_embeddings) == 0:
        raise ValueError("Không trích xuất được embedding nào từ data_loader.")

    # Gộp và Normalize vector 1 lần duy nhất
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if selected_trials is not None:
        print(f"[Evaluation] Scoring fixed trials from: {trials_path}")
        score_start_time = time.perf_counter()

        key_to_indices = defaultdict(list)
        for index, utt in enumerate(all_utt_ids):
            for key in _candidate_keys(utt):
                key_to_indices[key].append(index)

        idx1_list, idx2_list, y_true = [], [], []
        missing_trials = 0
        for label, path1, path2 in selected_trials:
            indices1 = None
            indices2 = None
            for key in _candidate_keys(path1):
                if key in key_to_indices:
                    indices1 = key_to_indices[key]
                    break
            for key in _candidate_keys(path2):
                if key in key_to_indices:
                    indices2 = key_to_indices[key]
                    break

            if not indices1 or not indices2:
                missing_trials += 1
                continue

            idx1_list.append(indices1[0])
            idx2_list.append(indices2[0])
            y_true.append(label)

        if len(idx1_list) == 0:
            raise ValueError("No valid trial pairs matched current utterance IDs.")

        emb1 = all_embeddings[idx1_list]
        emb2 = all_embeddings[idx2_list]

        scores = np.sum(emb1 * emb2, axis=1)
        y_true = np.array(y_true)

        eer, eer_thresh = compute_eer(y_true, scores)
        min_dcf, dcf_thresh = compute_mindcf(y_true, scores, p_target=p_target)

        score_time_sec = float(time.perf_counter() - score_start_time)
        total_runtime_sec = float(time.perf_counter() - run_start_time)
        total_samples_accounted = int(len(all_labels))
        peak_gpu_memory_allocated_mb = 0.0
        peak_gpu_memory_reserved_mb = 0.0
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            peak_gpu_memory_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

        return {
            "Num Pairs": int(len(y_true)),
            "Missing Trials": int(missing_trials),
            "EER (%)": float(eer * 100),
            "EER Threshold": float(eer_thresh),
            f"MinDCF (p={p_target})": float(min_dcf),
            "MinDCF Threshold": float(dcf_thresh),
            "Runtime Total (s)": total_runtime_sec,
            "Runtime Extract (s)": extract_time_sec,
            "Runtime Score (s)": score_time_sec,
            "GFLOPs/sample": float(gflops_per_sample),
            "GFLOPs total": float(gflops_per_sample * total_samples_accounted),
            "Total Samples Accounted": total_samples_accounted,
            "Peak GPU Memory Allocated (MB)": peak_gpu_memory_allocated_mb,
            "Peak GPU Memory Reserved (MB)": peak_gpu_memory_reserved_mb,
            "Total Params": model_stats["total_params"],
            "Trainable Params": model_stats["trainable_params"],
            "Param Memory (MB)": model_stats["total_param_memory_mb"],
            "Trainable Param Memory (MB)": model_stats["trainable_param_memory_mb"],
        }
    
    print(f"[Evaluation] Đã trích xuất {len(all_labels)} vector. Đang tạo các cặp cân bằng...")
    score_start_time = time.perf_counter()
    
    # 1. Phân nhóm index theo ID người nói
    speaker_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        speaker_indices[label].append(idx)

    # 2. Tạo Positive Pairs (Cùng người)
    pos_pairs = []
    for label, indices in speaker_indices.items():
        n = len(indices)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    pos_pairs.append((indices[i], indices[j]))

    random.seed(42) # Cố định seed để dễ so sánh giữa các lần test
    random.shuffle(pos_pairs)
    
    # Giới hạn số lượng cặp Cùng người (1 nửa của num_pairs)
    half_pairs = num_pairs // 2
    if len(pos_pairs) > half_pairs:
        pos_pairs = pos_pairs[:half_pairs]

    # 3. Tạo Negative Pairs (Khác người) BẰNG ĐÚNG số lượng Positive
    neg_pairs = []
    labels_unique = list(speaker_indices.keys())
    
    if len(labels_unique) < 2:
        raise ValueError("Tập Test phải có ít nhất 2 người nói khác nhau để tạo Negative Pairs.")
        
    while len(neg_pairs) < len(pos_pairs):
        spk1, spk2 = random.sample(labels_unique, 2)
        idx1 = random.choice(speaker_indices[spk1])
        idx2 = random.choice(speaker_indices[spk2])
        neg_pairs.append((idx1, idx2))

    # 4. Tính điểm Cosine Similarity (Tối ưu RAM, chỉ tính cho các cặp đã chọn)
    print(f"[Evaluation] Đang tính Cosine Similarity cho {len(pos_pairs)} cặp Pos và {len(neg_pairs)} cặp Neg...")
    scores = []
    y_true = []

    for idx1, idx2 in pos_pairs:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(1)

    for idx1, idx2 in neg_pairs:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(0)

    scores = np.array(scores)
    y_true = np.array(y_true)
    
    # 5. Tính toán EER & MinDCF cuối cùng
    print("[Evaluation] Đang tính toán EER và MinDCF...")
    eer, eer_thresh = compute_eer(y_true, scores)
    min_dcf, dcf_thresh = compute_mindcf(y_true, scores, p_target=p_target)
    score_time_sec = float(time.perf_counter() - score_start_time)
    total_runtime_sec = float(time.perf_counter() - run_start_time)
    total_samples_accounted = int(len(all_labels))
    peak_gpu_memory_allocated_mb = 0.0
    peak_gpu_memory_reserved_mb = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        peak_gpu_memory_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
    
    return {
        "EER (%)": float(eer * 100),
        "EER Threshold": float(eer_thresh),
        f"MinDCF (p={p_target})": float(min_dcf),
        "MinDCF Threshold": float(dcf_thresh),
        "Total Balanced Pairs": len(scores), # Log thêm để biết đã test trên bao nhiêu cặp
        "Runtime Total (s)": total_runtime_sec,
        "Runtime Extract (s)": extract_time_sec,
        "Runtime Score (s)": score_time_sec,
        "GFLOPs/sample": float(gflops_per_sample),
        "GFLOPs total": float(gflops_per_sample * total_samples_accounted),
        "Total Samples Accounted": total_samples_accounted,
        "Peak GPU Memory Allocated (MB)": peak_gpu_memory_allocated_mb,
        "Peak GPU Memory Reserved (MB)": peak_gpu_memory_reserved_mb,
        "Total Params": model_stats["total_params"],
        "Trainable Params": model_stats["trainable_params"],
        "Param Memory (MB)": model_stats["total_param_memory_mb"],
        "Trainable Param Memory (MB)": model_stats["trainable_param_memory_mb"],
    }