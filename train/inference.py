import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import os
import time
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
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

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
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    
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
        "Total Params": model_stats["total_params"],
        "Trainable Params": model_stats["trainable_params"],
        "Param Memory (MB)": model_stats["total_param_memory_mb"],
        "Trainable Param Memory (MB)": model_stats["trainable_param_memory_mb"],
    }