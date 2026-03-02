import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from metrics import compute_eer, compute_mindcf
from tqdm import tqdm

def evaluate_speaker_verification(model, data_loader, device, num_pairs=50000, p_target=0.05):
    """
    Đánh giá mô hình trên tập Test với thuật toán Lấy mẫu cân bằng (Balanced Sampling)
    Đảm bảo tỷ lệ Positive / Negative là 50/50 để đo EER chuẩn xác nhất.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    print("\n[Evaluation] Trích xuất Embeddings từ tập Test...")
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting"):
            labels = batch_data["label"]
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}
            
            # Trích xuất embedding (bỏ qua logits)
            _, embeddings = model(**inputs) 
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    # Gộp và Normalize vector 1 lần duy nhất
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"[Evaluation] Đã trích xuất {len(all_labels)} vector. Đang tạo các cặp cân bằng...")
    
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
    
    return {
        "EER (%)": float(eer * 100),
        "EER Threshold": float(eer_thresh),
        f"MinDCF (p={p_target})": float(min_dcf),
        "MinDCF Threshold": float(dcf_thresh),
        "Total Balanced Pairs": len(scores) # Log thêm để biết đã test trên bao nhiêu cặp
    }