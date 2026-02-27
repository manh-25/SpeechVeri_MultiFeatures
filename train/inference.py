import torch
import torch.nn.functional as F
import numpy as np
from metrics import compute_eer, compute_mindcf
from tqdm import tqdm

def evaluate_speaker_verification(model, data_loader, device, num_pairs=50000, p_target=0.05):
    model.eval()
    all_embeddings = []
    all_labels = []

    print("\n[Evaluation] Extracting embeddings...")
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting"):
            labels = batch_data["label"]
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}
            _, embeddings = model(**inputs) 
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = F.normalize(torch.cat(all_embeddings, dim=0), p=2, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    
    print("[Evaluation] Computing similarity scores...")
    sim_matrix = torch.mm(all_embeddings, all_embeddings.T).numpy()
    label_matrix = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).numpy().astype(int)
    
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    scores = sim_matrix[triu_indices]
    y_true = label_matrix[triu_indices]
    
    if len(scores) > num_pairs:
        indices = np.random.choice(len(scores), num_pairs, replace=False)
        scores = scores[indices]
        y_true = y_true[indices]
    
    eer, eer_thresh = compute_eer(y_true, scores)
    min_dcf, dcf_thresh = compute_mindcf(y_true, scores, p_target=p_target)
    
    return {
        "EER (%)": float(eer * 100),
        "EER Threshold": float(eer_thresh),
        f"MinDCF (p={p_target})": float(min_dcf),
        "MinDCF Threshold": float(dcf_thresh)
    }