import numpy as np

def compute_eer(y_true, scores):
    y_true = np.array(y_true)
    scores = np.array(scores)
    
    # Sắp xếp scores từ thấp đến cao
    indices = np.argsort(scores)
    y_true = y_true[indices]
    scores = scores[indices]
    
    # Tính số lượng mẫu Positive (target) và Negative (impostor)
    num_pos = np.sum(y_true)
    num_neg = len(y_true) - num_pos
    
    # Tính False Negative Rate (FNR) - Tỷ lệ bị từ chối sai
    # Càng nhiều mẫu target nằm dưới ngưỡng thì FNR càng cao
    fnr = np.cumsum(y_true) / num_pos
    
    # Tính False Positive Rate (FPR) - Tỷ lệ chấp nhận sai
    # Càng ít mẫu impostor nằm trên ngưỡng thì FPR càng thấp
    fpr = 1 - np.cumsum(1 - y_true) / num_neg
    
    # Tìm chỉ số nơi FNR và FPR gần nhau nhất (điểm EER)
    idx = np.nanargmin(np.absolute(fnr - fpr))
    
    # EER là giá trị trung bình tại điểm giao thoa đó
    eer = (fnr[idx] + fpr[idx]) / 2
    
    return eer, scores[idx]

def compute_mindcf(y_true, scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    y_true = np.array(y_true)
    scores = np.array(scores)
    
    indices = np.argsort(scores)
    y_true = y_true[indices]
    scores = scores[indices]
    
    num_pos = np.sum(y_true)
    num_neg = len(y_true) - num_pos
    
    fnr = np.cumsum(y_true) / num_pos
    fpr = 1 - np.cumsum(1 - y_true) / num_neg
    
    # 1. Tính Detection Cost thô
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_c_det = np.min(dcf)
    
    # 2. Tính Default Cost (Hệ thống tệ nhất có thể làm)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    
    # 3. Chuẩn hóa (Đây là lý do minDCF sẽ nhảy từ 0.02 lên tầm 0.4)
    min_dcf_norm = min_c_det / c_def
    
    return min_dcf_norm, scores[np.argmin(dcf)]