import os
import torch
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def worker_bypass_ipc(task_file):
    """
    Tiến trình con chỉ nhận 1 chuỗi string (tên file lệnh). 
    Hoàn toàn không đi qua ống nước IPC của Windows!
    """
    # Ép luồng này không được ôm quá nhiều CPU ảo
    torch.set_num_threads(1)
    
    # 1. Đọc file lệnh
    task_data = torch.load(task_file, map_location='cpu')
    
    features = []
    valid_speaker_ids = []
    valid_filenames = []
    
    # 2. Làm việc nặng nhọc
    for hc_path, spk_id, fname in zip(task_data['hc_paths'], task_data['speaker_ids'], task_data['filenames']):
        if hc_path is not None:
            feat = torch.load(hc_path, map_location='cpu')
            features.append(feat.half()) # Ép float16 cho nhẹ RAM
            valid_speaker_ids.append(spk_id)
            valid_filenames.append(fname)
            
    # 3. Lưu thành quả Shard lớn
    torch.save({
        'features': features,
        'speaker_ids': valid_speaker_ids,
        'filenames': valid_filenames
    }, os.path.join(task_data['save_dir'], task_data['target_shard_name']))
    
    # 4. Làm xong thì phi tang file lệnh
    try:
        os.remove(task_file)
    except:
        pass
        
    return task_data['target_shard_name']

def pack_bulletproof(ptm_shard_dir, hc_dir, save_dir, num_processes=8):
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = os.path.join(save_dir, "temp_ipc_bypass")
    os.makedirs(temp_dir, exist_ok=True)
    
    ptm_shards = sorted(glob.glob(os.path.join(ptm_shard_dir, "*.pt")))
    
    print("🔍 Đang quét các file Handcrafted lẻ...")
    hc_files = glob.glob(os.path.join(hc_dir, "**", "*.pt"), recursive=True)
    hc_map = {os.path.basename(f): f for f in hc_files}
    print(f"✅ Đã tìm thấy {len(hc_files)} file đặc trưng.")
    
    print("⚙️ Tạo file lệnh rẽ nhánh (Chống kẹt IPC Windows)...")
    task_files = []
    
    for i, ptm_path in enumerate(tqdm(ptm_shards, desc="Ghi file lệnh")):
        ptm_data = torch.load(ptm_path, map_location='cpu')
        shard_name = os.path.basename(ptm_path)
        target_shard_name = shard_name.replace("hubert", "hc").replace("wavlm", "hc").replace("wav2vec2", "hc")
        
        hc_paths = []
        for fname in ptm_data['filenames']:
            pt_name = os.path.splitext(fname)[0] + ".pt"
            hc_paths.append(hc_map.get(pt_name, None))
            
        # Ném toàn bộ Data nặng xuống ổ cứng
        task_file = os.path.join(temp_dir, f"task_{i}.pt")
        torch.save({
            'target_shard_name': target_shard_name,
            'save_dir': save_dir,
            'hc_paths': hc_paths,
            'speaker_ids': ptm_data['speaker_ids'],
            'filenames': ptm_data['filenames']
        }, task_file)
        
        task_files.append(task_file) # Chỉ gửi mỗi cái tên file này cho luồng con
        
    print(f"🚀 Bắt đầu đóng gói với {num_processes} TIẾN TRÌNH...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Gửi đúng chuỗi string nhẹ hều qua IPC
        futures = [executor.submit(worker_bypass_ipc, tf) for tf in task_files]
        for future in tqdm(as_completed(futures), total=len(task_files), desc="Đóng gói Shards"):
            future.result()
            
    # Dọn dẹp thư mục tạm
    try:
        os.rmdir(temp_dir)
    except:
        pass
        
    print(f"\n🎉 HOÀN THÀNH TỐC ĐỘ CAO! Đã lưu vào: {save_dir}")    
    
PTM_SHARD_DIR = r"D:\Embeddings\train_raw\hubert_shards"
HANDCRAFTED_DIR = r"H:\extracted_features\train_raw\Only MFBE"
OUTPUT_DIR = r"D:\extracted_features\train_raw\mfbe_shards"

if __name__ == "__main__":
    pack_bulletproof(PTM_SHARD_DIR, HANDCRAFTED_DIR, OUTPUT_DIR, num_processes=8)