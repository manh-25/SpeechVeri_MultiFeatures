import os
from pathlib import Path

def find_missing_files(source_dir: str, target_dir: str, file_extension: str = ".wav"):
    """
    So sánh thư mục gốc và thư mục đích để tìm các file bị thiếu.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    print("Đang quét thư mục gốc...")
    # Lấy tập hợp tên tất cả các file trong thư mục chưa tách
    source_files = {f.name for f in source_path.glob(f"*{file_extension}")}

    print("Đang quét thư mục đã tách...")
    # rglob giúp quét đệ quy (recursive) tìm tất cả các file bên trong các thư mục con idxxx
    target_files = {f.name for f in target_path.rglob(f"*{file_extension}")}

    # Dùng phép trừ tập hợp (set difference) để tìm file có ở gốc nhưng không có ở đích
    missing_files = source_files - target_files

    # In báo cáo thống kê
    print("-" * 30)
    print(f"Tổng số file ban đầu: {len(source_files)}")
    print(f"Tổng số file sau khi tách: {len(target_files)}")
    print(f"Số lượng file bị thiếu: {len(missing_files)}")
    print("-" * 30)

    # Nếu có file thiếu, in ra và ghi vào log
    if missing_files:
        log_filename = "missing_files_log.txt"
        with open(log_filename, "w", encoding="utf-8") as f:
            for missing_file in sorted(missing_files):
                f.write(f"{missing_file}\n")
        
        print(f"\nĐã tìm thấy {len(missing_files)} file bị thiếu!")
        print(f"Danh sách chi tiết đã được lưu vào file: {log_filename}")
        
        # In thử 5 file đầu tiên ra console để bạn xem nhanh
        print("Một số file bị thiếu (Top 5):")
        for file in list(missing_files):
            print(f" - {file}")
    else:
        print("\nTuyệt vời! Dữ liệu đã được chuyển đủ, không có file nào bị thất thoát.")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN Ở ĐÂY
# ==========================================
SOURCE_DIR = "D:\\Speak_Verification\\Diarization\\test_data" # Thư mục chứa 20,000 file idxxx_yyyy
TARGET_DIR = "D:\\Speak_Verification\\Diarization\\test_output3_final - Copy"   # Thư mục lớn chứa các thư mục con idxxx

if __name__ == "__main__":
    find_missing_files(SOURCE_DIR, TARGET_DIR, file_extension=".wav")