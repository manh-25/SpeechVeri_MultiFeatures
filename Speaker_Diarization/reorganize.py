import shutil
from pathlib import Path

def flatten_safely(split_dir: str, flat_dir: str):
    split_path = Path(split_dir)
    flat_path = Path(flat_dir)
    
    # Tạo thư mục đích
    flat_path.mkdir(parents=True, exist_ok=True)
    
    print("Đang sao chép các file, vui lòng đợi...")
    copied_count = 0
    
    for file_path in split_path.rglob('*.*'):
        if file_path.is_file():
            # Đường dẫn đích dự kiến ban đầu
            new_path = flat_path / file_path.name
            
            # XỬ LÝ TRÙNG TÊN: Nếu file đã tồn tại, tự động thêm số vào cuối tên file
            counter = 1
            while new_path.exists():
                new_name = f"{file_path.stem}_{counter}{file_path.suffix}"
                new_path = flat_path / new_name
                counter += 1
                
            # Dùng copy2 để sao chép an toàn (giữ nguyên ngày tháng tạo file)
            shutil.copy2(file_path, new_path)
            copied_count += 1
            
    print(f"\nTuyệt vời! Đã gom thành công toàn bộ {copied_count} file vào thư mục: {flat_dir}")

# ==========================================
# ĐIỀN ĐƯỜNG DẪN CỦA BẠN VÀO ĐÂY
# ==========================================
SPLIT_FOLDER = "D:\\Speak_Verification\\Diarization\\test_output3_final - Copy"     # Nơi chứa các folder idxxx
FLAT_FOLDER = "D:\\Speak_Verification\\Diarization\\final_folder"    # Nơi lưu TẤT CẢ file sau khi gom

if __name__ == "__main__":
    flatten_safely(SPLIT_FOLDER, FLAT_FOLDER)