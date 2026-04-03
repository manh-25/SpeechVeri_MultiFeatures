import os
import glob
import torch
import torchaudio
import gc
from pyannote.audio import Pipeline

# ==========================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================================
HF_TOKEN = "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"
INPUT_FOLDER = r"D:\Speak_Verification\Diarization\test_data"
OUTPUT_FOLDER = r"D:\Speak_Verification\Diarization\test_output_final"

# ==========================================================
# 2. CÁC HÀM XỬ LÝ CHÍNH
# ==========================================================

def load_pipeline(token):
    print("🚀 Đang khởi tạo Pyannote 3.1...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("💎 Đang chạy trên GPU (CUDA).")
    else:
        print("⚠️ Không tìm thấy GPU, đang chạy trên CPU.")
    return pipeline

def get_speaker_info(pipeline, audio_path):
    """
    Phân tích file, trả về segments người nói chính và metadata âm thanh.
    Đồng thời in ra thống kê tất cả người nói để kiểm tra nhiễu.
    """
    # Load âm thanh vào RAM (Tránh lỗi AudioDecoder trên Windows)
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
    
    # Chạy mô hình
    raw_output = pipeline(audio_in_memory)
    
    # Xử lý lỗi DiarizeOutput (Bóc tách Annotation)
    if hasattr(raw_output, "speaker_diarization"):
        diarization = raw_output.speaker_diarization
    else:
        diarization = raw_output
        
    speaker_segments = {}
    speaker_durations = {}
    
    # Gom nhóm và tính thời lượng
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
            speaker_durations[speaker] = 0.0
        speaker_segments[speaker].append((turn.start, turn.end))
        speaker_durations[speaker] += (turn.end - turn.start)
    
    if not speaker_segments:
        return None, waveform, sample_rate

    # In bảng thống kê người nói
    sorted_spk = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
    print(f"   📊 Thống kê: {len(sorted_spk)} người nói.")
    for spk, dur in sorted_spk:
        tag = "(Chính)" if spk == sorted_spk[0][0] else "(Phụ)"
        print(f"      - {spk}: {dur:.2f}s {tag}")

    # Lấy người nói chính (thời lượng dài nhất)
    main_speaker = sorted_spk[0][0]
    return speaker_segments[main_speaker], waveform, sample_rate

def process_batch(pipeline, input_dir, output_dir):
    """Quét thư mục và xử lý hàng loạt theo cấu trúc idxxx_yyy"""
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print("❌ Không tìm thấy file .wav nào!")
        return

    print(f"📂 Tìm thấy {len(wav_files)} file. Bắt đầu xử lý...")

    for idx, file_path in enumerate(wav_files):
        filename = os.path.basename(file_path)
        
        # 1. Tách Speaker ID từ tên file (idxxx_yyy.wav -> idxxx)
        spk_id = filename.split('_')[0] if '_' in filename else "unknown"
        
        # 2. Tạo đường dẫn đầu ra (D:\Output\idxxx\filename.wav)
        spk_dir = os.path.join(output_dir, spk_id)
        final_out_path = os.path.join(spk_dir, filename)
        
        # 3. Cơ chế RESUME: Nếu file đã tồn tại thì bỏ qua
        if os.path.exists(final_out_path):
            print(f"[{idx+1}/{len(wav_files)}] ⏭️ Bỏ qua {filename} (Đã xử lý)")
            continue

        print(f"[{idx+1}/{len(wav_files)}] 🔍 Đang xử lý: {filename}")
        
        try:
            # Phân tích
            segments, waveform, sample_rate = get_speaker_info(pipeline, file_path)
            
            if segments:
                # Cắt và ghép các đoạn của người nói chính
                main_spk_waveform = [waveform[:, int(s*sample_rate):int(e*sample_rate)] for s, e in segments]
                final_waveform = torch.cat(main_spk_waveform, dim=1)
                
                # Lưu file
                os.makedirs(spk_dir, exist_ok=True)
                torchaudio.save(final_out_path, final_waveform, sample_rate)
                print(f"      ✅ Đã lưu vào: {spk_id}/")
            
            # 4. QUẢN LÝ BỘ NHỚ: Dọn rác sau mỗi file để tránh tràn 70GB
            del waveform, segments
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"      ❌ Lỗi tại file {filename}: {e}")

# ==========================================================
# 3. KÍCH HOẠT
# ==========================================================
if __name__ == "__main__":
    # Khởi tạo 1 lần duy nhất
    pipe = load_pipeline(HF_TOKEN)
    
    # Chạy hàng loạt
    process_batch(pipe, INPUT_FOLDER, OUTPUT_FOLDER)
    
    print("\n✨ TẤT CẢ ĐÃ HOÀN THÀNH!")