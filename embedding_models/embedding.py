import torch
import torchaudio
import glob
import os
import numpy as np
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    HubertModel,
    WavLMModel
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load processor for all model input
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base") # Do các model WavLM, HUBERT không hỗ trợ input raw audio, nên cần lấy cấu trúc dữ liệu nhận raw audio từ wav2vec2processor, không ảnh hưởng khi embedding = model khác

wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-base"
).to(device).eval()

hubert = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960"
).to(device).eval()

wav2vec2 = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device).eval()

def load_clean_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    assert sr == 16000
    assert waveform.dtype == torch.float32

    return waveform

@torch.no_grad() # Không tính gradient để tiết kiệm bộ nhớ
def wavlm_embedding(waveform):
    inputs = processor(
        waveform.squeeze(0),
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    outputs = wavlm(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

@torch.no_grad()
def hubert_embedding(waveform):
    inputs = processor(
        waveform.squeeze(0),
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    outputs = hubert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

@torch.no_grad()
def wav2vec2_embedding(waveform):
    inputs = processor(
        waveform.squeeze(0),
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    outputs = wav2vec2(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

def process_speaker_folder(folder_path, embedding_func, save_path=None):
    """
    Process all audio files in a folder and return embeddings in PyTorch format.
    Recursively searches for audio files at any depth.
    
    Returns:
        dict: Contains 'embeddings' (stacked tensor) and 'speaker_ids' (list of speaker IDs)
    """

    # Normalize path
    folder_path = os.path.abspath(folder_path)

    # Extract speaker ID from the TOP folder (id00002)
    id_speaker = os.path.basename(folder_path.rstrip(os.sep))

    # Recursively find all wav files
    audio_files = glob.glob(
        os.path.join(folder_path, "**", "*.wav"),
        recursive=True
    )

    if len(audio_files) == 0:
        print(f"No wav files found in {folder_path}")
        return None

    embeddings_list = []
    ids_list = []

    for audio_file in audio_files:
        try:
            waveform = load_clean_audio(audio_file)
            embedding = embedding_func(waveform)

            embeddings_list.append(embedding)
            ids_list.append(id_speaker)

            print(f"Processed: {audio_file}")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    print(f"\nTotal files processed for {id_speaker}: {len(embeddings_list)}")

    # Stack embeddings into a tensor
    embeddings_tensor = torch.stack(embeddings_list)
    
    # Create output dictionary
    output_data = {
        'embeddings': embeddings_tensor,
        'speaker_ids': ids_list
    }
    
    # Save to .pt file if save_path provided
    if save_path:
        torch.save(output_data, save_path)
        print(f"Saved to {save_path}")
    
    return output_data

# Example usage:
wav2vec2_id00002 = process_speaker_folder(r'D:\Speak_Verification\id00002', wav2vec2_embedding, save_path='embeddings_id00002_wav2vec2.pt')

