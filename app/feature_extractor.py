from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torchaudio
import torchaudio.transforms as T

try:
    import librosa
except Exception:
    librosa = None

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
except Exception:
    Wav2Vec2FeatureExtractor = None
    WavLMModel = None


@dataclass
class ExtractorConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    pitch_fmin: float = 60.0
    pitch_fmax: float = 500.0
    ptm_repo: str = "microsoft/wavlm-base"


class DemoFeatureExtractor:
    def __init__(self, device: str = "cpu", config: ExtractorConfig | None = None) -> None:
        self.device = torch.device(device)
        self.config = config or ExtractorConfig()
        self.mel = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            center=False,
        ).to(self.device)
        self._ptm_model = None
        self._ptm_processor = None

    def _get_ptm_components(self):
        if self._ptm_model is not None and self._ptm_processor is not None:
            return self._ptm_model, self._ptm_processor

        if WavLMModel is None or Wav2Vec2FeatureExtractor is None:
            raise ImportError(
                "transformers chưa được cài đặt. Hãy cài dependencies trong app/requirements.txt"
            )

        self._ptm_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.ptm_repo)
        self._ptm_model = WavLMModel.from_pretrained(
            self.config.ptm_repo,
            output_hidden_states=True,
        ).to(self.device).eval()
        return self._ptm_model, self._ptm_processor

    @staticmethod
    def _cmvn(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True).clamp_min(eps)
        return (feat - mean) / std

    @staticmethod
    def _align_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        min_t = min(a.shape[-1], b.shape[-1])
        return a[..., :min_t], b[..., :min_t]

    def load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = T.Resample(sr, self.config.sample_rate)(waveform)
        if waveform.shape[-1] < self.config.n_fft:
            pad = self.config.n_fft - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def _pitch_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if librosa is not None:
            wav_np = waveform.squeeze(0).cpu().numpy()
            f0, _, _ = librosa.pyin(
                wav_np,
                fmin=self.config.pitch_fmin,
                fmax=self.config.pitch_fmax,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                frame_length=self.config.n_fft,
                center=False,
            )
            pitch = torch.tensor(f0, dtype=torch.float32).nan_to_num(0.0).unsqueeze(0)
            return pitch

        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate=self.config.sample_rate,
            frame_time=self.config.hop_length / self.config.sample_rate,
        )
        return pitch

    @torch.inference_mode()
    def _ptm_embedding_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        model, processor = self._get_ptm_components()
        wav_np = waveform.squeeze(0).cpu().numpy()

        inputs = processor(
            wav_np,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].to(self.device)

        outputs = model(input_values=input_values, output_hidden_states=True)
        stacked = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, D)
        pooled = stacked.mean(dim=2).permute(1, 0, 2)  # (B, L, D)
        return pooled.squeeze(0).cpu()

    def extract_handcrafted_mfbe_pitch(self, audio_path: str) -> torch.Tensor:
        waveform = self.load_audio(audio_path)

        mel = self.mel(waveform.to(self.device))
        mfbe = torch.log(mel + 1e-6).squeeze(0).cpu()
        pitch = self._pitch_from_waveform(waveform)

        mfbe, pitch = self._align_time(mfbe, pitch)
        handcrafted = self._cmvn(torch.cat([mfbe, pitch], dim=0))
        return handcrafted

    def extract_mode3_inputs(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform = self.load_audio(audio_path)

        mel = self.mel(waveform.to(self.device))
        mfbe = torch.log(mel + 1e-6).squeeze(0).cpu()
        pitch = self._pitch_from_waveform(waveform)
        mfbe, pitch = self._align_time(mfbe, pitch)
        handcrafted = self._cmvn(torch.cat([mfbe, pitch], dim=0))

        embedding = self._ptm_embedding_from_waveform(waveform)
        return embedding, handcrafted
