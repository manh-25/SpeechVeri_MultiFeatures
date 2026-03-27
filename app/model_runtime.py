from __future__ import annotations

import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.feature_extractor import DemoFeatureExtractor

_MODEL_PATH = TRAIN_DIR / "model.py"
_MODEL_SPEC = importlib.util.spec_from_file_location("sv_train_model", _MODEL_PATH)
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise ImportError(f"Cannot load model module from {_MODEL_PATH}")
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)
get_model = _MODEL_MODULE.get_model


@dataclass
class RuntimeConfig:
    mode: int = 3
    fusion_method: str = "concat"
    feature_mode: str = "mfbe_pitch"


class SpeakerDemoRuntime:
    def __init__(self, checkpoint_path: str, device: str = "cpu", config: RuntimeConfig | None = None) -> None:
        self.device = torch.device(device)
        self.config = config or RuntimeConfig()
        self.extractor = DemoFeatureExtractor(device=str(self.device))

        self.model = get_model(
            num_speakers=1,
            device=str(self.device),
            mode=self.config.mode,
            fusion_method=self.config.fusion_method,
            feature_mode=self.config.feature_mode,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def embedding_from_audio(self, audio_path: str) -> torch.Tensor:
        embedding, handcrafted = self.extractor.extract_mode3_inputs(audio_path)
        _, emb = self.model(
            embedding=embedding.unsqueeze(0).to(self.device),
            feature=handcrafted.unsqueeze(0).to(self.device),
        )
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu()

    @torch.inference_mode()
    def compare_two_audio(self, audio_a: str, audio_b: str) -> float:
        emb_a = self.embedding_from_audio(audio_a)
        emb_b = self.embedding_from_audio(audio_b)
        score = torch.sum(emb_a * emb_b).item()
        return float(score)

    @staticmethod
    def identify(embedding: torch.Tensor, centroids: Dict[str, torch.Tensor]) -> Tuple[str | None, Dict[str, float]]:
        if not centroids:
            return None, {}

        scores: Dict[str, float] = {}
        for speaker_id, centroid in centroids.items():
            scores[speaker_id] = float(torch.sum(embedding * centroid).item())

        best_speaker = max(scores.items(), key=lambda x: x[1])[0]
        return best_speaker, scores
