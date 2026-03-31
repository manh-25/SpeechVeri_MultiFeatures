from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F


class SpeakerRegistry:
    def __init__(self, registry_path: str) -> None:
        self.registry_path = registry_path
        self.data = self._load()

    def _load(self) -> Dict[str, Dict[str, List[torch.Tensor] | int]]:
        if not os.path.exists(self.registry_path):
            return {"speakers": {}}
        payload = torch.load(self.registry_path, map_location="cpu")
        if "speakers" not in payload:
            payload = {"speakers": {}}
        return payload

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        torch.save(self.data, self.registry_path)

    def add_embedding(self, speaker_id: str, embedding: torch.Tensor) -> None:
        speakers = self.data.setdefault("speakers", {})
        if speaker_id not in speakers:
            speakers[speaker_id] = {"embeddings": [], "num_samples": 0}

        speakers[speaker_id]["embeddings"].append(embedding.float().cpu())
        speakers[speaker_id]["num_samples"] = int(speakers[speaker_id]["num_samples"]) + 1

    def list_speakers(self) -> List[str]:
        return sorted(self.data.get("speakers", {}).keys())

    def num_samples(self, speaker_id: str) -> int:
        info = self.data.get("speakers", {}).get(speaker_id, {})
        return int(info.get("num_samples", 0))

    def centroids(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for speaker_id, info in self.data.get("speakers", {}).items():
            embs = info.get("embeddings", [])
            if not embs:
                continue
            mat = torch.stack([e.float() for e in embs], dim=0)
            center = mat.mean(dim=0)
            out[speaker_id] = F.normalize(center, p=2, dim=0)
        return out
