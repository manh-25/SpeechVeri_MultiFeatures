import importlib
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


BENCH_ROOT = Path(__file__).resolve().parent

ARCH_DEFAULT_EMBEDDING_DIM = {
    "campp": 512,
    "eres2netv2": 192,
    "mfa_conformer": 192,
    "redimnet": 192,
}


def _ensure_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_campp(feature_dim: int):
    cam_dir = BENCH_ROOT / "CAM++"
    _ensure_path(cam_dir)
    mod = _load_module_from_file("benchmark_campp_model", cam_dir / "model.py")
    return mod.CAMPPlus(feat_dim=feature_dim)


def _build_eres2netv2(feature_dim: int):
    eres_dir = BENCH_ROOT / "ERes2netv2"
    _ensure_path(eres_dir)
    mod = _load_module_from_file("benchmark_eres2netv2_model", eres_dir / "model.py")
    return mod.ERes2NetV2(feat_dim=feature_dim)


def _build_mfa_conformer(feature_dim: int):
    mfa_dir = BENCH_ROOT / "MFA_Conformer"
    _ensure_path(mfa_dir)
    mod = _load_module_from_file("benchmark_mfa_conformer_model", mfa_dir / "model.py")
    return mod.conformer(n_mels=feature_dim)


def _build_redimnet(feature_dim: int):
    _ensure_path(BENCH_ROOT)
    redim_mod = importlib.import_module("Redimnet.model")
    model = redim_mod.ReDimNetWrap(F=feature_dim, C=12, feat_type="pt")
    # Use precomputed features directly (skip waveform->spec front-end in wrapper).
    model.spec = nn.Identity()
    return model


class BenchmarkFeatureAdapter(nn.Module):
    def __init__(self, arch_name: str, feature_dim: int, embedding_dim_override: int | None = None):
        super().__init__()
        self.arch_name = str(arch_name).lower()
        self.feature_dim = int(feature_dim)

        if self.arch_name == "campp":
            self.backbone = _build_campp(feature_dim=self.feature_dim)
        elif self.arch_name == "eres2netv2":
            self.backbone = _build_eres2netv2(feature_dim=self.feature_dim)
        elif self.arch_name == "mfa_conformer":
            self.backbone = _build_mfa_conformer(feature_dim=self.feature_dim)
        elif self.arch_name == "redimnet":
            self.backbone = _build_redimnet(feature_dim=self.feature_dim)
        else:
            raise ValueError(f"Unsupported benchmark architecture: {arch_name}")

        native_dim = int(ARCH_DEFAULT_EMBEDDING_DIM[self.arch_name])
        target_dim = int(embedding_dim_override) if embedding_dim_override is not None else native_dim
        self.native_embedding_dim = native_dim
        self.output_embedding_dim = target_dim
        self.proj = nn.Identity() if target_dim == native_dim else nn.Linear(native_dim, target_dim)

    def forward(self, feature=None, **kwargs):
        if feature is None:
            raise KeyError("BenchmarkFeatureAdapter requires 'feature' tensor input.")

        x = feature.float()

        if self.arch_name in {"campp", "eres2netv2"}:
            # Expected by these models: (B, T, F)
            x = x.transpose(1, 2)
        elif self.arch_name == "mfa_conformer":
            # Expected by model code: (B, 1, F, T)
            x = x.unsqueeze(1)
        elif self.arch_name == "redimnet":
            # Wrapper can take (B, F, T) when spec=Identity.
            pass

        emb = self.backbone(x)
        if emb.ndim > 2:
            emb = emb.reshape(emb.size(0), -1)
        emb = self.proj(emb)
        return None, emb


def build_benchmark_get_model(arch_name: str, feature_mode: str, embedding_dim_override: int | None = None):
    # Feature dims aligned with train/config.py convention.
    dim_map = {
        "mfbe_pitch": 81,
        "mfcc_pitch": 41,
        "fbank_pitch": 81,
        "mfbe_only": 80,
        "mfcc_only": 40,
        "fbank_only": 80,
        "pitch_only": 1,
    }
    if feature_mode not in dim_map:
        valid = ", ".join(sorted(dim_map.keys()))
        raise ValueError(f"Unknown feature_mode={feature_mode}. Valid: {valid}")

    feature_dim = int(dim_map[feature_mode])

    def _factory(num_speakers, device="cuda", mode=2, fusion_method="concat", feature_mode=feature_mode, **kwargs):
        model = BenchmarkFeatureAdapter(
            arch_name=arch_name,
            feature_dim=feature_dim,
            embedding_dim_override=embedding_dim_override,
        )
        return model.to(device)

    return _factory
