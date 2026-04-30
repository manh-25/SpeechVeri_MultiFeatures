"""
Dataset loader for Speaker Verification
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import random
import os
import glob
from config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from functools import partial
import copy
import math
import gc
from collections import defaultdict
import torchaudio


def _safe_torch_load(path):
    """Load .pt with mmap when supported to reduce host RAM pressure."""
    resolved_path = os.path.abspath(os.path.normpath(os.fspath(path)))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"PT file not found: {resolved_path}")

    file_size = os.path.getsize(resolved_path)
    if file_size <= 0:
        raise OSError(f"PT file is empty: {resolved_path}")

    errors = []
    use_mmap = os.getenv("SV_TORCH_LOAD_MMAP", "1") == "1"
    if use_mmap:
        try:
            return torch.load(resolved_path, map_location='cpu', mmap=True)
        except (TypeError, OSError, RuntimeError) as ex:
            errors.append(f"mmap load failed: {type(ex).__name__}: {ex}")

    try:
        return torch.load(resolved_path, map_location='cpu')
    except (OSError, RuntimeError, ValueError) as ex:
        errors.append(f"direct load failed: {type(ex).__name__}: {ex}")

    # Some Windows/filesystem combinations fail with path-based reader.
    try:
        with open(resolved_path, "rb") as f:
            return torch.load(f, map_location='cpu')
    except (OSError, RuntimeError, ValueError) as ex:
        errors.append(f"file-object load failed: {type(ex).__name__}: {ex}")

    size_mb = file_size / (1024 ** 2)
    detail = " | ".join(errors)
    raise OSError(
        f"Unable to load PT file: {resolved_path} ({size_mb:.2f} MB). {detail}"
    )


def _filter_tiny_shards(shard_paths, label):
    """Drop suspicious tiny shard files that are very likely corrupted/truncated."""
    min_mb = float(os.getenv("SV_MIN_SHARD_SIZE_MB", "1"))
    if min_mb <= 0:
        return list(shard_paths)

    min_bytes = int(min_mb * 1024 * 1024)
    kept = []
    skipped = []
    for p in shard_paths:
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = -1
        if sz >= min_bytes:
            kept.append(p)
        else:
            skipped.append((p, sz))

    if skipped:
        print(
            f"[WARN] Skip {len(skipped)} tiny {label} shard(s) (< {min_mb:.2f} MB). "
            f"Set SV_MIN_SHARD_SIZE_MB=0 to disable this filter."
        )
        for p, sz in skipped[:10]:
            sz_mb = (sz / (1024 ** 2)) if sz >= 0 else -1.0
            print(f"       - {p} ({sz_mb:.2f} MB)")

    return kept


def _utt_key_candidates(path_text):
    """Build robust matching keys for utterance filenames across pipelines."""
    p = str(path_text).replace("\\", "/").strip()
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    rel = p.lstrip("./")
    keys = []
    for k in (p, rel, base, stem):
        if k and k not in keys:
            keys.append(k)
    return keys


def _build_hc_lookup(hc_speaker_ids, hc_filenames, hc_feature_index):
    """Index HC records by multiple filename keys to support robust PTM-HC join."""
    lookup = defaultdict(list)
    for i, (spk, fn, rec) in enumerate(zip(hc_speaker_ids, hc_filenames, hc_feature_index)):
        item = {
            "speaker_id": spk,
            "filename": str(fn),
            "feature_index": rec,
            "matched": False,
            "id": i,
        }
        for key in _utt_key_candidates(fn):
            lookup[key].append(item)
    return lookup


def _pop_hc_match(hc_lookup, ptm_filename, ptm_speaker_id=None, strict_speaker=True):
    """Find one unused HC record matching PTM filename (and optionally speaker)."""
    fallback = None
    for key in _utt_key_candidates(ptm_filename):
        for item in hc_lookup.get(key, []):
            if item["matched"]:
                continue
            if ptm_speaker_id is not None and strict_speaker and item["speaker_id"] != ptm_speaker_id:
                if fallback is None:
                    fallback = item
                continue
            item["matched"] = True
            return item
    if fallback is not None and not strict_speaker:
        fallback["matched"] = True
        return fallback
    return None


def _resolve_audio_path(utt_id, audio_base_dir=None):
    """Resolve utterance id/filename to an existing audio path."""
    text = str(utt_id).strip().replace("\\", "/")
    env_base_dir = os.getenv("SV_AUDIO_BASE_DIR", "").strip()
    base_dirs = []
    if audio_base_dir:
        base_dirs.append(audio_base_dir)
    if env_base_dir:
        base_dirs.append(env_base_dir)

    candidates = []
    if text:
        candidates.append(text)
        candidates.append(text.lstrip("./"))

    stem, ext = os.path.splitext(os.path.basename(text))
    has_ext = bool(ext)
    exts = [ext] if has_ext else [".wav", ".flac", ".mp3", ".m4a"]

    for base in base_dirs:
        if text:
            candidates.append(os.path.join(base, text))
            candidates.append(os.path.join(base, text.lstrip("./")))
        if stem:
            for cand_ext in exts:
                candidates.append(os.path.join(base, stem + cand_ext))

    seen = set()
    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(cand)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm
    return None


class SpeakerDataset(Dataset):
    """
    Dataset for loading PTM embeddings and handcrafted features.
    Supports 3 modes:
    - Mode 1: PTM embeddings only
    - Mode 2: Handcrafted features only
    - Mode 3: Both PTM and handcrafted features
    """

    def __init__(
        self,
        embedding_data,
        feature_data=None,
        speaker_to_idx=None,
        mode=1,
        use_ptm_on_the_fly=False,
        audio_base_dir=None,
        audio_sample_rate=16000,
        max_audio_seconds=8.0,
        random_audio_crop=False,
    ):
        """
        Args:
            embedding_data: Dict chứa PTM embeddings (đã load từ shard)
            handcrafted_mapping: Dict map từ 'filename' sang 'đường dẫn file .pt'
            mode: 1 (PTM), 2 (Handcrafted), 3 (Both)
        """
        self.mode = mode
        self.embedding_data = embedding_data
        self.feature_data = feature_data
        self.speaker_to_idx = speaker_to_idx or {}
        self.use_ptm_on_the_fly = bool(use_ptm_on_the_fly)
        self.audio_base_dir = audio_base_dir
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_audio_seconds = float(max_audio_seconds) if max_audio_seconds is not None else None
        self.random_audio_crop = bool(random_audio_crop)

        # Build speaker_to_idx if not provided
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(embedding_data["speaker_ids"]))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

        self.num_speakers = len(self.speaker_to_idx)
        self.num_samples = len(embedding_data["speaker_ids"])

        # Per-process shard cache to avoid reloading the same shard repeatedly.
        self._embedding_shard_cache = {}
        self._feature_shard_cache = {}
        self._active_embedding_shard = None
        self._active_feature_shard = None
        self._audio_path_cache = {}
        self._audio_info_cache = {}

    def _set_single_shard_cache(self, cache_dict, active_attr_name, shard_path, shard_data):
        active_path = getattr(self, active_attr_name)
        if active_path != shard_path:
            cache_dict.clear()
            cache_dict[shard_path] = shard_data
            setattr(self, active_attr_name, shard_path)

    def _get_embedding_at(self, idx):
        if "embedding_index" in self.embedding_data:
            rec = self.embedding_data["embedding_index"][idx]
            shard_path = rec["shard_path"]
            local_idx = int(rec["local_idx"])

            if shard_path not in self._embedding_shard_cache:
                shard_data = _safe_torch_load(shard_path)
                self._set_single_shard_cache(
                    self._embedding_shard_cache,
                    "_active_embedding_shard",
                    shard_path,
                    shard_data,
                )
            shard_data = self._embedding_shard_cache[shard_path]
            emb = shard_data["embeddings"][local_idx].float()
            if emb.dim() == 3:
                # Revert to static PTM embedding format by averaging time axis.
                emb = emb.mean(dim=1)
            emb_len = rec.get("length", None)
            if emb_len is None and "lengths" in shard_data:
                emb_len = int(shard_data["lengths"][local_idx])
            return emb, emb_len

        emb = self.embedding_data["embeddings"][idx].float()
        if emb.dim() == 3:
            # Revert to static PTM embedding format by averaging time axis.
            emb = emb.mean(dim=1)
        emb_len = None
        if "lengths" in self.embedding_data and len(self.embedding_data["lengths"]) > idx:
            emb_len = int(self.embedding_data["lengths"][idx])
        return emb, emb_len

    def _get_feature_at(self, idx):
        if self.feature_data is None:
            return None
        if "feature_index" in self.feature_data:
            rec = self.feature_data["feature_index"][idx]
            shard_path = rec["shard_path"]
            local_idx = int(rec["local_idx"])

            if shard_path not in self._feature_shard_cache:
                shard_data = _safe_torch_load(shard_path)
                self._set_single_shard_cache(
                    self._feature_shard_cache,
                    "_active_feature_shard",
                    shard_path,
                    shard_data,
                )
            shard_data = self._feature_shard_cache[shard_path]
            return shard_data["features"][local_idx].float()

        return self.feature_data["features"][idx].float()

    def _get_audio_at(self, idx):
        filenames = self.embedding_data.get("filenames", [])
        if idx >= len(filenames):
            raise RuntimeError(f"Missing filename for on-the-fly PTM at idx={idx}")

        utt_id = str(filenames[idx])
        audio_path = self._audio_path_cache.get(utt_id)
        if audio_path is None:
            audio_path = _resolve_audio_path(utt_id, self.audio_base_dir)
            if audio_path is not None:
                self._audio_path_cache[utt_id] = audio_path
        if audio_path is None:
            raise FileNotFoundError(
                "Cannot resolve audio path for on-the-fly PTM. "
                f"utt_id={utt_id}, audio_base_dir={self.audio_base_dir}, "
                f"SV_AUDIO_BASE_DIR={os.getenv('SV_AUDIO_BASE_DIR', '')}"
            )

        target_sr = int(self.audio_sample_rate)
        max_samples = None
        if self.max_audio_seconds is not None and float(self.max_audio_seconds) > 0:
            max_samples = max(1, int(float(self.max_audio_seconds) * target_sr))

        frame_offset = 0
        num_frames = -1
        src_sr = target_sr
        if max_samples is not None:
            audio_info = self._audio_info_cache.get(audio_path)
            if audio_info is None:
                try:
                    audio_info = torchaudio.info(audio_path)
                except Exception:
                    audio_info = None
                self._audio_info_cache[audio_path] = audio_info

            src_sr = int(getattr(audio_info, "sample_rate", target_sr)) if audio_info is not None else target_sr
            src_num_frames = int(getattr(audio_info, "num_frames", 0)) if audio_info is not None else 0

            # Approximate requested frames in source sample-rate for partial decode.
            src_max_frames = max_samples if src_sr == target_sr else max(1, int(round(max_samples * float(src_sr) / float(target_sr))))
            if src_num_frames > 0 and src_num_frames > src_max_frames:
                if self.random_audio_crop:
                    frame_offset = random.randint(0, src_num_frames - src_max_frames)
                num_frames = src_max_frames

        # Offset theo sample-rate đích để đồng bộ cửa sổ PTM (audio) và HC (feature time axis).
        if int(frame_offset) <= 0:
            offset_samples_target = 0
        elif int(src_sr) == int(target_sr):
            offset_samples_target = int(frame_offset)
        else:
            offset_samples_target = int(round(float(frame_offset) * float(target_sr) / float(src_sr)))
        offset_samples_target = max(0, int(offset_samples_target))

        wav, sr = torchaudio.load(audio_path, frame_offset=int(frame_offset), num_frames=int(num_frames))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 2:
            wav = wav.squeeze(0)

        if int(sr) != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=target_sr)

        if max_samples is not None and wav.numel() > max_samples:
            wav = wav[:max_samples]

        wav = wav.float().contiguous()
        return wav, int(wav.shape[0]), int(offset_samples_target)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        speaker_id = self.embedding_data["speaker_ids"][idx]
        speaker_label = self.speaker_to_idx[speaker_id]
        if "filenames" in self.embedding_data and len(self.embedding_data["filenames"]) > idx:
            wav_filename = str(self.embedding_data["filenames"][idx])
        else:
            wav_filename = str(idx)

        data = {"label": speaker_label, "utt_id": wav_filename}

        # 1. PTM Embedding 
        if self.mode in [1, 3]:
            if self.use_ptm_on_the_fly:
                wav, wav_len, wav_start = self._get_audio_at(idx)
                data["audio"] = wav
                data["audio_length"] = int(wav_len)
                data["audio_start_sample"] = int(wav_start)
            else:
                emb, emb_len = self._get_embedding_at(idx)
                data["embedding"] = emb
                if emb_len is not None:
                    data["embedding_length"] = int(emb_len)
                elif emb.dim() == 3:
                    data["embedding_length"] = int(emb.shape[1])

        # 2. Handcrafted Feature (Giữ nguyên C, T để cho ECAPA-TDNN)
        if self.mode in [2, 3]:
            data["feature"] = self._get_feature_at(idx)

        return data


class SpeakerBalancedBatchSampler(Sampler):
    """Batch sampler theo cấu trúc P speakers x K utterances để tăng ổn định metric-learning."""

    def __init__(
        self,
        indices,
        speaker_ids,
        speaker_to_idx,
        speakers_per_batch,
        utt_per_speaker,
        seed=RANDOM_SEED,
        drop_last=True,
    ):
        self.indices = list(indices)
        self.speaker_ids = speaker_ids
        self.speaker_to_idx = speaker_to_idx
        self.speakers_per_batch = max(2, int(speakers_per_batch))
        self.utt_per_speaker = max(1, int(utt_per_speaker))
        self.batch_size = self.speakers_per_batch * self.utt_per_speaker
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

        self.label_to_indices = {}
        for idx in self.indices:
            spk = self.speaker_ids[idx]
            if spk not in self.speaker_to_idx:
                continue
            label = self.speaker_to_idx[spk]
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels = sorted(self.label_to_indices.keys())
        if len(self.labels) < self.speakers_per_batch:
            raise ValueError(
                f"Không đủ speakers cho batch cân bằng: cần >= {self.speakers_per_batch}, "
                f"nhưng chỉ có {len(self.labels)}"
            )

        if self.drop_last:
            self.num_batches = max(1, len(self.indices) // self.batch_size)
        else:
            self.num_batches = max(1, math.ceil(len(self.indices) / self.batch_size))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        labels = self.labels.copy()

        for _ in range(self.num_batches):
            chosen_labels = rng.sample(labels, self.speakers_per_batch)
            batch = []
            for label in chosen_labels:
                pool = self.label_to_indices[label]
                if len(pool) >= self.utt_per_speaker:
                    picked = rng.sample(pool, self.utt_per_speaker)
                else:
                    picked = [rng.choice(pool) for _ in range(self.utt_per_speaker)]
                batch.extend(picked)

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def collate_fn_general(
    batch,
    mode,
    is_train=True,
    max_frames=350,
    use_ptm_on_the_fly=False,
    audio_sample_rate=16000,
    max_audio_seconds=8.0,
):
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels, "utt_id": [item["utt_id"] for item in batch]}

    if mode in [1,3]:
        if use_ptm_on_the_fly:
            audios = [item["audio"] for item in batch]
            lengths = [int(item.get("audio_length", a.shape[0])) for item, a in zip(batch, audios)]
            base_offsets = [int(item.get("audio_start_sample", 0)) for item in batch]
            target_n = max(lengths)

            if max_audio_seconds is not None:
                max_n = max(1, int(float(max_audio_seconds) * int(audio_sample_rate)))
                target_n = min(target_n, max_n)

            processed = []
            out_lengths = []
            out_offsets = []
            for wav, wav_len, base_offset in zip(audios, lengths, base_offsets):
                wav_len = int(max(1, min(wav_len, int(wav.shape[0]))))
                start = 0
                if wav_len > target_n:
                    if is_train:
                        start = random.randint(0, wav_len - target_n)
                    else:
                        start = 0
                    wav = wav[start:start + target_n]
                    wav_len = target_n
                else:
                    wav = wav[:target_n]
                    if wav.shape[0] < target_n:
                        wav = F.pad(wav, (0, target_n - wav.shape[0]))
                processed.append(wav)
                out_lengths.append(wav_len)
                out_offsets.append(int(base_offset) + int(start))

            output["audio"] = torch.stack(processed)
            output["audio_lengths"] = torch.tensor(out_lengths, dtype=torch.long)
            output["audio_start_samples"] = torch.tensor(out_offsets, dtype=torch.long)
        else:
            embeddings = [item["embedding"] for item in batch]
            emb_example = embeddings[0]

            # Old format: (L, D)
            if emb_example.dim() == 2:
                output["embedding"] = torch.stack(embeddings)

            # Temporal format: (L, T, D) -> collapse to static (L, D)
            elif emb_example.dim() == 3:
                output["embedding"] = torch.stack([emb.mean(dim=1) for emb in embeddings])
            else:
                raise ValueError(f"PTM embedding shape không hợp lệ: {tuple(emb_example.shape)}")

    # Gom Handcrafted Features (Dynamic Padding chiều T bằng Replicate)
    if mode in [2,3]:
        features = [item["feature"] for item in batch]
        processed_features = []
        sync_hc_with_audio = (
            bool(use_ptm_on_the_fly)
            and int(mode) == 3
            and "audio_start_samples" in output
            and "audio" in output
        )
        
        # 1. BƯỚC LỌC CHUẨN: Ép tất cả feature về 2D [C, T] ngay từ đầu
        safe_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(0)  # [T] thành [1, T] (VD: pitch)
            safe_features.append(f)
            
        # 2. XỬ LÝ THEO TRAIN/VAL
        if sync_hc_with_audio:
            # Mode3 on-the-fly: ép HC dùng cùng cửa sổ thời gian với audio PTM.
            hop_hc = max(1, int(round(float(audio_sample_rate) / 100.0)))
            audio_target_n = int(output["audio"].shape[-1])
            if max_frames is not None:
                target_t = max(1, int(max_frames))
            else:
                target_t = max(1, int(math.ceil(float(audio_target_n) / float(hop_hc))))

            audio_start_samples = output["audio_start_samples"].tolist()
            for f, start_sample in zip(safe_features, audio_start_samples):
                t = int(f.shape[-1])
                start_frame = max(0, int(start_sample) // int(hop_hc))
                end_frame = start_frame + target_t

                if start_frame >= t:
                    f = f[:, -1:].clone()
                else:
                    f = f[:, start_frame:end_frame]

                if f.shape[-1] > target_t:
                    f = f[:, :target_t]
                elif f.shape[-1] < target_t:
                    pad_len = target_t - int(f.shape[-1])
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed_features.append(f)

        elif is_train:
            # TỐI ƯU TỐC ĐỘ: Cắt ngẫu nhiên max_frames lúc Train
            for f in safe_features:
                c, t = f.shape
                if t > max_frames:
                    start = random.randint(0, t - max_frames)
                    f = f[:, start:start + max_frames]
                elif t < max_frames:
                    pad_len = max_frames - t
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed_features.append(f)
        else:
            # Lúc Val/Test: Giữ nguyên độ dài, nhưng CẮT BỎ phần thừa nếu quá dài
            max_t = max([f.shape[-1] for f in safe_features])
            if max_frames is not None:
                max_t = min(max_t, int(max_frames))
            max_t = min(max_t, 1000) # <-- SAFETY CAP CHỐNG TRÀN VRAM
            max_t = max(1, int(max_t))
            
            for f in safe_features:
                if f.shape[-1] > max_t:
                    f = f[:, :max_t] # Cắt cụt nếu dài hơn max_t
                
                pad_len = max_t - f.shape[-1]
                if pad_len > 0:
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed_features.append(f)
            
        output["feature"] = torch.stack(processed_features)

    return output


def _order_indices_by_embedding_shard(indices, embedding_index, rng=None, shuffle_shards=False):
    """Reduce random shard hopping by grouping indices by PTM shard."""
    if not indices or not embedding_index:
        return list(indices)

    shard_groups = {}
    for idx in indices:
        rec = embedding_index[idx]
        shard_path = rec["shard_path"]
        local_idx = int(rec.get("local_idx", 0))
        shard_groups.setdefault(shard_path, []).append((local_idx, idx))

    shard_paths = list(shard_groups.keys())
    if shuffle_shards and rng is not None:
        rng.shuffle(shard_paths)
    else:
        shard_paths.sort()

    ordered = []
    for shard_path in shard_paths:
        pairs = shard_groups[shard_path]
        pairs.sort(key=lambda x: x[0])
        ordered.extend([idx for _, idx in pairs])
    return ordered


def load_data(
    embedding_path,
    feature_dir=None,
    mode=1,
    use_ptm_on_the_fly=False,
    preload_all_ram=None,
    preload_ptm_ram=None,
    preload_hc_ram=None,
):
    embedding_data = {
        "speaker_ids": [],
        "filenames": [],
        "embeddings": [],
        "lengths": [],
        "embedding_index": [],
    }
    feature_data = {
        "features": [],
        "feature_index": [],
    }

    hc_speaker_ids = []
    hc_filenames = []

    use_aligned_fastpath = (
        mode == 3
        and feature_dir is not None
        and os.path.isdir(feature_dir)
        and os.getenv("SV_ASSUME_ALIGNED_SHARDS", "1") == "1"
    )
    allow_unaligned_mode3_fullscan = os.getenv("SV_ALLOW_UNALIGNED_MODE3_FULLSCAN", "0") == "1"
    aligned_ptm_samples_per_shard = int(os.getenv("SV_ALIGNED_PTM_SAMPLES_PER_SHARD", "5000"))
    aligned_ptm_total_samples = int(os.getenv("SV_ALIGNED_PTM_TOTAL_SAMPLES", "0"))
    aligned_use_ptm_meta_counts = os.getenv("SV_ALIGNED_USE_PTM_META_COUNTS", "1") == "1"
    mode3_match_by_filename = int(mode) == 3 and os.getenv("SV_MODE3_MATCH_BY_FILENAME", "1") == "1"
    mode3_strict_speaker_match = os.getenv("SV_MODE3_STRICT_SPK_MATCH", "1") == "1"
    use_ptm_on_the_fly = bool(use_ptm_on_the_fly)

    # Default behavior: preload everything to RAM for all modes.
    # Can be overridden per-run via function args (used by train-all loops).
    if preload_all_ram is None:
        preload_all_ram = os.getenv("SV_PRELOAD_ALL_RAM", "1") == "1"
    else:
        preload_all_ram = bool(preload_all_ram)

    if preload_ptm_ram is None:
        preload_ptm_ram = os.getenv("SV_PRELOAD_PTM_RAM", "1" if preload_all_ram else "0") == "1"
    else:
        preload_ptm_ram = bool(preload_ptm_ram)

    if preload_hc_ram is None:
        preload_hc_ram = os.getenv("SV_PRELOAD_HC_RAM", "1" if preload_all_ram else "0") == "1"
    else:
        preload_hc_ram = bool(preload_hc_ram)

    preload_ptm_ram = preload_ptm_ram and (not use_ptm_on_the_fly) and int(mode) in [1, 3]
    preload_hc_ram = preload_hc_ram and int(mode) in [2, 3]

    # Backward compatibility for older mode-2-only knob.
    if int(mode) == 2 and not preload_hc_ram and os.getenv("SV_MODE2_PRELOAD_HC_RAM", "0") == "1":
        preload_hc_ram = True

    if mode3_match_by_filename and use_aligned_fastpath:
        use_aligned_fastpath = False
        print("ℹ Mode3 join mode: filename-based (aligned fast-path disabled).")

    if preload_ptm_ram and use_aligned_fastpath:
        use_aligned_fastpath = False
        print("ℹ Full-RAM PTM preload bật: chuyển sang full metadata scan (disable aligned fast-path).")

    hc_shard_counts = []
    hc_shards = []
    hc_global_idx = 0
    ptm_global_idx = 0

    # Fast path: when PTM/HC are already aligned in order, use HC metadata only.
    if mode in [2, 3]:
        if feature_dir is None or not os.path.isdir(feature_dir):
            raise ValueError(f"Mode {mode} yêu cầu feature_dir là đường dẫn thư mục chứa Shards")

        print(f"🔍 Đang nạp metadata Handcrafted từ: {feature_dir}...")
        hc_shards = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        hc_shards = _filter_tiny_shards(hc_shards, "HC")
        if not hc_shards:
            raise FileNotFoundError(f"Không tìm thấy HC shards trong {feature_dir}")

        valid_hc_shards = []
        for shard in hc_shards:
            try:
                shard_data = _safe_torch_load(shard)
            except Exception as ex:
                print(f"[WARN] Skip unreadable HC shard: {shard} | {type(ex).__name__}: {ex}")
                continue

            valid_hc_shards.append(shard)
            shard_features = shard_data["features"]
            shard_count = int(shard_features.shape[0]) if torch.is_tensor(shard_features) else len(shard_features)
            hc_shard_counts.append(shard_count)

            if preload_hc_ram:
                if torch.is_tensor(shard_features):
                    for i in range(shard_count):
                        feature_data["features"].append(shard_features[i].float().cpu())
                else:
                    for x in shard_features:
                        feature_data["features"].append(x.float().cpu())

            for i in range(shard_count):
                feature_data["feature_index"].append(
                    {"shard_path": shard, "local_idx": i, "global_idx": hc_global_idx}
                )
                hc_global_idx += 1

            # Keep HC metadata for later split/join.
            hc_speaker_ids.extend(shard_data["speaker_ids"])
            hc_filenames.extend(shard_data["filenames"])

            del shard_data
            gc.collect()

        hc_shards = valid_hc_shards
        if not hc_shard_counts:
            raise RuntimeError("Không load được HC shard nào hợp lệ sau khi lọc/cảnh báo lỗi.")

        if preload_hc_ram and len(feature_data["features"]) == 0:
            raise RuntimeError("HC preload bật nhưng không có HC feature nào hợp lệ để nạp RAM.")

        hc_total_samples = (
            len(feature_data["features"])
            if preload_hc_ram
            else len(feature_data["feature_index"])
        )

        if preload_hc_ram:
            print(
                f"✅ HC preload RAM: đã nạp {len(hc_shards)} HC shards vào RAM. "
                f"Tổng sample HC: {hc_total_samples}"
            )
        else:
            feature_data.pop("features", None)

        print(f"✅ Đã nạp xong metadata {len(hc_shards)} HC shards. Tổng số sample HC: {hc_total_samples}")

        if len(hc_filenames) != len(hc_speaker_ids):
            hc_filenames = [str(i) for i in range(len(hc_speaker_ids))]

    if mode in [1, 3] and use_ptm_on_the_fly:
        if int(mode) == 1:
            raise NotImplementedError(
                "Mode 1 on-the-fly chưa được hỗ trợ trong load_data hiện tại. "
                "Hãy dùng Mode 3 hoặc tắt use_ptm_on_the_fly."
            )
        if int(mode) == 3:
            if len(hc_speaker_ids) == 0 or len(hc_filenames) == 0:
                raise RuntimeError("Mode3 on-the-fly yêu cầu HC metadata khả dụng để build sample list.")

            embedding_data["speaker_ids"] = list(hc_speaker_ids)
            embedding_data["filenames"] = list(hc_filenames)
            embedding_data.pop("embedding_index", None)
            embedding_data.pop("embeddings", None)
            embedding_data.pop("lengths", None)

            print("📦 PTM input source: on-the-fly audio")
            print(
                "✅ Mode3 on-the-fly: dùng HC metadata làm sample index "
                f"({len(embedding_data['speaker_ids'])} samples), bỏ qua PTM precomputed shards."
            )

    # 1. LOAD PTM EMBEDDINGS (Hỗ trợ nhiều file Shard)
    elif mode in [1, 3]:
        if os.path.isdir(embedding_path):
            print("📦 PTM input source: precomputed shards directory")
            print(f"🔍 Đang quét các file shard PTM tại: {embedding_path}...")
            shard_files = sorted(glob.glob(os.path.join(embedding_path, "*.pt")))
            shard_files = _filter_tiny_shards(shard_files, "PTM")
            
            if not shard_files:
                raise FileNotFoundError(f"Không tìm thấy file .pt nào trong thư mục {embedding_path}")

            if mode == 3 and hc_shards and len(shard_files) != len(hc_shards):
                msg = (
                    f"Mode 3 yêu cầu PTM/HC đồng bộ shard khi dùng aligned fast-path, "
                    f"nhưng hiện tại PTM={len(shard_files)} shards, HC={len(hc_shards)} shards. "
                    f"Điều này thường xảy ra khi PTM chưa extract đủ sample."
                )
                if use_aligned_fastpath and not allow_unaligned_mode3_fullscan:
                    print("⚠ " + msg)
                    print(
                        "⚠ Dùng aligned-prefix mode: giữ thứ tự sample, lấy prefix HC theo số sample PTM hiện có. "
                        f"Mặc định giả định {aligned_ptm_samples_per_shard} sample/PTM shard "
                        "(set SV_ALIGNED_PTM_SAMPLES_PER_SHARD để đổi)."
                    )
                if use_aligned_fastpath and allow_unaligned_mode3_fullscan:
                    print("⚠ " + msg)
                    print("⚠ SV_ALLOW_UNALIGNED_MODE3_FULLSCAN=1 -> fallback sang full metadata scan PTM (có thể tốn RAM lớn).")
                    use_aligned_fastpath = False
            
            ptm_speaker_ids = []
            ptm_filenames = []
            ptm_embedding_index = []

            if use_aligned_fastpath:
                # Build PTM index by mirrored shard order/count from HC without loading giant PTM tensors.
                if len(shard_files) == len(hc_shard_counts):
                    for shard, shard_count in zip(shard_files, hc_shard_counts):
                        for i in range(int(shard_count)):
                            ptm_embedding_index.append(
                                {"shard_path": shard, "local_idx": i, "global_idx": ptm_global_idx}
                            )
                            ptm_global_idx += 1
                else:
                    if aligned_ptm_samples_per_shard <= 0:
                        raise RuntimeError(
                            "SV_ALIGNED_PTM_SAMPLES_PER_SHARD phải > 0 khi dùng aligned-prefix mode."
                        )

                    hc_total = len(feature_data["feature_index"])
                    if aligned_use_ptm_meta_counts:
                        ptm_counts = []
                        valid_shard_files = []
                        for shard in shard_files:
                            try:
                                shard_data = _safe_torch_load(shard)
                            except Exception as ex:
                                print(f"[WARN] Skip unreadable PTM shard: {shard} | {type(ex).__name__}: {ex}")
                                continue
                            shard_embeddings = shard_data["embeddings"]
                            shard_count = int(shard_embeddings.shape[0]) if torch.is_tensor(shard_embeddings) else len(shard_embeddings)
                            ptm_counts.append(int(shard_count))
                            valid_shard_files.append(shard)
                            del shard_data
                            gc.collect()
                        shard_files = valid_shard_files
                        if not shard_files:
                            raise RuntimeError("Không load được PTM shard nào hợp lệ sau khi lọc/cảnh báo lỗi.")
                        ptm_capacity = int(sum(ptm_counts))
                    else:
                        ptm_counts = [int(aligned_ptm_samples_per_shard)] * len(shard_files)
                        ptm_capacity = len(shard_files) * aligned_ptm_samples_per_shard

                    if aligned_ptm_total_samples > 0:
                        ptm_capacity = min(ptm_capacity, aligned_ptm_total_samples)
                    ptm_total = min(hc_total, ptm_capacity)
                    if ptm_total <= 0:
                        raise RuntimeError("Không có sample PTM hợp lệ để build aligned-prefix mode.")

                    remaining = int(ptm_total)
                    for shard, shard_count in zip(shard_files, ptm_counts):
                        take_n = min(int(shard_count), remaining)
                        for i in range(int(take_n)):
                            ptm_embedding_index.append(
                                {"shard_path": shard, "local_idx": i, "global_idx": ptm_global_idx}
                            )
                            ptm_global_idx += 1
                        remaining -= int(take_n)
                        if remaining <= 0:
                            break

                    hc_speaker_ids = hc_speaker_ids[:ptm_total]
                    hc_filenames = hc_filenames[:ptm_total]
                    feature_data["feature_index"] = feature_data["feature_index"][:ptm_total]
                    if preload_hc_ram and "features" in feature_data:
                        feature_data["features"] = feature_data["features"][:ptm_total]

                    print(
                        f"✅ Aligned-prefix Mode3: PTM samples={ptm_total} | HC prefix samples={ptm_total} "
                        f"| PTM shards used={len(shard_files)}"
                    )
            else:
                valid_shard_files = []
                for shard in shard_files:
                    try:
                        shard_data = _safe_torch_load(shard)
                    except Exception as ex:
                        print(f"[WARN] Skip unreadable PTM shard: {shard} | {type(ex).__name__}: {ex}")
                        continue

                    valid_shard_files.append(shard)
                    shard_speakers = list(shard_data["speaker_ids"])
                    shard_filenames = list(shard_data.get("filenames", []))
                    if len(shard_filenames) != len(shard_speakers):
                        shard_filenames = [str(i) for i in range(len(shard_speakers))]

                    shard_embeddings = shard_data["embeddings"]
                    if torch.is_tensor(shard_embeddings):
                        shard_count = int(shard_embeddings.shape[0])
                    else:
                        shard_count = len(shard_embeddings)

                    shard_lengths = shard_data.get("lengths", None)
                    if shard_lengths is not None:
                        if torch.is_tensor(shard_lengths):
                            shard_lengths = shard_lengths.long().tolist()
                        else:
                            shard_lengths = [int(x) for x in shard_lengths]

                    for i in range(shard_count):
                        ptm_speaker_ids.append(shard_speakers[i])
                        ptm_filenames.append(str(shard_filenames[i]))
                        rec = {"shard_path": shard, "local_idx": i, "global_idx": ptm_global_idx}
                        ptm_global_idx += 1
                        if shard_lengths is not None and i < len(shard_lengths):
                            rec["length"] = int(shard_lengths[i])
                        ptm_embedding_index.append(rec)
                        if preload_ptm_ram:
                            ptm_item = shard_embeddings[i]
                            if torch.is_tensor(ptm_item):
                                embedding_data["embeddings"].append(ptm_item.float().cpu())
                            else:
                                embedding_data["embeddings"].append(torch.tensor(ptm_item, dtype=torch.float32))
                            if shard_lengths is not None and i < len(shard_lengths):
                                embedding_data["lengths"].append(int(shard_lengths[i]))

                    del shard_data
                    gc.collect()

                if not valid_shard_files:
                    raise RuntimeError("Không load được PTM shard nào hợp lệ sau khi lọc/cảnh báo lỗi.")
                shard_files = valid_shard_files

            if int(mode) == 1:
                embedding_data["speaker_ids"] = ptm_speaker_ids
                embedding_data["filenames"] = ptm_filenames
                embedding_data["embedding_index"] = ptm_embedding_index
            elif int(mode) == 3:
                if mode3_match_by_filename:
                    if not feature_data.get("feature_index"):
                        raise RuntimeError("Mode3 filename matching yêu cầu feature_index khả dụng.")
                    hc_lookup = _build_hc_lookup(hc_speaker_ids, hc_filenames, feature_data["feature_index"])

                    matched_speakers = []
                    matched_filenames = []
                    matched_ptm_index = []
                    matched_hc_index = []

                    unmatched_ptm = 0
                    speaker_mismatch = 0

                    for spk, fn, rec in zip(ptm_speaker_ids, ptm_filenames, ptm_embedding_index):
                        hc_item = _pop_hc_match(
                            hc_lookup,
                            ptm_filename=fn,
                            ptm_speaker_id=spk,
                            strict_speaker=mode3_strict_speaker_match,
                        )
                        if hc_item is None:
                            unmatched_ptm += 1
                            continue
                        if hc_item["speaker_id"] != spk:
                            speaker_mismatch += 1

                        matched_speakers.append(spk)
                        matched_filenames.append(str(fn))
                        matched_ptm_index.append(rec)
                        matched_hc_index.append(hc_item["feature_index"])

                    if len(matched_ptm_index) == 0:
                        raise RuntimeError(
                            "Mode3 filename matching produced 0 matched samples. "
                            "Kiểm tra định dạng filename giữa PTM và HC shards."
                        )

                    embedding_data["speaker_ids"] = matched_speakers
                    embedding_data["filenames"] = matched_filenames
                    embedding_data["embedding_index"] = matched_ptm_index
                    feature_data["feature_index"] = matched_hc_index

                    if preload_ptm_ram and "embeddings" in embedding_data:
                        reordered_emb = []
                        reordered_len = []
                        have_len = len(embedding_data.get("lengths", [])) == len(embedding_data.get("embeddings", []))
                        for rec in matched_ptm_index:
                            gidx = int(rec.get("global_idx", -1))
                            if gidx < 0 or gidx >= len(embedding_data["embeddings"]):
                                raise RuntimeError("PTM preload map lỗi: global_idx vượt phạm vi.")
                            reordered_emb.append(embedding_data["embeddings"][gidx])
                            if have_len:
                                reordered_len.append(int(embedding_data["lengths"][gidx]))
                        embedding_data["embeddings"] = reordered_emb
                        if have_len:
                            embedding_data["lengths"] = reordered_len

                    if preload_hc_ram and "features" in feature_data:
                        reordered_feat = []
                        for rec in matched_hc_index:
                            gidx = int(rec.get("global_idx", -1))
                            if gidx < 0 or gidx >= len(feature_data["features"]):
                                raise RuntimeError("HC preload map lỗi: global_idx vượt phạm vi.")
                            reordered_feat.append(feature_data["features"][gidx])
                        feature_data["features"] = reordered_feat

                    print(
                        "✅ Mode3 filename-join: "
                        f"matched={len(matched_ptm_index)} | unmatched_ptm={unmatched_ptm} "
                        f"| speaker_mismatch={speaker_mismatch}"
                    )
                else:
                    # Legacy fallback: rely on positional alignment.
                    n = min(len(ptm_embedding_index), len(feature_data.get("feature_index", [])))
                    embedding_data["embedding_index"] = ptm_embedding_index[:n]
                    feature_data["feature_index"] = feature_data["feature_index"][:n]
                    embedding_data["speaker_ids"] = hc_speaker_ids[:n]
                    embedding_data["filenames"] = hc_filenames[:n]
                    if preload_ptm_ram and "embeddings" in embedding_data:
                        embedding_data["embeddings"] = embedding_data["embeddings"][:n]
                        if "lengths" in embedding_data:
                            embedding_data["lengths"] = embedding_data["lengths"][:n]
                    if preload_hc_ram and "features" in feature_data:
                        feature_data["features"] = feature_data["features"][:n]

            if not preload_ptm_ram:
                embedding_data.pop("embeddings", None)
                embedding_data.pop("lengths", None)
            print(f"✅ Đã load gộp {len(shard_files)} file shards. Tổng số sample PTM: {len(embedding_data['embedding_index'])}")
        else:
            print("📦 PTM input source: precomputed single .pt file")
            # Fallback nếu truyền vào đường dẫn của 1 file duy nhất
            raw_data = _safe_torch_load(embedding_path)
            embedding_data["speaker_ids"] = list(raw_data.get("speaker_ids", []))
            embedding_data["filenames"] = list(raw_data.get("filenames", []))

            raw_embeddings = raw_data.get("embeddings", None)
            if raw_embeddings is None:
                raise ValueError(f"File PTM không có key 'embeddings': {embedding_path}")
            if len(embedding_data["filenames"]) != len(embedding_data["speaker_ids"]):
                embedding_data["filenames"] = [str(i) for i in range(len(embedding_data["speaker_ids"]))]

            raw_lengths = raw_data.get("lengths", None)
            if raw_lengths is not None:
                if torch.is_tensor(raw_lengths):
                    raw_lengths = raw_lengths.long().tolist()
                else:
                    raw_lengths = [int(x) for x in raw_lengths]

            if torch.is_tensor(raw_embeddings):
                n = int(raw_embeddings.shape[0])
            else:
                n = len(raw_embeddings)

            for i in range(n):
                rec = {"shard_path": embedding_path, "local_idx": i, "global_idx": ptm_global_idx}
                ptm_global_idx += 1
                if raw_lengths is not None and i < len(raw_lengths):
                    rec["length"] = int(raw_lengths[i])
                embedding_data["embedding_index"].append(rec)
                if preload_ptm_ram:
                    ptm_item = raw_embeddings[i]
                    if torch.is_tensor(ptm_item):
                        embedding_data["embeddings"].append(ptm_item.float().cpu())
                    else:
                        embedding_data["embeddings"].append(torch.tensor(ptm_item, dtype=torch.float32))
                    if raw_lengths is not None and i < len(raw_lengths):
                        embedding_data["lengths"].append(int(raw_lengths[i]))

            if not preload_ptm_ram:
                embedding_data.pop("embeddings", None)
                embedding_data.pop("lengths", None)

            del raw_data
            gc.collect()

            print(f"✅ Đã load 1 file PTM tổng. Tổng số sample PTM: {len(embedding_data['speaker_ids'])}")

    if int(mode) == 2:
        embedding_data["speaker_ids"] = list(hc_speaker_ids)
        embedding_data["filenames"] = list(hc_filenames)

    if preload_ptm_ram and int(mode) in [1, 3] and not use_ptm_on_the_fly:
        if len(embedding_data.get("embeddings", [])) != len(embedding_data.get("speaker_ids", [])):
            raise RuntimeError(
                "PTM preload RAM không đồng bộ kích thước: "
                f"embeddings={len(embedding_data.get('embeddings', []))}, "
                f"speaker_ids={len(embedding_data.get('speaker_ids', []))}"
            )
        embedding_data.pop("embedding_index", None)
        print(f"✅ PTM preload RAM active: {len(embedding_data['embeddings'])} samples in-memory.")

    if preload_hc_ram and int(mode) in [2, 3]:
        if len(feature_data.get("features", [])) != len(embedding_data.get("speaker_ids", [])):
            raise RuntimeError(
                "HC preload RAM không đồng bộ kích thước: "
                f"features={len(feature_data.get('features', []))}, "
                f"speaker_ids={len(embedding_data.get('speaker_ids', []))}"
            )
        feature_data.pop("feature_index", None)
        print(f"✅ HC preload RAM active: {len(feature_data['features'])} samples in-memory.")

    if (
        use_aligned_fastpath
        and int(mode) == 3
        and "embedding_index" in embedding_data
        and "feature_index" in feature_data
    ):
        # Defensive check: PTM/HC index lengths must match exactly.
        if len(embedding_data["embedding_index"]) != len(feature_data["feature_index"]):
            raise RuntimeError(
                f"PTM index ({len(embedding_data['embedding_index'])}) != HC index ({len(feature_data['feature_index'])})."
            )

    unique_speakers = sorted(set(embedding_data["speaker_ids"]))
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

    return embedding_data, feature_data, speaker_to_idx


def create_train_val_loaders(
    embedding_path,
    feature_path,
    mode,
    batch_size,
    num_workers=0,
    use_speaker_balanced=False,
    speakers_per_batch=16,
    utt_per_speaker=4,
    max_frames=350,
    val_batch_size=32,
    pin_memory=None,
    persistent_workers=None,
    prefetch_factor=2,
    use_ptm_on_the_fly=False,
    audio_base_dir=None,
    audio_sample_rate=16000,
    max_audio_seconds=8.0,
    train_subset_fraction=1.0,
    max_train_samples=0,
    preload_all_ram=None,
    preload_ptm_ram=None,
    preload_hc_ram=None,
):
    # Nạp dữ liệu
    embedding_data, feature_data, speaker_to_idx = load_data(
        embedding_path,
        feature_path,
        mode,
        use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
        preload_all_ram=preload_all_ram,
        preload_ptm_ram=preload_ptm_ram,
        preload_hc_ram=preload_hc_ram,
    )
    
    # --- LỌC INDEX THEO SPEAKER ID (TRÁNH DATA LEAKAGE) ---
    speaker_ids = embedding_data["speaker_ids"]
    unique_speakers = sorted(set(speaker_ids))
    
    # Xáo trộn danh sách người nói (cố định seed để dễ tái lập)
    shuffled_speakers = list(unique_speakers)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shuffled_speakers)
    
    # Cắt 85% NGƯỜI NÓI cho Train, 15% NGƯỜI NÓI cho Val
    num_train_spk = int(len(shuffled_speakers) * TRAIN_RATIO)
    train_speakers = set(shuffled_speakers[:num_train_spk])
    val_speakers = set(shuffled_speakers[num_train_spk:])
    
    train_indices = []
    val_indices = []
    
    # Phân loại từng mẫu âm thanh về đúng tập dựa trên speaker
    for i, spk in enumerate(speaker_ids):
        if spk in train_speakers:
            train_indices.append(i)
        else:
            val_indices.append(i)
            
    print(f"\n🎤 CHIA DỮ LIỆU THEO OPEN-SET (Unseen Speakers):")
    print(f"   - Tập Train: {len(train_speakers)} speakers ({len(train_indices)} samples)")
    print(f"   - Tập Val:   {len(val_speakers)} speakers ({len(val_indices)} samples)\n")

    # Tạo label mapping độc lập cho train/val (open-set đúng nghĩa)
    train_speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(train_speakers))}
    val_speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(val_speakers))}

    mode3_shard_locality = (
        int(mode) == 3
        and os.getenv("SV_MODE3_SHARD_LOCALITY", "1") == "1"
        and "embedding_index" in embedding_data
        and len(embedding_data["embedding_index"]) > 0
    )

    # Xáo trộn có seed để tái lập tuyệt đối giữa các lần chạy
    if mode3_shard_locality:
        train_indices = _order_indices_by_embedding_shard(
            train_indices,
            embedding_data["embedding_index"],
            rng=rng,
            shuffle_shards=True,
        )
        val_indices = _order_indices_by_embedding_shard(
            val_indices,
            embedding_data["embedding_index"],
            rng=None,
            shuffle_shards=False,
        )
        print("[IO-OPT] Mode3 shard-locality enabled để giảm thời gian đọc PTM shard.")
    else:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

    # Optional fast-training controls for very large on-the-fly runs.
    train_subset_fraction = float(train_subset_fraction)
    if train_subset_fraction > 0.0 and train_subset_fraction < 1.0:
        keep_n = max(1, int(len(train_indices) * train_subset_fraction))
        if keep_n < len(train_indices):
            train_indices = train_indices[:keep_n]
            print(
                f"[FAST-EPOCH] Train subset fraction applied: {train_subset_fraction:.3f} "
                f"-> {len(train_indices)} train samples"
            )

    max_train_samples = int(max_train_samples)
    if max_train_samples > 0 and len(train_indices) > max_train_samples:
        train_indices = train_indices[:max_train_samples]
        print(f"[FAST-EPOCH] Max train samples applied: {len(train_indices)}")

    full_dataset = SpeakerDataset(
        embedding_data,
        feature_data,
        speaker_to_idx,
        mode,
        use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
        audio_base_dir=audio_base_dir,
        audio_sample_rate=int(audio_sample_rate),
        max_audio_seconds=float(max_audio_seconds),
        random_audio_crop=False,
    )

    train_dataset = copy.copy(full_dataset)
    train_dataset.speaker_to_idx = train_speaker_to_idx
    train_dataset.num_speakers = len(train_speaker_to_idx)
    train_dataset.random_audio_crop = True

    val_dataset = copy.copy(full_dataset)
    val_dataset.speaker_to_idx = val_speaker_to_idx
    val_dataset.num_speakers = len(val_speaker_to_idx)
    val_dataset.random_audio_crop = False

    num_workers = max(0, int(num_workers))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if use_speaker_balanced:
        batch_sampler = SpeakerBalancedBatchSampler(
            indices=train_indices,
            speaker_ids=speaker_ids,
            speaker_to_idx=train_speaker_to_idx,
            speakers_per_batch=speakers_per_batch,
            utt_per_speaker=utt_per_speaker,
            seed=RANDOM_SEED,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                collate_fn_general,
                mode=mode,
                is_train=True,
                max_frames=max_frames,
                use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
                audio_sample_rate=int(audio_sample_rate),
                max_audio_seconds=float(max_audio_seconds),
            ),
            **loader_kwargs,
        )
    else:
        train_shuffle = not mode3_shard_locality
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size, shuffle=train_shuffle,
            collate_fn=partial(
                collate_fn_general,
                mode=mode,
                is_train=True,
                max_frames=max_frames,
                use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
                audio_sample_rate=int(audio_sample_rate),
                max_audio_seconds=float(max_audio_seconds),
            ),
            **loader_kwargs,
        )
    
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=int(val_batch_size), shuffle=False,
        collate_fn=partial(
            collate_fn_general,
            mode=mode,
            is_train=False,
            max_frames=max_frames,
            use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
            audio_sample_rate=int(audio_sample_rate),
            max_audio_seconds=float(max_audio_seconds),
        ),
        **loader_kwargs,
    )

    return train_loader, val_loader, train_speaker_to_idx, len(train_speaker_to_idx)

def create_test_loader(
    test_embedding_path,
    test_feature_path=None,
    mode=1,
    batch_size=64,
    num_workers=0,
    pin_memory=None,
    persistent_workers=None,
    prefetch_factor=2,
    use_ptm_on_the_fly=False,
    audio_base_dir=None,
    audio_sample_rate=16000,
    max_audio_seconds=8.0,
):
    """CHỈ DÙNG LÚC TEST: Nhận data của Unseen Speakers và ném tất cả vào 1 Loader"""
    embedding_data, feature_data, speaker_to_idx = load_data(
        test_embedding_path,
        test_feature_path,
        mode,
        use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
    )
    test_dataset = SpeakerDataset(
        embedding_data,
        feature_data,
        speaker_to_idx,
        mode,
        use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
        audio_base_dir=audio_base_dir,
        audio_sample_rate=int(audio_sample_rate),
    )
    
    num_workers = max(0, int(num_workers))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        # Thay lambda bằng partial
        collate_fn=partial(
            collate_fn_general,
            mode=mode,
            is_train=False,
            use_ptm_on_the_fly=bool(use_ptm_on_the_fly),
            audio_sample_rate=int(audio_sample_rate),
            max_audio_seconds=float(max_audio_seconds),
        ),
        **loader_kwargs,
    )
    return test_loader, len(speaker_to_idx)