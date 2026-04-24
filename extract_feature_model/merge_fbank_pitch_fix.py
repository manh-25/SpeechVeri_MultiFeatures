import os
import re
import torch
from tqdm import tqdm

TEST_BASE = r"D:\test_o"
SPLITS = ["full_length", "test_3s", "test_5s", "test_7s", "test_vi"]
FBANK_FILE = "FBank.pt"
PITCH_FILE = "Only_Pitch.pt"
OUT_FILE = "FBank_Pitch.pt"
SAVE_HALF = True
TRIAL_DIR = os.path.join(TEST_BASE, "list_gt")
TRIAL_FILE_MAP = {
    "full_length": "test_list_gt.csv",
    "test_vi": "test_list_gt_vi.csv",
    "test_3s": "test_list_gt_3s.csv",
    "test_5s": "test_list_gt_5s.csv",
    "test_7s": "test_list_gt_7s.csv",
}


def load_trial_paths(split_name):
    trial_name = TRIAL_FILE_MAP.get(split_name)
    if not trial_name:
        return []
    trial_path = os.path.join(TRIAL_DIR, trial_name)
    if not os.path.exists(trial_path):
        return []

    paths = []
    seen = set()
    with open(trial_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().strip('"')
            if not s:
                continue
            parts = [p for p in re.split(r"[,\s]+", s) if p]
            if len(parts) < 3:
                continue
            for p in (parts[1], parts[2]):
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


def to_2d(feat):
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float()
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    return feat


def concat_fbank_pitch(fbank_feat, pitch_feat):
    fbank = to_2d(fbank_feat)
    pitch = to_2d(pitch_feat)
    t = min(fbank.shape[-1], pitch.shape[-1])
    out = torch.cat([fbank[..., :t], pitch[..., :t]], dim=0)
    return out.half() if SAVE_HALF else out


def basename_key(name):
    return os.path.basename(str(name).replace("\\", "/"))


def extract_name_to_feat(obj):
    # Main format in this project: dict where each key is filename and each value is a tensor.
    if isinstance(obj, dict):
        if "features" not in obj and "feature" not in obj:
            if len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values()):
                return {str(k): v for k, v in obj.items()}

        feats = obj.get("features", obj.get("feature", []))
        names = obj.get("filenames", obj.get("filename", []))
        if isinstance(names, list) and len(names) > 0:
            if isinstance(feats, (list, tuple)) and len(feats) == len(names):
                return {str(n): f for n, f in zip(names, feats)}
            if torch.is_tensor(feats) and feats.dim() >= 3 and feats.shape[0] == len(names):
                return {str(names[i]): feats[i] for i in range(len(names))}

    return {}


def merge_split(split_name):
    feat_dir = os.path.join(TEST_BASE, split_name, "features")
    fbank_path = os.path.join(feat_dir, FBANK_FILE)
    pitch_path = os.path.join(feat_dir, PITCH_FILE)
    out_path = os.path.join(feat_dir, OUT_FILE)

    if not os.path.exists(fbank_path) or not os.path.exists(pitch_path):
        print(f"[{split_name}] missing FBank.pt or Only_Pitch.pt")
        return

    fb_map = extract_name_to_feat(torch.load(fbank_path, map_location="cpu"))
    pg_map = extract_name_to_feat(torch.load(pitch_path, map_location="cpu"))

    expected = len(load_trial_paths(split_name))
    print(f"[{split_name}] expected={expected} fbank={len(fb_map)} pitch={len(pg_map)}")

    if len(fb_map) == 0 or len(pg_map) == 0:
        print(f"[{split_name}] parse failed")
        return

    pg_by_base = {basename_key(k): k for k in pg_map.keys()}
    out_feats = []
    out_names = []

    miss = 0
    for fb_name, fb_feat in tqdm(fb_map.items(), desc=f"merge {split_name}"):
        key = basename_key(fb_name)
        pg_name = pg_by_base.get(key)
        if pg_name is None:
            miss += 1
            continue
        out_feats.append(concat_fbank_pitch(fb_feat, pg_map[pg_name]))
        out_names.append(key)

    payload = {
        "features": out_feats,
        "filenames": out_names,
        "speaker_ids": [None] * len(out_feats),
    }
    torch.save(payload, out_path)
    print(f"[{split_name}] saved={out_path} merged={len(out_feats)} miss={miss}")


if __name__ == "__main__":
    for split in SPLITS:
        merge_split(split)
