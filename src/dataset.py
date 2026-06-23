"""Datasets for the CIL monocular-depth data (RGB .png + depth .npy pairs).

Two families live here:

* ``TrainDataset`` / ``TestDataset`` -- the original **per-image** datasets for
  the single-frame Kaggle CIL pipeline.
* ``VideoTrainDataset`` / ``VideoTestDataset`` -- **per-clip** datasets for the
  video-native model: each item is ``T`` consecutive frames (and, for training,
  their depths). These are config-driven so they adapt to whatever on-disk
  layout the video depth data ends up in (see ``VideoTrainDataset``).
"""

import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """Paired RGB image and ground-truth depth map for training."""

    def __init__(self, train_dir):
        self.train_dir = Path(train_dir)

        # Sorted globs keep image[i] aligned with its depth[i].
        self.image_paths = sorted(list(self.train_dir.glob('*.png')))
        self.depth_paths = sorted(list(self.train_dir.glob('*.npy')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Raw uint8 RGB in [0, 255]; preprocessing/normalization happens later.
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)

        # convert HWC to CHW
        image = image.permute(2, 0, 1)

        sample = {'image': image}

        # Depth stored as a (H, W) numpy array; add a channel dim -> (1, H, W).
        depth_path = self.depth_paths[idx]
        depth = np.load(depth_path)
        depth = torch.tensor(depth, dtype=torch.float32)

        depth = depth.unsqueeze(0)

        sample['depth'] = depth

        return sample

class TestDataset(Dataset):
    """Test-set RGB images (no depth); yields image + id for submission."""

    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.image_paths = sorted(list(self.test_dir.glob("*_rgb.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        # "0123_rgb.png" -> "0123"; used to build the submission row id.
        image_id = image_path.stem.replace("_rgb", "")

        return {
            "image": image,
            "id": image_id,
        }


# -----------------------------------------------------------------------------
# Video (clip) datasets
# -----------------------------------------------------------------------------

def _group_sequences(root, rgb_glob, seq_from, seq_regex):
    """Discover ordered RGB frame paths grouped into sequences.

    Returns a list of (sequence_id, [frame_path, ...]) with each frame list
    sorted by filename (so consecutive frames are temporally adjacent).

    ``seq_from``:
      * ``"parent"`` -- the sequence id is the frame's parent directory (each
        subfolder of ``root`` is one video). The default; matches a layout of
        ``root/<seq>/<frame>_rgb.png``.
      * ``"regex"``  -- the sequence id is ``seq_regex`` group 1 matched against
        the file *stem*. Use for a flat directory like ``seq07_0003_rgb.png``
        with ``seq_regex=r"(.+?)_\\d+_rgb"``.
    """
    root = Path(root)
    frames = sorted(root.rglob(rgb_glob))
    if not frames:
        raise FileNotFoundError(
            f"No RGB frames matching {rgb_glob!r} under {root}."
        )

    groups = {}
    if seq_from == "parent":
        for f in frames:
            groups.setdefault(str(f.parent), []).append(f)
    elif seq_from == "regex":
        pattern = re.compile(seq_regex)
        for f in frames:
            m = pattern.match(f.stem)
            if m is None:
                raise ValueError(
                    f"seq_regex {seq_regex!r} did not match frame stem {f.stem!r}."
                )
            groups.setdefault(m.group(1), []).append(f)
    else:
        raise ValueError(f"Unknown seq_from={seq_from!r}; use 'parent' or 'regex'.")

    return [(k, sorted(v)) for k, v in sorted(groups.items())]


def _enumerate_clips(sequences, clip_len, frame_stride, clip_step):
    """Flatten sequences into a list of (seq_idx, [frame_indices]) windows.

    A clip spans ``(clip_len - 1) * frame_stride + 1`` frames; windows start
    every ``clip_step`` frames. Sequences shorter than one clip are padded by
    repeating their last frame (so short videos still yield exactly one clip).
    """
    span = (clip_len - 1) * frame_stride + 1
    clips = []
    for s_idx, (_, frame_paths) in enumerate(sequences):
        n = len(frame_paths)
        if n < span:
            # pad: one clip, last frame repeated to fill the temporal window
            idxs = list(range(0, n, frame_stride))[:clip_len]
            idxs += [n - 1] * (clip_len - len(idxs))
            clips.append((s_idx, idxs))
            continue
        last_start = n - span
        for start in range(0, last_start + 1, clip_step):
            clips.append(
                (s_idx, list(range(start, start + span, frame_stride)))
            )
    return clips


def _rgb_to_chw_tensor(path):
    image = np.array(Image.open(path).convert("RGB"))
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [3,H,W]


def _depth_path_for(rgb_path, rgb_suffix, depth_suffix):
    """Map an RGB frame path to its depth ``.npy`` sibling by suffix swap."""
    name = rgb_path.name
    if rgb_suffix and name.endswith(rgb_suffix):
        depth_name = name[: -len(rgb_suffix)] + depth_suffix
    else:  # no rgb suffix to strip -> just swap the extension
        depth_name = rgb_path.stem + depth_suffix
    return rgb_path.with_name(depth_name)


def _resize_clip(rgb, depth, target_hw):
    """Optionally resize a clip's RGB (bilinear) and depth (nearest) to target_hw."""
    if target_hw is None:
        return rgb, depth
    rgb = F.interpolate(rgb, size=target_hw, mode="bilinear", align_corners=False)
    if depth is not None:
        depth = F.interpolate(depth, size=target_hw, mode="nearest")
    return rgb, depth


class VideoTrainDataset(Dataset):
    """Clips of ``T`` consecutive RGB frames + per-frame depth, for training.

    Config-driven so it fits whatever the final on-disk layout is:

        root/
          seq000/ 0000_rgb.png 0000_depth.npy 0001_rgb.png 0001_depth.npy ...
          seq001/ ...

    is the default (``seq_from="parent"``). A flat directory whose filenames
    encode the sequence (e.g. ``seq07_0003_rgb.png``) works with
    ``seq_from="regex", seq_regex=r"(.+?)_\\d+_rgb"``.

    Each item is ``{"image": [T, 3, H, W], "depth": [T, 1, H, W]}`` (raw uint8
    range RGB; normalization happens in ``vjepa_preprocessing``). Pass
    ``target_hw=(H, W)`` to resize every clip to a common size so that clips
    from differently-sized sequences can share a batch.

    Args:
        root: dataset root directory.
        clip_len: frames per clip (``T``).
        frame_stride: temporal gap between sampled frames (1 = consecutive).
        clip_step: stride between successive clip windows in a sequence
            (default ``clip_len`` = non-overlapping windows).
        rgb_glob: glob (recursive) for RGB frames, e.g. ``"*_rgb.png"``.
        rgb_suffix / depth_suffix: filename suffixes used to pair RGB->depth.
        seq_from / seq_regex: sequence grouping strategy (see ``_group_sequences``).
        target_hw: optional ``(H, W)`` to resize clips to (None = keep native).
    """

    def __init__(
        self,
        root,
        clip_len=16,
        frame_stride=1,
        clip_step=None,
        rgb_glob="*_rgb.png",
        rgb_suffix="_rgb.png",
        depth_suffix="_depth.npy",
        seq_from="parent",
        seq_regex=r"(.+?)_\d+_rgb",
        target_hw=None,
    ):
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.clip_step = clip_step or clip_len
        self.rgb_suffix = rgb_suffix
        self.depth_suffix = depth_suffix
        self.target_hw = target_hw

        self.sequences = _group_sequences(root, rgb_glob, seq_from, seq_regex)
        self.clips = _enumerate_clips(
            self.sequences, clip_len, frame_stride, self.clip_step
        )
        if not self.clips:
            raise RuntimeError(f"No clips enumerated under {root}.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        s_idx, frame_idxs = self.clips[idx]
        _, frame_paths = self.sequences[s_idx]

        rgb_frames, depth_frames = [], []
        for fi in frame_idxs:
            rgb_path = frame_paths[fi]
            rgb_frames.append(_rgb_to_chw_tensor(rgb_path))

            depth_path = _depth_path_for(rgb_path, self.rgb_suffix, self.depth_suffix)
            depth = np.load(depth_path)
            depth_frames.append(torch.tensor(depth, dtype=torch.float32).unsqueeze(0))

        rgb = torch.stack(rgb_frames, dim=0)      # [T, 3, H, W]
        depth = torch.stack(depth_frames, dim=0)  # [T, 1, H, W]
        rgb, depth = _resize_clip(rgb, depth, self.target_hw)

        return {"image": rgb, "depth": depth}


class VideoTestDataset(Dataset):
    """Clips of ``T`` consecutive RGB frames (no depth) + per-frame ids.

    Yields ``{"image": [T, 3, H, W], "id": [id0, ..., id_{T-1}]}``. Clips are
    enumerated with ``clip_step=clip_len`` by default (non-overlapping) so every
    frame is predicted exactly once; for temporally-smoothed inference use an
    overlapping ``clip_step`` and average per-frame predictions downstream.
    """

    def __init__(
        self,
        root,
        clip_len=16,
        frame_stride=1,
        clip_step=None,
        rgb_glob="*_rgb.png",
        rgb_suffix="_rgb.png",
        seq_from="parent",
        seq_regex=r"(.+?)_\d+_rgb",
        target_hw=None,
    ):
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.clip_step = clip_step or clip_len
        self.rgb_suffix = rgb_suffix
        self.target_hw = target_hw

        self.sequences = _group_sequences(root, rgb_glob, seq_from, seq_regex)
        self.clips = _enumerate_clips(
            self.sequences, clip_len, frame_stride, self.clip_step
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        s_idx, frame_idxs = self.clips[idx]
        _, frame_paths = self.sequences[s_idx]

        rgb_frames, ids = [], []
        for fi in frame_idxs:
            rgb_path = frame_paths[fi]
            rgb_frames.append(_rgb_to_chw_tensor(rgb_path))
            stem = rgb_path.stem
            if self.rgb_suffix and stem.endswith(self.rgb_suffix.rsplit(".", 1)[0]):
                stem = stem[: -len(self.rgb_suffix.rsplit(".", 1)[0])]
            ids.append(stem)

        rgb = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]
        rgb, _ = _resize_clip(rgb, None, self.target_hw)

        return {"image": rgb, "id": ids}