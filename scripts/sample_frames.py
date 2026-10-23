#!/usr/bin/env python3
"""Sample positive/negative frames from the runway video for detector training.

Positives are drawn from inside annotated episodes (aircraft present), negatives
from the footage between *all* episodes (aircraft absent).  Episodes are used
for the split named in their `split` field; the number of frames taken from an
episode is proportional to its length, and no two sampled frames are closer
together than --min-gap seconds.

    python scripts/sample_frames.py data/rot/gt_data.json \
        data/rot/2026-05-25.mp4 -o data/model_train/frames
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
from pathlib import Path

import cv2

# How many frames to sample per split: (positives, negatives).
SAMPLES_SCHEMA = {
    "train": (400, 30),
    "val": (50, 20),
    "test": (0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_json", type=Path, help="Episode annotations.")
    parser.add_argument("video", type=Path, help="Source video.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("data/model_train/frames")
    )
    parser.add_argument(
        "--min-gap", type=float, default=2.0,
        help="Minimum seconds between any two sampled frames. Default: 2.0",
    )
    parser.add_argument(
        "--episode-margin", type=float, default=1.0,
        help="Seconds trimmed off each end of an episode. Default: 1.0",
    )
    parser.add_argument(
        "--negative-margin", type=float, default=10.0,
        help="Seconds kept clear of any episode when sampling negatives. "
             "Default: 10.0",
    )
    parser.add_argument(
        "--max-per-episode", type=int, default=30,
        help="Cap on frames from a single episode. Default: 30",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan only, write no images."
    )
    return parser.parse_args()


def parse_time(value: str | float) -> float:
    """Parse HH:MM:SS / MM:SS / plain seconds into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def load_episodes(gt_json: Path) -> tuple[list[dict], float]:
    """Read the annotations into episodes with frame ranges."""
    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    fps = float(payload["fps"])

    episodes = []
    for raw in payload["episodes"]:
        start, end = parse_time(raw["start_time"]), parse_time(raw["end_time"])
        if end <= start:
            print(f"  ! episode {raw['episode_id']}: end <= start, skipped")
            continue
        episodes.append(
            {
                "episode_id": int(raw["episode_id"]),
                "start": int(start * fps),
                "end": int(end * fps),
                "action_type": raw.get("action_type", ""),
                "split": (raw.get("split") or "").strip().lower(),
            }
        )
    episodes.sort(key=lambda ep: ep["start"])
    return episodes, fps


def free_spans(episodes: list[dict], margin: int, total: int) -> list[tuple]:
    """Frame ranges outside every episode -- the aircraft-free footage."""
    busy = []
    for ep in episodes:
        start, end = ep["start"] - margin, ep["end"] + margin
        if busy and start <= busy[-1][1]:
            busy[-1] = (busy[-1][0], max(busy[-1][1], end))
        else:
            busy.append((start, end))

    spans, cursor = [], 0
    for start, end in busy:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        spans.append((cursor, total))
    return spans


def allocate(weights: list[int], caps: list[int], total: int) -> list[int]:
    """Share `total` out proportionally to `weights`, respecting `caps`."""
    alloc = [0] * len(weights)
    remaining = total
    while remaining > 0:
        open_idx = [i for i in range(len(caps)) if alloc[i] < caps[i]]
        if not open_idx:
            break
        weight_sum = sum(weights[i] for i in open_idx) or 1
        for i in sorted(open_idx, key=lambda j: -weights[j]):
            if remaining == 0:
                break
            share = max(1, round(weights[i] / weight_sum * remaining))
            take = min(share, caps[i] - alloc[i], remaining)
            alloc[i] += take
            remaining -= take
    return alloc


def pick_frames(
    spans: list[tuple], count: int, min_gap: int, rng: random.Random,
    taken: list[int], cap: int = 10**9,
) -> list[int]:
    """Draw `count` frames spread over `spans`, keeping them min_gap apart."""
    usable = [(s, e) for s, e in spans if e - s >= min_gap]
    if not usable or count <= 0:
        return []

    lengths = [e - s for s, e in usable]
    caps = [min(cap, length // min_gap) for length in lengths]
    per_span = allocate(lengths, caps, count)

    picked = []
    for (start, end), n in zip(usable, per_span):
        # Stride is >= min_gap, and jitter stays inside the stride, so frames
        # from one span are always far enough apart by construction.
        stride = (end - start) / n if n else 0
        for k in range(n):
            low = int(start + k * stride)
            high = low + max(0, int(stride - min_gap))
            # Retry the jitter: where episodes of different splits overlap, a
            # frame may fall too close to one already taken by another split.
            for _ in range(5):
                frame = rng.randint(low, high)
                i = bisect.bisect_left(taken, frame)
                near_prev = i > 0 and frame - taken[i - 1] < min_gap
                near_next = i < len(taken) and taken[i] - frame < min_gap
                if not (near_prev or near_next):
                    picked.append(frame)
                    taken.insert(i, frame)
                    break
    return sorted(picked)


def plan(episodes: list[dict], fps: float, total: int, args) -> list[dict]:
    """Choose every frame to extract, without touching the video."""
    rng = random.Random(args.seed)
    min_gap = max(1, int(args.min_gap * fps))
    margin = int(args.episode_margin * fps)
    taken: list[int] = []
    samples: list[dict] = []

    # Smallest split first: where episodes of different splits overlap, the
    # eval splits get first claim on the footage rather than train.
    for split, (n_pos, _) in sorted(SAMPLES_SCHEMA.items(), key=lambda kv: kv[1][0]):
        pool = [ep for ep in episodes if ep["split"] == split]
        if not pool or n_pos == 0:
            continue
        spans = [(ep["start"] + margin, ep["end"] - margin) for ep in pool]
        lengths = [max(0, e - s) for s, e in spans]
        caps = [
            min(args.max_per_episode, length // min_gap) for length in lengths
        ]

        for ep, span, n in zip(pool, spans, allocate(lengths, caps, n_pos)):
            for frame in pick_frames([span], n, min_gap, rng, taken):
                samples.append(
                    {
                        "split": split,
                        "label": "positive",
                        "frame": frame,
                        "episode_id": ep["episode_id"],
                        "action_type": ep["action_type"],
                    }
                )

    # Negatives: whole free spans are handed to one split each, so no two
    # splits draw background frames from the same stretch of video.
    spans = free_spans(episodes, int(args.negative_margin * fps), total)
    wanted = {s: n for s, (_, n) in SAMPLES_SCHEMA.items() if n > 0}
    held = {s: 0 for s in wanted}
    per_split: dict[str, list[tuple]] = {s: [] for s in wanted}
    for span in sorted(spans, key=lambda sp: sp[0] - sp[1]):
        split = min(wanted, key=lambda s: held[s] / wanted[s])
        per_split[split].append(span)
        held[split] += span[1] - span[0]

    for split, n_neg in wanted.items():
        frames = pick_frames(
            sorted(per_split[split]), n_neg, min_gap, rng, taken
        )
        samples.extend(
            {
                "split": split,
                "label": "negative",
                "frame": frame,
                "episode_id": "",
                "action_type": "",
            }
            for frame in frames
        )

    for sample in samples:
        sample["time_sec"] = round(sample["frame"] / fps, 3)
    return sorted(samples, key=lambda s: s["frame"])


def extract(samples: list[dict], video: Path, out_dir: Path) -> None:
    """Seek to each planned frame and write it out."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    for i, sample in enumerate(samples, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["frame"])
        ok, frame = cap.read()
        if not ok:
            print(f"  ! could not read frame {sample['frame']}")
            sample["image"] = ""
            continue

        episode = sample["episode_id"]
        stem = f"ep{episode:03d}" if episode != "" else "neg"
        name = f"{stem}_f{sample['frame']:08d}.jpg"
        rel = Path(sample["split"], sample["label"], name)
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / rel), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sample["image"] = rel.as_posix()

        if i % 50 == 0 or i == len(samples):
            print(f"  extracted {i}/{len(samples)}")
    cap.release()


def main() -> None:
    args = parse_args()
    episodes, fps = load_episodes(args.gt_json)
    print(f"Loaded {len(episodes)} episodes at {fps:g} fps")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    samples = plan(episodes, fps, total, args)
    for split, (n_pos, n_neg) in SAMPLES_SCHEMA.items():
        for label, want in (("positive", n_pos), ("negative", n_neg)):
            got = sum(
                1 for s in samples
                if s["split"] == split and s["label"] == label
            )
            flag = "" if got >= want else "  <- not enough spaced footage"
            print(f"  {split:<5} {label:<8} {got}/{want}{flag}")

    if args.dry_run:
        print(f"Dry run: {len(samples)} frames planned")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract(samples, args.video, args.output_dir)

    manifest = args.output_dir / "manifest.csv"
    fields = [
        "image", "split", "label", "frame", "time_sec", "episode_id",
        "action_type",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: s.get(k, "") for k in fields} for s in samples)
    print(f"Wrote {len(samples)} frames and {manifest}")


if __name__ == "__main__":
    main()
