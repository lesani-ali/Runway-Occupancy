import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gc
import re
import imageio.v2 as iio


def save_csv(
    out_path: Path, times_sec: np.ndarray, raw: np.ndarray, filled: np.ndarray
):
    """Save presence detection results to CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("time_sec,present_raw,present_filled\n")
        for t, r, fi in zip(times_sec, raw, filled):
            f.write(f"{t:.6f},{int(r)},{int(fi)}\n")


def plot_step_chart(
    out_path: Optional[Path], times_sec: np.ndarray, series: np.ndarray, title: str
):
    """Create and save a step chart showing aircraft presence over time."""

    plt.figure()
    plt.step(times_sec, series, where="post")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["0 (no aircraft)", "1 (aircraft)"])
    plt.xlabel("Time (seconds from video start)")
    plt.ylabel("Aircraft visible")
    plt.title(title)
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved plot: {out_path}")
    else:
        plt.show()

    plt.close()


def generate_video(input_folder: str, output_video: str, fps: int = 30):
    """Generate video from images in a folder.

    Args:
        input_folder (str): Path to folder containing images
        output_video (str): Output video file path
        fps (int): Frames per second for output video
    """
    writer = iio.get_writer(output_video, fps=fps)

    # Get all image files and sort them numerically
    img_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0,
    )

    if not img_files:
        raise ValueError(f"No image files found in {input_folder}")

    print(f"Found {len(img_files)} images")

    for img_file in tqdm(img_files, desc="Generating video", ncols=100):
        img_path = os.path.join(input_folder, img_file)
        frame = iio.imread(img_path)
        writer.append_data(frame)
        gc.collect()

    writer.close()