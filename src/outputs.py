import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
import os
import gc
import re
import imageio.v2 as iio

from utils import console, hms


def save_csv(
    out_path: Path,
    times_sec: np.ndarray,
    presence: np.ndarray,
    episode_labels: np.ndarray,
) -> None:
    """Save per-frame presence and episode labels to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("time_sec,presence,episode_id\n")
        for t, p, ep in zip(times_sec, presence, episode_labels):
            f.write(f"{t:.6f},{int(p)},{int(ep)}\n")


_EP_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def plot_episode_chart(
    out_path: Optional[Path],
    times_sec: np.ndarray,
    presence: np.ndarray,
    episode_labels: np.ndarray,
    episodes: List,
    title: str = "Aircraft Episodes",
) -> None:
    """
    Step chart with colour-shaded bands per episode.
    The trace is coloured per episode; grey where no episode is active.
    """
    times_sec = np.asarray(times_sec)
    presence = np.asarray(presence)
    episode_labels = np.asarray(episode_labels)

    fig, ax = plt.subplots(figsize=(max(12, len(times_sec) / 50), 4))

    legend_patches = []
    ep_color_map = {}
    for ep in episodes:
        color = _EP_COLORS[(ep.episode_id - 1) % len(_EP_COLORS)]
        ep_color_map[ep.episode_id] = color
        ax.axvspan(ep.start_sec, ep.end_sec, alpha=0.15, color=color, linewidth=0)

        mid = (ep.start_sec + ep.end_sec) / 2
        dur_str = hms(ep.rot_seconds)
        ax.text(
            mid, 1.05, f"Ep {ep.episode_id}\n{dur_str}",
            ha="center", va="bottom", fontsize=7, color=color, fontweight="bold",
        )
        legend_patches.append(
            mpatches.Patch(color=color, alpha=0.6,
                           label=f"Ep {ep.episode_id} ({dur_str})"))

    # Coloured trace: one segment per unique episode label (0 = grey)
    unique_labels = sorted(set(episode_labels))
    for ep_id in unique_labels:
        mask = episode_labels == ep_id
        color = ep_color_map.get(ep_id, "#aaaaaa")  # grey for ep_id == 0
        lw    = 1.5 if ep_id > 0 else 0.8
        # Draw masked frames; gaps between segments handled by NaN
        y = np.where(mask, presence.astype(float), np.nan)
        ax.step(times_sec, y, where="post", color=color, linewidth=lw)

    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (absent)", "1 (present)"])
    ax.set_xlabel("Time (seconds from video start)")
    ax.set_ylabel("Aircraft visible")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def generate_video(input_folder: str, output_video: str, fps: int = 30) -> None:
    """Generate a video from a folder of sequentially-named images."""
    writer = iio.get_writer(output_video, fps=fps)

    img_files = sorted(
        [f for f in os.listdir(input_folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0,
    )

    if not img_files:
        raise ValueError(f"No image files found in {input_folder}")

    console.print(f"[cyan][INFO][/cyan]\t\t[green]Found {len(img_files)} images[/green]")

    with Progress(
        TextColumn("[green]{task.description}[/green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating video", total=len(img_files))
        for img_file in img_files:
            img_path = os.path.join(input_folder, img_file)
            frame = iio.imread(img_path)
            writer.append_data(frame)
            gc.collect()
            progress.update(task, advance=1)

    writer.close()