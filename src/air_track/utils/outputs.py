import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Optional, List
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
import os
import gc
import re
import imageio.v2 as iio

from rich.console import Console as _FileConsole
from rich.table import Table

from air_track.utils.utils import hms
from air_track.logging.logger import get_logger, get_console

logger = get_logger(__name__)


def save_episode_summary(out_path: Path, episodes: List) -> None:
    """Write the episode summary table to a plain-text file."""
    tbl = Table(title="Episode Summary", show_header=True)
    tbl.add_column("Episode", justify="center")
    tbl.add_column("Start", justify="right")
    tbl.add_column("End", justify="right")
    tbl.add_column("ROT", justify="right")
    for ep in episodes:
        tbl.add_row(
            str(ep.episode_id),
            hms(ep.start_sec),
            hms(ep.end_sec),
            hms(ep.rot_seconds),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        _FileConsole(file=f, no_color=True, highlight=False, width=60).print(tbl)


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
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def _tick_interval(duration_sec: float) -> float:
    """Pick a readable major-tick spacing (in seconds) for a given video duration."""
    if duration_sec < 120:
        return 10
    if duration_sec < 300:
        return 30
    if duration_sec < 900:
        return 60
    if duration_sec < 1800:
        return 120
    if duration_sec < 3600:
        return 300
    return 600


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
    Axis labels are formatted as MM:SS regardless of video length.
    """
    times_sec = np.asarray(times_sec)
    presence = np.asarray(presence)
    episode_labels = np.asarray(episode_labels)

    duration = float(times_sec[-1]) if len(times_sec) > 0 else 1.0
    width = float(np.clip(duration / 15, 14, 36))
    fig, ax = plt.subplots(figsize=(width, 4))

    legend_patches = []
    ep_color_map = {}
    min_label_width = duration / 20

    for ep in episodes:
        color = _EP_COLORS[(ep.episode_id - 1) % len(_EP_COLORS)]
        ep_color_map[ep.episode_id] = color
        ax.axvspan(ep.start_sec, ep.end_sec, alpha=0.15, color=color, linewidth=0)

        dur_str = hms(ep.rot_seconds)
        band_width = ep.end_sec - ep.start_sec
        if band_width >= min_label_width:
            mid = (ep.start_sec + ep.end_sec) / 2
            ax.text(
                mid,
                1.05,
                f"Ep {ep.episode_id}\n{dur_str}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
                clip_on=True,
            )
        legend_patches.append(
            mpatches.Patch(
                color=color, alpha=0.6, label=f"Ep {ep.episode_id} ({dur_str})"
            )
        )

    unique_labels = sorted(set(episode_labels))
    for ep_id in unique_labels:
        mask = episode_labels == ep_id
        color = ep_color_map.get(ep_id, "#aaaaaa")
        lw = 1.5 if ep_id > 0 else 0.8
        draw_mask = (
            mask & (presence == 1) if ep_id > 0 else mask & presence.astype(bool)
        )
        y = np.where(draw_mask, 1.0 if ep_id > 0 else presence.astype(float), np.nan)
        ax.step(times_sec, y, where="post", color=color, linewidth=lw)

    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (absent)", "1 (present)"])
    ax.set_ylabel("Aircraft visible")
    ax.set_title(title)

    tick_step = _tick_interval(duration)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_step / 5))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda s, _: f"{int(s)//60:02d}:{int(s)%60:02d}")
    )
    ax.set_xlabel("Time (MM:SS from video start)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.2, alpha=0.3)

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=7,
            ncol=max(1, len(legend_patches) // 4),
        )

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def generate_video(input_folder: str, output_video: str, fps: int = 30) -> None:
    """Generate a video from a folder of sequentially-named images."""
    writer = iio.get_writer(output_video, fps=fps)

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

    logger.info(f"Found {len(img_files)} images for video generation")

    with Progress(
        TextColumn("[green]{task.description}[/green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    ) as progress:
        task = progress.add_task("Generating video", total=len(img_files))
        for img_file in img_files:
            img_path = os.path.join(input_folder, img_file)
            frame = iio.imread(img_path)
            writer.append_data(frame)
            gc.collect()
            progress.update(task, advance=1)

    writer.close()
