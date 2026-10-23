import numpy as np
import matplotlib

matplotlib.use("Agg")  # files only, no GUI
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List
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
from air_track.logging import get_logger, get_console

logger = get_logger(__name__)


def episode_table(episodes: List, title: str = "Episode Summary") -> Table:
    """Rich table of episodes (shared by console and file output)."""
    tbl = Table(title=title, show_header=True, header_style="bold cyan")
    for col in ("Episode", "Track", "Type", "Start", "End", "ROT"):
        tbl.add_column(col, justify="right")
    for ep in episodes:
        tbl.add_row(
            str(ep.episode_id),
            str(ep.track_id),
            ep.action_type,
            hms(ep.start_sec),
            hms(ep.end_sec),
            f"{ep.rot_seconds:.1f} s",
        )
    return tbl


def save_episode_summary(out_path: Path, episodes: List) -> None:
    """Write the episode summary table to a plain-text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        _FileConsole(file=f, no_color=True, highlight=False, width=80).print(
            episode_table(episodes)
        )


def save_episodes_csv(out_path: Path, episodes: List) -> None:
    """Save one row per episode to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("episode_id,track_id,action_type,start_sec,end_sec,rot_sec\n")
        for ep in episodes:
            f.write(
                f"{ep.episode_id},{ep.track_id},{ep.action_type},"
                f"{ep.start_sec:.3f},{ep.end_sec:.3f},{ep.rot_seconds:.3f}\n"
            )


_ACTION_COLORS = {
    "landing": "#4e79a7",
    "take-off": "#e15759",
    "touch-and-go": "#59a14f",
    "ground": "#9c755f",
}


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


def plot_occupancy(
    out_path: Path,
    times_sec: np.ndarray,
    episodes: List,
    title: str = "Runway Occupancy",
) -> None:
    """Number of aircraft on the runway over time, with episode bands."""
    times_sec = np.asarray(times_sec)
    occupancy = np.zeros(len(times_sec), dtype=int)
    for ep in episodes:
        occupancy += (times_sec >= ep.start_sec) & (times_sec <= ep.end_sec)

    duration = float(times_sec[-1]) if len(times_sec) > 0 else 1.0
    width = float(np.clip(duration / 15, 14, 36))
    fig, ax = plt.subplots(figsize=(width, 4))

    for ep in episodes:
        ax.axvspan(
            ep.start_sec,
            ep.end_sec,
            alpha=0.2,
            color=_ACTION_COLORS.get(ep.action_type, "#bab0ac"),
            linewidth=0,
        )
    ax.step(times_sec, occupancy, where="post", color="black", linewidth=1)

    ax.legend(
        handles=[
            mpatches.Patch(color=c, alpha=0.5, label=a)
            for a, c in _ACTION_COLORS.items()
        ],
        loc="upper right",
        fontsize=7,
    )
    ax.set_ylim(-0.1, max(2, int(occupancy.max(initial=0))) + 0.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylabel("Aircraft on runway")
    ax.set_title(title)

    tick_step = _tick_interval(duration)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda s, _: hms(s))
    )
    ax.set_xlabel("Time (HH:MM:SS from video start)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_video(input_folder: str, output_video: str, fps: int = 30) -> None:
    """Generate a video from a folder of sequentially-named images."""
    writer = iio.get_writer(output_video, fps=fps)

    img_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=lambda x: (
            int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0
        ),
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


def save_config(out_path: Path, cfg) -> None:
    """Save the config to a YAML file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(cfg.model_dump_json(indent=4))
