from __future__ import annotations

import numpy as np
from pathlib import Path

import click
from rich.table import Table

from air_track.config import load_config
from air_track.detection import process_video
from air_track.tracking import run_episode_state_machine
from air_track.utils.outputs import (
    save_csv,
    plot_episode_chart,
    generate_video,
    save_episode_summary,
)
from air_track.utils.mkv2mp4 import mkv_to_mp4
from air_track.utils import hms, increment_path
from air_track.logging import setup_logger, get_logger, get_console

logger = get_logger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default="config/config.yaml",
    help="Path to config YAML file",
)
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    default="data/videos/runway_occupancy.mp4",
    help="Path to input video file",
)
def run(config, source) -> None:
    # Load configuration and setup logger
    cfg = load_config(config)
    log_cfg = cfg.logging
    setup_logger(
        level=log_cfg.level,
        log_file=getattr(log_cfg, "log_file", None),
        verbose=log_cfg.verbose,
    )
    logger.info(f"Loaded config from {config}")

    # Setup output directory
    out_cfg = cfg.outputs
    run_name = Path(source).stem
    output_dir = Path(out_cfg.output_dir).resolve()
    save_dir = increment_path(output_dir / run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")

    # Process video
    logger.info(f"Processing video: {source}")
    img_save_dir = save_dir / "images" if out_cfg.save_img else None
    if img_save_dir:
        img_save_dir.mkdir(parents=True, exist_ok=True)

    times_sec, sampled_dets, fps = process_video(
        video_path=source,
        config=cfg,
        save_dir=img_save_dir,
    )

    # Run episode state machine
    logger.info("Running episode state machine …")
    result = run_episode_state_machine(
        num_frames=len(sampled_dets),
        detections=sampled_dets,
        times_sec=times_sec,
        roi_poly=cfg.roi.polygon,
        params=cfg.tracking,
    )

    presence = result["presence"]
    episode_labels = result["episode_labels"]
    episodes = result["episodes"]

    # Save CSV
    if out_cfg.save_csv:
        csv_path = save_dir / "presence.csv"
        save_csv(csv_path, times_sec, presence, episode_labels)
        logger.info(f"Saved CSV: {csv_path}")

    # Episode chart
    plot_episode_chart(
        out_path=save_dir / "presence_episodes.png",
        times_sec=times_sec,
        presence=presence,
        episode_labels=episode_labels,
        episodes=episodes,
        title="Aircraft Episodes — Runway Occupancy",
    )
    logger.info(f"Saved episode chart: {save_dir / 'presence_episodes.png'}")

    # Detection video
    if out_cfg.generate_video and img_save_dir is not None:
        video_out = save_dir / "detection_video.mp4"
        generate_video(str(img_save_dir), str(video_out), fps=int(fps))
        logger.info(f"Generated video: {video_out}")

    # Console summary
    console = get_console()
    console.print(f"\n[green bold][DONE][/green bold]\t\tResults saved to: {save_dir}")
    console.print(
        f"  [dim]•[/dim] FPS:                [blue]{fps:.2f}[/blue] frames/sec"
    )
    console.print(f"  [dim]•[/dim] Frames analysed:    [blue]{len(times_sec)}[/blue]")
    console.print(
        f"  [dim]•[/dim] Aircraft presence:  [blue]{int(np.sum(presence))}[/blue] frames"
    )
    console.print(f"  [dim]•[/dim] Episodes found:     [blue]{len(episodes)}[/blue]")

    if episodes:
        tbl = Table(title="Episode Summary", show_header=True, header_style="bold cyan")
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
        console.print(tbl)

        summary_path = save_dir / "episodes.txt"
        save_episode_summary(summary_path, episodes)
        logger.info(f"Saved episode summary: {summary_path}")


if __name__ == "__main__":
    cli()
