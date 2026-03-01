from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from rich.table import Table

from config import load_config
from inference import process_video
from episode_machine import run_episode_state_machine
from outputs import save_csv, plot_episode_chart, generate_video
from utils import console, hms, increment_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect aircraft presence in video using Faster R-CNN + episode segmentation"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--source", required=True, help="Path to input video file")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame state machine debug info")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    console.print(f"[cyan][INFO][/cyan]\t\t[green]Loaded config from {args.config}[/green]")

    # Setup output directory
    run_name = Path(args.source).stem
    output_dir = Path(cfg.output_dir).resolve()
    save_dir = increment_path(output_dir / run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan][INFO][/cyan]\t\t[green]Output directory: {save_dir}[/green]")

    # Process video
    console.print(f"[cyan][INFO][/cyan]\t\t[green]Processing video: {args.source}[/green]")
    img_save_dir = save_dir / "images" if cfg.save_img else None
    if img_save_dir:
        img_save_dir.mkdir(parents=True, exist_ok=True)

    times_sec, sampled_dets, fps = process_video(
        video_path=args.source,
        config=cfg,
        save_dir=img_save_dir,
    )

    # Run episode state machine
    console.print("[cyan][INFO][/cyan]\t\t[green]Running episode state machine …[/green]")

    result = run_episode_state_machine(
        num_frames=len(sampled_dets),
        detections=sampled_dets,
        times_sec=times_sec,
        roi_poly=cfg.roi,
        params=cfg.episode_state_machine,
        verbose=args.verbose,
    )

    presence = result["presence"]
    episode_labels = result["episode_labels"]
    episodes = result["episodes"]

    # Save CSV
    if cfg.save_csv:
        csv_path = save_dir / "presence.csv"
        save_csv(csv_path, times_sec, presence, episode_labels)
        console.print(f"[cyan][INFO][/cyan]\t\t[green]Saved CSV: {csv_path}[/green]")

    # Episode chart
    plot_episode_chart(
        out_path=save_dir / "presence_episodes.png",
        times_sec=times_sec,
        presence=presence,
        episode_labels=episode_labels,
        episodes=episodes,
        title="Aircraft Episodes — Runway Occupancy",
    )
    console.print(
        f"[cyan][INFO][/cyan]\t\t[green]Saved episode chart: {save_dir / 'presence_episodes.png'}[/green]"
    )

    # Detection video
    if cfg.generate_video and img_save_dir is not None:
        video_out = save_dir / "detection_video.mp4"
        generate_video(str(img_save_dir), str(video_out), fps=int(fps))
        console.print(f"[cyan][INFO][/cyan]\t\t[green]Generated video: {video_out}[/green]")

    # Console summary
    console.print(f"\n[green bold][DONE][/green bold]\t\t[green]Results saved to: {save_dir}[/green]")
    console.print(f"  [dim]•[/dim] FPS:                [blue]{fps:.2f}[/blue] frames/sec")
    console.print(f"  [dim]•[/dim] Frames analysed:    [blue]{len(times_sec)}[/blue]")
    console.print(f"  [dim]•[/dim] Aircraft presence:  [blue]{int(np.sum(presence))}[/blue] frames")
    console.print(f"  [dim]•[/dim] Episodes found:     [blue]{len(episodes)}[/blue]")

    if episodes:
        tbl = Table(title="Episode Summary", show_header=True, header_style="bold cyan")
        tbl.add_column("Episode", justify="center")
        tbl.add_column("Start",   justify="right")
        tbl.add_column("End",     justify="right")
        tbl.add_column("ROT",     justify="right")
        for ep in episodes:
            tbl.add_row(
                str(ep.episode_id),
                hms(ep.start_sec),
                hms(ep.end_sec),
                hms(ep.rot_seconds),
            )
        console.print(tbl)


if __name__ == "__main__":
    main()
