from __future__ import annotations

import numpy as np
from pathlib import Path

import click

from air_track.pipeline.config import Config
from air_track.pipeline.video import process_video, roi_per_frame
from air_track.pipeline.cache import save_detections, load_detections
from air_track.pipeline.evaluation import Evaluator, load_ground_truth_JSON
from air_track.tracking import (
    extract_episodes,
    get_tracker,
    run_tracker,
    stitch_tracks,
)
from air_track.utils.outputs import (
    save_config,
    save_episodes_csv,
    plot_occupancy,
    generate_video,
    save_episode_summary,
    episode_table,
)
from air_track.logging import setup_logger, get_logger, get_console
from air_track.utils import increment_path

logger = get_logger(__name__)


def _save_outputs(episodes, save_dir, times_sec, fps, img_save_dir, out_cfg):
    if out_cfg.save_csv:
        save_episodes_csv(save_dir / "episodes.csv", episodes)
    if out_cfg.save_plot:
        plot_occupancy(save_dir / "occupancy.png", times_sec, episodes)
    save_episode_summary(save_dir / "episodes.txt", episodes)

    if out_cfg.generate_video and img_save_dir is not None:
        video_out = save_dir / "detection_video.mp4"
        generate_video(str(img_save_dir), str(video_out), fps=int(fps))
        logger.info(f"Generated video: {video_out}")

    console = get_console()
    console.print(
        f"\n[green bold][DONE][/green bold]\t\tResults saved to: {save_dir}"
    )
    console.print(
        f"  [dim]•[/dim] Frames analysed:    [blue]{len(times_sec)}[/blue]"
    )
    console.print(
        f"  [dim]•[/dim] Episodes found:     [blue]{len(episodes)}[/blue]"
    )
    if episodes:
        console.print(episode_table(episodes))


@click.command()
@click.option(
    "-c", "--config", "config_path",
    type=click.Path(exists=True), default="config/inference.yaml",
    help="Path to config YAML file",
)
@click.option(
    "-s", "--source",
    type=click.Path(exists=True), default="data/videos/runway_occupancy.mp4",
    help="Path to input video file",
)
@click.option(
    "--gt", "gt_path",
    type=click.Path(exists=True), default=None,
    help="Ground-truth JSON for evaluation (optional).",
)
@click.option(
    "-d", "--detections", "detections_path",
    type=click.Path(exists=True), default=None,
    help="Cached detections JSON. If set, skips inference and reuses these detections.",
)
def run(config_path, source, gt_path, detections_path) -> None:
    # Load configuration and setup logger
    cfg = Config.from_yaml(config_path)
    log_cfg = cfg.logging
    setup_logger(
        name=cfg.package_name,
        level=log_cfg.level,
        log_file=getattr(log_cfg, "log_file", None),
    )
    logger.info(f"Loaded config from {config_path}")

    # Setup output directory
    out_cfg = cfg.outputs
    run_name = Path(source).stem
    output_dir = Path(out_cfg.output_dir).resolve()
    save_dir = increment_path(output_dir / run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Frames are only annotated during inference, not when reusing a cache
    save_imgs = out_cfg.save_img and not detections_path
    img_save_dir = save_dir / "images" if save_imgs else None
    if img_save_dir:
        img_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")

    # Detections: reuse cache if given, otherwise run inference over the video
    if detections_path:
        logger.info(f"Loading cached detections: {detections_path}")
        times_sec, sampled_dets, fps = load_detections(detections_path)
        logger.info(f"Loaded {len(sampled_dets)} cached frames")
    else:
        logger.info(f"Processing video: {source}")
        times_sec, sampled_dets, fps = process_video(
            video_path=source,
            config=cfg,
            save_dir=img_save_dir,
        )
        save_detections(
            save_dir / "detections.json", times_sec, sampled_dets, fps, source
        )

    # Track aircraft, then turn tracks into runway-occupancy episodes
    trk_cfg = cfg.tracking
    frame_rate = 1.0 / float(np.median(np.diff(times_sec)))
    tracker = get_tracker(trk_cfg.tracker)(frame_rate, **trk_cfg.params)
    tracks = run_tracker(tracker, sampled_dets)
    logger.info(f"{len(tracks)} tracks from '{trk_cfg.tracker}'")
    # tracks = stitch_tracks(
    #     tracks, times_sec,
    #     trk_cfg.stitch_max_gap_sec, trk_cfg.stitch_max_dist_ratio,
    # )
    logger.info(f"{len(tracks)} tracks after stitching")

    # Define regions of interest for each frame
    rois = roi_per_frame(
        source, cfg.roi.polygon, times_sec, cfg.roi.drift_every_sec
    )

    # Extract runway-occupancy episodes from tracks
    episodes = extract_episodes(tracks, times_sec, rois, cfg.rot)

    # Save outputs
    _save_outputs(episodes, save_dir, times_sec, fps, img_save_dir, out_cfg)
    save_config(save_dir / "config.json", cfg)

    # Evaluate against ground truth if provided
    if gt_path:
        logger.info("Running evaluation …")
        evaluator = Evaluator(cfg.evaluation.iou_threshold)
        ground_truth = load_ground_truth_JSON(gt_path)
        eval_results = evaluator.evaluate(episodes, ground_truth)
        evaluator.print_report(eval_results)
        evaluator.save_report(eval_results, save_dir / "evaluation_report.txt")
        logger.info(f"Saved evaluation report: {save_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    run()
