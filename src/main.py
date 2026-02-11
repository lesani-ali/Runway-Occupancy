import argparse
from pathlib import Path
from rich.console import Console

from config import load_config
from inference import process_video
from postprocessing import fill_short_gaps
from outputs import save_csv, plot_step_chart, generate_video
from utils import increment_path

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="Detect aircraft presence in video using Faster R-CNN"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--source", required=True, help="Path to input video file")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    console.print(
        f"[cyan][INFO][/cyan]\t\tLoaded config from [yellow]{args.config}[/yellow]"
    )

    # Setup output directory
    run_name = Path(args.source).stem
    output_dir = Path(cfg.output_dir).resolve()
    save_dir = increment_path(output_dir / run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[cyan][INFO][/cyan]\t\tOutput directory: [yellow]{save_dir}[/yellow]"
    )

    # Process video
    console.print(
        f"[cyan][INFO][/cyan]\t\tProcessing video: [yellow]{args.source}[/yellow]"
    )
    img_save_dir = save_dir / "images" if cfg.save_img else None
    if img_save_dir:
        img_save_dir.mkdir(parents=True, exist_ok=True)

    times_sec, present_raw, fps = process_video(
        video_path=args.source,
        config=cfg,
        save_img=cfg.save_img,
        save_dir=img_save_dir,
        hide_conf=cfg.hide_conf,
    )

    # Post-process: fill short gaps
    present_filled = fill_short_gaps(present_raw, cfg.max_gap)

    # Save results
    if cfg.save_csv:
        csv_path = save_dir / "presence.csv"
        save_csv(csv_path, times_sec, present_raw, present_filled)
        console.print(f"[cyan][INFO][/cyan]\t\tSaved CSV: [yellow]{csv_path}[/yellow]")

    if cfg.save_plot:
        plot_path = save_dir / "presence_filled.png"
        plot_step_chart(
            out_path=plot_path,
            times_sec=times_sec,
            series=present_filled,
            title=f"Aircraft Presence (gap-filled, max_gap={cfg.max_gap})",
        )

    if cfg.generate_video and img_save_dir is not None:
        video_output_path = save_dir / "detection_video.mp4"
        generate_video(
            input_folder=str(img_save_dir),
            output_video=str(video_output_path),
            fps=int(fps),
        )
        console.print(
            f"[cyan][INFO][/cyan]\t\tGenerated video: [yellow]{video_output_path}[/yellow]"
        )

    console.print(
        f"\n[green bold][DONE][/green bold]\t\tResults saved to: [yellow]{save_dir}[/yellow]"
    )
    console.print(f"  [dim]•[/dim] FPS: [blue]{fps:.2f}[/blue]")
    console.print(f"  [dim]•[/dim] Frames analyzed: [blue]{len(times_sec)}[/blue]")
    console.print(
        f"  [dim]•[/dim] Raw detections: [blue]{present_raw.sum()}[/blue] frames with aircraft"
    )
    console.print(
        f"  [dim]•[/dim] After gap-fill: [blue]{present_filled.sum()}[/blue] frames with aircraft"
    )


if __name__ == "__main__":
    main()
