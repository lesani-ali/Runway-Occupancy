import argparse
from pathlib import Path
import subprocess
from rich.console import Console

console = Console()


def mkv_to_mp4(input_file: str, output_file: str = None) -> bool:
    input_path = Path(input_file)

    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        return False

    if not input_path.suffix.lower() == ".mkv":
        console.print(
            f"[yellow]Warning:[/yellow]\t\tInput file is not .mkv: {input_file}"
        )

    # Generate output filename if not provided
    if output_file is None:
        output_path = input_path.with_suffix(".mp4")
    else:
        output_path = Path(output_file)

    console.print(
        f"[cyan]\t\tConverting:[/cyan] {input_path.name} → {output_path.name}"
    )

    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-c:v",
        "copy",  # Copy video codec
        "-c:a",
        "copy",  # Copy audio codec
        "-y",  # Overwrite output
        str(output_path),
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode == 0:
        console.print(f"[green]✓[/green] \t\tSuccess: {output_path}")

        # Remove the input MKV file after successful conversion
        try:
            input_path.unlink()
            console.print(f"[dim]\t\tRemoved: {input_path.name}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]\t\tWarning:[/yellow] \t\tCouldn't remove input file: {e}"
            )

        return True
    else:
        console.print(f"[red]✗[/red] \t\tFailed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input MKV file or directory")
    parser.add_argument(
        "-o", "--output", help="Output MP4 file (only for single file conversion)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        mkv_to_mp4(args.input, args.output)
    else:
        console.print(f"[red]Error:[/red]\t\tInvalid input: {args.input}")


if __name__ == "__main__":
    main()
