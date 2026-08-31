#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HEADER_ALIASES = {
    "episode_id": {"episode_id", "episode id", "Episode ID"},
    "start_time": {"start_time", "start_sec", "start sec", "Start Time", "start"},
    "end_time": {"end_time", "end_sec", "end sec", "End Time", "end"},
    "action_type": {"action_type", "action type", "Action Type"},
    "note": {"note", "Note"},
}

ACTION_MAP = {
    "landing": "landing",
    "take off": "take-off",
    "take-off": "take-off",
    "touch and go": "touch-and-go",
    "touch-and-go": "touch-and-go",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a runway episode CSV into a gt.json-style file."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        help="Path to the output JSON file. Defaults to input name with .json suffix.",
    )
    parser.add_argument(
        "--three-part-format",
        choices=("hms")
    )
    parser.add_argument(
        "--video",
        help="Video filename to store in the JSON metadata. Defaults to <input-stem>.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS value to store in the JSON metadata. Defaults to 30.0.",
    )
    parser.add_argument(
        "--annotator",
        default="",
        help="Annotator name to store in the JSON metadata. Defaults to an empty string.",
    )
    return parser.parse_args()


def _normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().replace("_", " ").split())


def _resolve_headers(fieldnames: list[str]) -> dict[str, str]:
    normalized = {_normalize_header(name): name for name in fieldnames}
    resolved: dict[str, str] = {}

    for canonical_name, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            match = normalized.get(_normalize_header(alias))
            if match:
                resolved[canonical_name] = match
                break

    required = {"episode_id", "start_time", "end_time"}
    missing = sorted(required - resolved.keys())
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_text}")

    return resolved




def _parse_time(value: str, three_part_format: str | None=None) -> float:
    value = value.strip()
    if not value:
        raise ValueError("Time value is empty")

    if not three_part_format:
        return value

    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        first, second, third = parts

        if three_part_format == "hms":
            return int(first) * 3600 + int(second) * 60 + float(third)

    raise ValueError(f"Unsupported time format: {value}")


def _normalize_action_type(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = " ".join(value.strip().lower().replace("-", " ").split())
    if not normalized:
        return None
    return ACTION_MAP.get(normalized, normalized.replace(" ", "-"))


def convert_csv_to_gt_json(input_csv: Path, three_part_format: str) -> list[dict]:
    with input_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")

        headers = _resolve_headers(reader.fieldnames)
        episodes = []

        for row_number, row in enumerate(reader, start=2):
            raw_episode_id = (row.get(headers["episode_id"]) or "").strip()
            raw_start = (row.get(headers["start_time"]) or "").strip()
            raw_end = (row.get(headers["end_time"]) or "").strip()

            if not raw_episode_id and not raw_start and not raw_end:
                continue

            try:
                episode = {
                    "episode_id": int(raw_episode_id),
                    "start_time": _parse_time(raw_start, three_part_format),
                    "end_time": _parse_time(raw_end, three_part_format),
                    "action_type": _normalize_action_type(
                        row.get(headers.get("action_type", ""))
                    ),
                    "note": (row.get(headers.get("note", "")) or "").strip(),
                }
            except Exception as exc:
                raise ValueError(f"Row {row_number}: {exc}") from exc

            episodes.append(episode)

    return sorted(episodes, key=lambda item: item["start_time"])


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    output_json = args.output_json or input_csv.with_suffix(".json")
    video_name = args.video or f"{input_csv.stem}.mp4"

    episodes = convert_csv_to_gt_json(input_csv, args.three_part_format)
    payload = {
        "video": video_name,
        "fps": args.fps,
        "annotator": args.annotator,
        "episodes": episodes,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(episodes)} episodes to {output_json}")


if __name__ == "__main__":
    main()
