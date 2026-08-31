from __future__ import annotations

import csv
import json
import math
from io import StringIO
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from rich.console import Console as _FileConsole
from rich.table import Table

from air_track.tracking.episode_machine import Episode
from air_track.logging import get_logger, get_console

logger = get_logger(__name__)


@dataclass
class GTEpisode:
    episode_id: int
    start_sec: float
    end_sec: float
    action_type: Literal["take-off", "landing", "touch-and-go"] | None = None
    note: str | None = None


    @property
    def rot_seconds(self) -> float:
        return self.end_sec - self.start_sec


def _parse_time(value: str) -> float:
    """Convert ``HH:MM:SS`` / ``MM:SS`` / float string → seconds."""
    value = value.strip()
    if ":" in value:
        parts = value.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        m, s = parts
        return int(m) * 60 + float(s)
    return float(value)


def load_ground_truth_CSV(path: str | Path) -> List[GTEpisode]:
    """Load ground-truth episodes from a CSV file."""
    path = Path(path)
    episodes: List[GTEpisode] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ep = GTEpisode(
                episode_id=int(row["episode_id"]),
                start_sec=_parse_time(row["start_time"]),
                end_sec=_parse_time(row["end_time"]),
                note=row.get("note"),
            )
            episodes.append(ep)
    return sorted(episodes, key=lambda e: e.start_sec)


def load_ground_truth_JSON(path: str | Path) -> List[GTEpisode]:
    """Load ground-truth episodes from a JSON file."""
    path = Path(path)
    episodes: List[GTEpisode] = []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
        for row in data["episodes"]:
            ep = GTEpisode(
                episode_id=int(row["episode_id"]),
                start_sec=_parse_time(row["start_time"]),
                end_sec=_parse_time(row["end_time"]),
                action_type=row.get("action_type"),
                note=row.get("note"),
            )
            episodes.append(ep)
    return sorted(episodes, key=lambda e: e.start_sec)


def _tiou(pred: Episode, gt: GTEpisode) -> float:
    inter = max(0.0, min(pred.end_sec, gt.end_sec) - max(pred.start_sec, gt.start_sec))
    union = max(pred.end_sec, gt.end_sec) - min(pred.start_sec, gt.start_sec)
    return inter / union if union > 0 else 0.0


class Evaluator:

    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: Minimum temporal IoU to count a prediction as a true positive
        """
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predicted: List[Episode],
        ground_truth: List[GTEpisode],
    ) -> dict:
        """Match episodes, then compute detection and ROT accuracy metrics."""
        matches = self._greedy_match(predicted, ground_truth)
        detection = self._detection_metrics(matches, len(predicted), len(ground_truth))
        rot = self._rot_metrics(matches)
        return {**detection, **rot, "matches": matches}

    def print_report(self, results: dict) -> None:
        """Print the evaluation report to the console."""
        console = get_console()
        console.rule("[bold cyan]Evaluation[/bold cyan]")
        console.print(self._build_table(results))

    def save_report(self, results: dict, path: Path) -> None:
        """Write the evaluation report to a plain-text file."""
        buf = StringIO()
        file_console = _FileConsole(file=buf, no_color=True, highlight=False, width=80)
        file_console.rule("Evaluation")
        file_console.print(self._build_table(results))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(buf.getvalue(), encoding="utf-8")
        logger.info(f"Evaluation report saved → {path}")

    def _build_table(self, results: dict) -> Table:
        """Build and return a populated Rich Table from results."""
        def _f(v):  return "N/A" if math.isnan(v) else f"{v:.2f}"
        def _fp(v): return "N/A" if math.isnan(v) else f"{v:.1f}%"
        def _fs(v): return "N/A" if math.isnan(v) else f"{v:+.2f} s"

        tbl = Table(show_header=True)
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        tbl.add_row("GT episodes",        str(results["tp"] + results["fn"]))
        tbl.add_row("Predicted episodes", str(results["tp"] + results["fp"]))
        tbl.add_row("TP / FP / FN",       f"{results['tp']} / {results['fp']} / {results['fn']}")
        tbl.add_row("Precision",          _fp(results["precision"] * 100))
        tbl.add_row("Recall",             _fp(results["recall"]    * 100))
        tbl.add_row("F1",                 _fp(results["f1"]        * 100))
        tbl.add_row("Mean tIoU",          _f(results["mean_tiou"]))
        tbl.add_row("ROT MAE (s)",        _f(results["mae_sec"]))
        tbl.add_row("ROT RMSE (s)",       _f(results["rmse_sec"]))
        tbl.add_row("ROT Bias (s)",       _fs(results["mean_rot_err_sec"]))
        tbl.add_row("Mean Rel. ROT Err",  _fp(results["mean_rel_rot_err_pct"]))
        tbl.add_row("Mean Start Error",   _fs(results["mean_start_err_sec"]))
        tbl.add_row("Mean End Error",     _fs(results["mean_end_err_sec"]))
        return tbl

    def _greedy_match(
        self,
        predicted: List[Episode],
        ground_truth: List[GTEpisode],
    ) -> list:
        """Greedy tIoU matching. Returns list of (pred, gt, tiou) tuples."""
        scores = []
        for i, pred in enumerate(predicted):
            for j, gt in enumerate(ground_truth):
                tiou = _tiou(pred, gt)
                if tiou >= self.iou_threshold:
                    scores.append((tiou, i, j))
        scores.sort(key=lambda x: -x[0])

        matched_p, matched_g, matches = set(), set(), []
        for tiou, pi, gi in scores:
            if pi not in matched_p and gi not in matched_g:
                matches.append((predicted[pi], ground_truth[gi], tiou))
                matched_p.add(pi)
                matched_g.add(gi)
        return matches

    def _detection_metrics(self, matches: list, n_pred: int, n_gt: int) -> dict:
        """Episode-level Precision, Recall, F1 from match count."""
        tp = len(matches)
        fp = n_pred - tp
        fn = n_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
            else float("nan")
        )
        return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

    def _rot_metrics(self, matches: list) -> dict:
        """ROT accuracy and boundary metrics over matched pairs."""
        nan = float("nan")
        if not matches:
            return dict(
                mean_tiou=nan,
                mae_sec=nan, rmse_sec=nan,
                mean_rot_err_sec=nan, mean_rel_rot_err_pct=nan,
                mean_start_err_sec=nan, mean_end_err_sec=nan,
            )

        tious, rot_errs, abs_errs, rel_errs, start_errs, end_errs = [], [], [], [], [], []
        for pred, gt, tiou in matches:
            tious.append(tiou)
            rot_errs.append(pred.rot_seconds - gt.rot_seconds)
            abs_errs.append(abs(pred.rot_seconds - gt.rot_seconds))
            rel_errs.append(
                abs(pred.rot_seconds - gt.rot_seconds) / gt.rot_seconds * 100
                if gt.rot_seconds > 0 else 0
            )
            start_errs.append(pred.start_sec - gt.start_sec)
            end_errs.append(pred.end_sec - gt.end_sec)

        return dict(
            mean_tiou=float(np.mean(tious)),
            mae_sec=float(np.mean(abs_errs)),
            rmse_sec=float(np.sqrt(np.mean(np.square(abs_errs)))),
            mean_rot_err_sec=float(np.mean(rot_errs)),
            mean_rel_rot_err_pct=float(np.mean(rel_errs)) if rel_errs else nan,
            mean_start_err_sec=float(np.mean(start_errs)),
            mean_end_err_sec=float(np.mean(end_errs)),
        )
