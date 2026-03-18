from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from air_track.tracking.geometry import (
    Detection,
    Polygon,
    filter_by_roi,
    passes_gating,
    point_in_poly,
    select_best_near_prediction,
    select_highest_conf,
)
from air_track.tracking.tracker import SingleObjectTracker
from air_track.logging import get_logger, get_console

logger = get_logger(__name__)


class State(Enum):
    IDLE = 0
    ACTIVE = 1
    LOST = 2


@dataclass
class Episode:
    """Timing record for one aircraft-on-runway episode."""

    episode_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    rot_seconds: float


class EpisodeMachine:
    """3-state episode machine: IDLE → ACTIVE → LOST → IDLE."""

    def __init__(
        self,
        num_frames: int,
        times_sec: np.ndarray,
        roi_poly: Optional[Polygon],
        params,
    ):
        self.times_sec = times_sec
        self.roi_poly = roi_poly
        self.params = params

        # Outputs
        self.presence: List[int] = []
        self.episode_labels: np.ndarray = np.zeros(num_frames, dtype=np.int32)
        self.episodes: List[Episode] = []

        # Machine state
        self.state = State.IDLE
        self.episode_id = 0
        self.episode_start = None
        self.last_reliable = None
        self.missing_count = 0
        self.tracker: Optional[SingleObjectTracker] = None
        self.start_window: deque = deque(maxlen=params.start_streak_n)

    def step(self, t: int, raw_dets: List[Detection]) -> None:
        """Process one sampled frame."""
        self.presence.append(0)  # default; overwritten below
        candidates = filter_by_roi(raw_dets, self.roi_poly)

        logger.debug(f"State: {self.state.name}, t: {t}, candidates: {len(candidates)}")

        if self.state == State.IDLE:
            self._handle_idle(t, candidates)
        elif self.state == State.ACTIVE:
            self._handle_active(t, candidates)
        elif self.state == State.LOST:
            self._handle_lost(t, candidates)

    def finalize(self) -> None:
        """Close any episode still open at end of video."""
        if self.state in (State.ACTIVE, State.LOST) and self.episode_start is not None:
            logger.debug(
                f"Finalizing episode {self.episode_id}: state={self.state}, start={self.episode_start}, last_reliable={self.last_reliable}",
            )
            self.episodes.append(
                self._make_episode(self.episode_start, self.last_reliable)
            )

    def _handle_idle(self, t: int, candidates: List[Detection]) -> None:
        p = self.params
        best = select_highest_conf(candidates)
        strong = best is not None and best[1] >= p.start_conf
        self.start_window.append((t, strong))

        streak = sum(flag for _, flag in self.start_window)
        if streak < p.start_streak_k:
            return  # presence[t] stays 0

        # Streak met → open episode
        ep_start = min(fi for fi, flag in self.start_window if flag)
        self.episode_id += 1
        self.episode_start = ep_start
        self.last_reliable = t
        self.missing_count = 0
        self.tracker = SingleObjectTracker(
            best,
            dt=1.0,
            process_var=p.process_var,
            meas_var=p.meas_var,
        )
        self.state = State.ACTIVE
        self.start_window.clear()

        # Back-date presence and labels to the first strong frame
        for i in range(ep_start, t + 1):
            self.presence[i] = 1
        self.episode_labels[ep_start : t + 1] = self.episode_id

    def _handle_active(self, t: int, candidates: List[Detection]) -> None:
        p = self.params
        self.episode_labels[t] = self.episode_id

        pred_xy = self.tracker.predict()
        pred_reliable = self._pred_reliable(pred_xy)
        cand = select_best_near_prediction(candidates, pred_xy)
        good_det = cand is not None and cand[1] >= p.keep_conf

        if good_det and passes_gating(cand[0], pred_xy, p.gate_px):
            # Good detection inside gate → update tracker
            self.tracker.update_with_detection(cand)
            self.missing_count = 0
            self.last_reliable = t
            self.presence[t] = 1

        elif pred_reliable:
            # No gated detection but prediction is trustworthy → keep alive
            self.missing_count += 1
            self.presence[t] = 1

        else:
            # Nothing to hold on to → go LOST
            self.missing_count += 1
            self.presence[t] = 0
            self.state = State.LOST

    def _handle_lost(self, t: int, candidates: List[Detection]) -> None:
        p = self.params

        pred_xy = self.tracker.predict()
        cand = select_best_near_prediction(candidates, pred_xy)
        recovered = (
            cand is not None
            and cand[1] >= p.recover_conf
            and passes_gating(cand[0], pred_xy, p.gate_px)
        )

        if recovered:
            # Backfill the gap frames
            gap_start = t - self.missing_count
            for i in range(gap_start, t + 1):
                self.presence[i] = 1
            self.episode_labels[gap_start : t + 1] = self.episode_id
            self.tracker.update_with_detection(cand)
            self.missing_count = 0
            self.last_reliable = t
            self.state = State.ACTIVE
            return

        # Still missing
        self.missing_count += 1
        self.presence[t] = 0

        if self.missing_count >= p.lost_timeout:
            self.episodes.append(
                self._make_episode(self.episode_start, self.last_reliable)
            )
            self._reset_to_idle()

    def _pred_reliable(self, pred_xy) -> bool:
        """True if Kalman prediction is trustworthy (inside ROI and low uncertainty)."""
        in_roi = point_in_poly(pred_xy, self.roi_poly) if self.roi_poly else True
        return in_roi and self.tracker.uncertainty() <= self.params.max_pos_uncertainty

    def _make_episode(self, start_frame: int, end_frame: int) -> Episode:
        start_sec = float(self.times_sec[start_frame])
        end_sec = float(self.times_sec[end_frame])
        return Episode(
            episode_id=self.episode_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            rot_seconds=end_sec - start_sec,
        )

    def _reset_to_idle(self) -> None:
        self.state = State.IDLE
        self.tracker = None
        self.missing_count = 0
        self.episode_start = None
        self.last_reliable = None
        self.start_window.clear()


def run_episode_state_machine(
    num_frames: int,
    detections: List[List[Detection]],
    times_sec: np.ndarray,
    roi_poly: Optional[Polygon],
    params,
) -> Dict[str, Any]:
    """Run the episode machine over all sampled frames."""
    machine = EpisodeMachine(num_frames, times_sec, roi_poly, params)

    with Progress(
        TextColumn("[green]{task.description}[/green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    ) as progress:
        task = progress.add_task("Processing frames", total=num_frames)
        for t in range(num_frames):
            machine.step(t, detections[t])
            progress.update(task, advance=1)

    machine.finalize()

    return {
        "presence": machine.presence,
        "episode_labels": machine.episode_labels,
        "episodes": machine.episodes,
    }
