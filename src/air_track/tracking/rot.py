from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import numpy as np

from air_track.tracking.geometry import Polygon, Zone, ref_point, zone_of
from air_track.tracking.tracker import Track

# (starts airborne, ends airborne) -> action type
_ACTIONS = {
    (True, False): "landing",
    (False, True): "take-off",
    (True, True): "touch-and-go",
    (False, False): "ground",
}


@dataclass
class Episode:
    """One aircraft occupying the runway."""

    episode_id: int
    track_id: int
    action_type: str
    start_sec: float
    end_sec: float

    @property
    def rot_seconds(self) -> float:
        return self.end_sec - self.start_sec


class _Run(NamedTuple):
    zone: Zone
    first: int  # observation index
    last: int


def _runs(zones: List[Zone], min_len: int) -> List[_Run]:
    """Group equal consecutive zones, drop runs shorter than min_len, re-merge."""
    raw: List[_Run] = []
    for i, z in enumerate(zones):
        if raw and raw[-1].zone == z:
            raw[-1] = raw[-1]._replace(last=i)
        else:
            raw.append(_Run(z, i, i))

    runs: List[_Run] = []
    for r in raw:
        if r.last - r.first + 1 < min_len:
            continue
        if runs and runs[-1].zone == r.zone:
            runs[-1] = runs[-1]._replace(last=r.last)
        else:
            runs.append(r)
    return runs


def _bridge_off_gaps(
    zones: List[Zone],
    runs: List[_Run],
    obs_times: np.ndarray,
    max_gap_sec: float,
) -> List[Zone]:
    """Relabel short OFF gaps between runway runs with the zone before them."""
    zones = list(zones)
    for prev, run, nxt in zip(runs, runs[1:], runs[2:]):
        gap_sec = obs_times[nxt.first] - obs_times[prev.last]
        if run.zone == Zone.OFF and gap_sec <= max_gap_sec:
            for i in range(prev.last + 1, nxt.first):
                zones[i] = prev.zone
    return zones


def _runway_chunks(runs: List[_Run]) -> List[List[_Run]]:
    """Split runs at OFF; keep chunks where the aircraft touched the runway."""
    chunks, cur = [], []
    for r in runs + [_Run(Zone.OFF, -1, -1)]:
        if r.zone != Zone.OFF:
            cur.append(r)
            continue
        if any(c.zone == Zone.GROUND for c in cur):
            chunks.append(cur)
        cur = []
    return chunks


def _is_moving(track: Track, min_frames: int, min_disp_px: float) -> bool:
    """Reject short tracks and static ones (parked aircraft, false positives)."""
    if len(track) < min_frames:
        return False
    centers = np.array([ref_point(box, "center") for _, box in track])
    return np.linalg.norm(centers.max(0) - centers.min(0)) >= min_disp_px


def extract_episodes(
    tracks: Dict[int, Track],
    times_sec: np.ndarray,
    roi_poly: Polygon,
    params,
) -> List[Episode]:
    """Turn tracks into runway-occupancy episodes.

    Within each AIR/GROUND chunk of a track:
      start = first frame of the chunk (threshold crossing if airborne,
              runway entry if on the ground)
      end   = first frame of the final AIR run (liftoff) if it ends airborne,
              otherwise last GROUND frame (runway exit)
    """
    episodes: List[Episode] = []
    for track_id, track in tracks.items():
        if not _is_moving(
            track, params.min_track_frames, params.min_displacement_px
        ):
            continue

        # Air zone scales with the aircraft's own box height (perspective)
        zones = [
            zone_of(ref_point(box, params.ref_point), roi_poly,
                    params.air_height_ratio * (box[3] - box[1]))
            for _, box in track
        ]
        obs_times = times_sec[[frame for frame, _ in track]]
        zones = _bridge_off_gaps(
            zones,
            _runs(zones, params.min_run_frames),
            obs_times,
            params.max_off_gap_sec,
        )
        for chunk in _runway_chunks(_runs(zones, params.min_run_frames)):
            first, last = chunk[0], chunk[-1]
            starts_air = first.zone == Zone.AIR
            ends_air = last.zone == Zone.AIR
            end_idx = last.first if ends_air else last.last
            episodes.append(
                Episode(
                    episode_id=0,
                    track_id=track_id,
                    action_type=_ACTIONS[(starts_air, ends_air)],
                    start_sec=float(times_sec[track[first.first][0]]),
                    end_sec=float(times_sec[track[end_idx][0]]),
                )
            )

    episodes.sort(key=lambda e: e.start_sec)
    for i, ep in enumerate(episodes, start=1):
        ep.episode_id = i
    return episodes
