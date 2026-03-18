from __future__ import annotations

import numpy as np
from air_track.tracking.geometry import box_center, Detection, Point


class SimpleKalmanCV:
    """
    Constant-velocity Kalman filter tracking the center of a bounding box.

    State vector : [x, y, vx, vy]
    Measurement  : [x, y]  (box center)
    """

    def __init__(
        self,
        init_xy: Point,
        dt: float = 1.0,
        process_var: float = 50.0,
        meas_var: float = 25.0,
        init_vel: Point = (0.0, 0.0),
    ) -> None:
        self.dt = dt
        x0, y0 = init_xy
        vx0, vy0 = init_vel

        # State: [x, y, vx, vy]
        self.x = np.array([[x0], [y0], [vx0], [vy0]], dtype=float)

        # State transition matrix
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )

        # Measurement matrix (observe x, y only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # Process noise covariance Q
        q = process_var
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=float) * meas_var

        # Covariance P — large initial velocity uncertainty
        self.P = np.diag([meas_var, meas_var, 1000.0, 1000.0]).astype(float)
        self._I = np.eye(4, dtype=float)

    def predict(self) -> Point:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, meas_xy: Point) -> Point:
        z = np.array([[meas_xy[0]], [meas_xy[1]]], dtype=float)
        y_innov = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y_innov  # update state
        self.P = (self._I - K @ self.H) @ self.P  # update covariance
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def pos_uncertainty(self) -> float:
        """Scalar position uncertainty (trace of position covariance)."""
        return float(self.P[0, 0] + self.P[1, 1])


class SingleObjectTracker:
    """
    Wraps SimpleKalmanCV with detection association for a single object.
    One instance per active episode.
    """

    def __init__(
        self,
        init_det: Detection,
        dt: float = 1.0,
        process_var: float = 50.0,
        meas_var: float = 25.0,
    ) -> None:
        box, conf = init_det
        cx, cy = box_center(box)
        self.kf = SimpleKalmanCV(
            (cx, cy), dt=dt, process_var=process_var, meas_var=meas_var
        )
        self.last_conf = conf
        self.last_box = box

    def predict(self) -> Point:
        return self.kf.predict()

    def update_with_detection(self, det: Detection) -> Point:
        box, conf = det
        cx, cy = box_center(box)
        pos = self.kf.update((cx, cy))
        self.last_conf = conf
        self.last_box = box
        return pos

    def uncertainty(self) -> float:
        return self.kf.pos_uncertainty()
