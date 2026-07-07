"""StormFilter: Kalman predictive state for a storm.

Implements a Constant Velocity (CV) model with a 4D state:
[x, y, vx, vy]
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class StormFilter:
    """Encapsulates the Kalman filter (Constant Velocity) for tracking cells."""
    
    def __init__(
        self,
        initial_y: float, initial_x: float,
        initial_vy: float = 0.0, initial_vx: float = 0.0,
        initial_area: float = 1.0, initial_d_area: float = 0.0
    ):
        # Area parameters kept for backwards compatibility but ignored
        self._kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0  # timp arbitrar = 1 frame

        # State: [x, y, vx, vy]
        self._kf.x = np.array([
            [initial_x], [initial_y],
            [initial_vx], [initial_vy]
        ])

        self._kf.F = self._build_transition_matrix(dt)
        self._kf.H = self._build_measurement_matrix()
        self._kf.P *= 10.0  # Covariance / Uncertainty
        self._kf.Q = self._build_process_noise(dt)
        self._kf.R = self._build_measurement_noise()

    @staticmethod
    def _build_transition_matrix(dt: float) -> np.ndarray:
        """Transition matrix F (Constant Velocity Model)."""
        return np.array([
            [1.0, 0.0,  dt, 0.0], # x
            [0.0, 1.0, 0.0,  dt], # y
            [0.0, 0.0, 1.0, 0.0], # vx
            [0.0, 0.0, 0.0, 1.0], # vy
        ])

    @staticmethod
    def _build_measurement_matrix() -> np.ndarray:
        """Observation matrix H - measuring x, y directly."""
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

    @staticmethod
    def _build_process_noise(dt: float) -> np.ndarray:
        """Process noise matrix Q for Constant Velocity."""
        var = 0.1
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = (dt**4 / 4.0) * var
        Q[0, 2] = Q[1, 3] = Q[2, 0] = Q[3, 1] = (dt**3 / 2.0) * var
        Q[2, 2] = Q[3, 3] = (dt**2) * var
        return Q

    @staticmethod
    def _build_measurement_noise() -> np.ndarray:
        """Measurement noise matrix R for x, y."""
        return np.array([
            [10.0, 0.0],
            [0.0, 10.0]
        ])

    def predict(self) -> None:
        self._kf.predict()
        try:
            eigval, eigvec = np.linalg.eigh(self._kf.P)
            self._kf.P = (eigvec * np.clip(eigval, 1e-8, 50.0)) @ eigvec.T
        except np.linalg.LinAlgError:
            pass

    def update(self, observed_x: float, observed_y: float) -> None:
        obs_z = np.array([[observed_x], [observed_y]])
        
        # Save prior covariance for Joseph Form
        P_prior = self._kf.P_prior.copy() if hasattr(self._kf, 'P_prior') else self._kf.P.copy()
        
        try:
            self._kf.update(obs_z)
        except np.linalg.LinAlgError:
            return
            
        # Joseph Form Covariance Update
        I = np.eye(self._kf.dim_x)
        K = self._kf.K
        H = self._kf.H
        I_KH = I - np.dot(K, H)
        P_new = np.dot(np.dot(I_KH, P_prior), I_KH.T) + np.dot(np.dot(K, self._kf.R), K.T)
        self._kf.P = (P_new + P_new.T) / 2.0
        try:
            eigval, eigvec = np.linalg.eigh(self._kf.P)
            self._kf.P = (eigvec * np.maximum(eigval, 1e-8)) @ eigvec.T
        except np.linalg.LinAlgError:
            pass

    @property
    def x(self) -> float:
        return self._kf.x[0, 0]

    @property
    def y(self) -> float:
        return self._kf.x[1, 0]

    @property
    def v_x(self) -> float:
        return self._kf.x[2, 0]

    @property
    def v_y(self) -> float:
        return self._kf.x[3, 0]

    @property
    def positional_uncertainty(self) -> float:
        """Trace of covariance matrix for kinematic coordinates (x, y, vx, vy)."""
        return float(np.trace(self._kf.P[0:4, 0:4]))
