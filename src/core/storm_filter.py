"""StormFilter: Starea predictiva Kalman a unei furtuni.

Implementeaza un model Constant Acceleration (CA) cu stare 8D:
[x, y, vx, vy, ax, ay, area, d_area]
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class StormFilter:
    """Incapsuleaza filtrul Kalman (Constant Acceleration) pentru urmarirea celulelor."""
    
    def __init__(
        self,
        initial_y: float, initial_x: float,
        initial_vy: float = 0.0, initial_vx: float = 0.0,
        initial_area: float = 1.0, initial_d_area: float = 0.0
    ):
        self._kf = KalmanFilter(dim_x=8, dim_z=3)
        
        # State: [x, y, vx, vy, ax, ay, area, d_area]
        self._kf.x = np.array([
            [initial_x], [initial_y],
            [initial_vx], [initial_vy],
            [0.0], [0.0],  # ax, ay
            [initial_area], [initial_d_area]
        ])
        
        dt = 1.0  # timp arbitrar = 1 frame
        
        # Transition Matrix (F)
        # x = x + vx*dt + 0.5*ax*dt^2
        self._kf.F = np.array([
            [1.0, 0.0,  dt, 0.0, 0.5*dt**2, 0.0,       0.0, 0.0], # x
            [0.0, 1.0, 0.0,  dt, 0.0,       0.5*dt**2, 0.0, 0.0], # y
            [0.0, 0.0, 1.0, 0.0,  dt,       0.0,       0.0, 0.0], # vx
            [0.0, 0.0, 0.0, 1.0, 0.0,        dt,       0.0, 0.0], # vy
            [0.0, 0.0, 0.0, 0.0, 1.0,       0.0,       0.0, 0.0], # ax
            [0.0, 0.0, 0.0, 0.0, 0.0,       1.0,       0.0, 0.0], # ay
            [0.0, 0.0, 0.0, 0.0, 0.0,       0.0,       1.0,  dt], # area
            [0.0, 0.0, 0.0, 0.0, 0.0,       0.0,       0.0, 1.0], # d_area
        ])
        
        # Observation Matrix (H) - we observe x, y, area
        self._kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        
        # Covariance / Uncertainty
        self._kf.P *= 10.0
        
        # Process Noise
        self._kf.Q = np.eye(8) * 0.1
        self._kf.Q[4, 4] = 0.05  # ax noise (small, acceleration changes slowly)
        self._kf.Q[5, 5] = 0.05  # ay noise
        self._kf.Q[6, 6] = 2.0   # area noise
        self._kf.Q[7, 7] = 0.5   # d_area noise
        
        # Measurement Noise
        self._kf.R = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 20.0]  # higher uncertainty in measured area
        ])

    def predict(self) -> None:
        self._kf.predict()

    def update(self, observed_x: float, observed_y: float, observed_area: float) -> None:
        obs_z = np.array([[observed_x], [observed_y], [float(observed_area)]])
        self._kf.update(obs_z)

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
    def a_x(self) -> float:
        return self._kf.x[4, 0]

    @property
    def a_y(self) -> float:
        return self._kf.x[5, 0]

    @property
    def area(self) -> float:
        return max(1.0, self._kf.x[6, 0])

    @property
    def d_area(self) -> float:
        return self._kf.x[7, 0]
