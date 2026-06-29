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
        self._kf = KalmanFilter(dim_x=9, dim_z=3)
        
        # State: [x, y, vx, vy, ax, ay, area, d_area, dd_area]
        self._kf.x = np.array([
            [initial_x], [initial_y],
            [initial_vx], [initial_vy],
            [0.0], [0.0],  # ax, ay
            [np.log(max(initial_area, 1.0))], [initial_d_area],
            [0.0]  # dd_area
        ])
        
        dt = 1.0  # timp arbitrar = 1 frame
        gamma = 0.8  # Singer Damped Acceleration Model
        
        # Transition Matrix (F)
        self._kf.F = np.array([
            [1.0, 0.0,  dt, 0.0, 0.5*dt**2, 0.0,       0.0, 0.0,       0.0], # x
            [0.0, 1.0, 0.0,  dt, 0.0,       0.5*dt**2, 0.0, 0.0,       0.0], # y
            [0.0, 0.0, 1.0, 0.0,  dt,       0.0,       0.0, 0.0,       0.0], # vx
            [0.0, 0.0, 0.0, 1.0, 0.0,        dt,       0.0, 0.0,       0.0], # vy
            [0.0, 0.0, 0.0, 0.0, gamma,     0.0,       0.0, 0.0,       0.0], # ax
            [0.0, 0.0, 0.0, 0.0, 0.0,       gamma,     0.0, 0.0,       0.0], # ay
            [0.0, 0.0, 0.0, 0.0, 0.0,       0.0,       1.0,  dt, 0.5*dt**2], # area
            [0.0, 0.0, 0.0, 0.0, 0.0,       0.0,       0.0, 1.0,        dt], # d_area
            [0.0, 0.0, 0.0, 0.0, 0.0,       0.0,       0.0, 0.0,     gamma], # dd_area
        ])
        
        # Observation Matrix (H) - we observe x, y, area
        self._kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ])
        
        # Covariance / Uncertainty
        self._kf.P *= 10.0
        
        # Process Noise (Analytical Block Approximation)
        self._kf.Q = np.zeros((9, 9))
        
        def add_q_block(indices, dt, var):
            q = var * np.array([
                [dt**5/20, dt**4/8, dt**3/6],
                [dt**4/8,  dt**3/3, dt**2/2],
                [dt**3/6,  dt**2/2, dt    ]
            ])
            for i in range(3):
                for j in range(3):
                    self._kf.Q[indices[i], indices[j]] = q[i, j]

        add_q_block([0, 2, 4], dt, 0.05)
        add_q_block([1, 3, 5], dt, 0.05)
        add_q_block([6, 7, 8], dt, 0.01) # log-space area noise
        
        # Measurement Noise
        self._kf.R = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.2]  # scaled down for log-space
        ])

    def predict(self) -> None:
        self._kf.predict()

    def update(self, observed_x: float, observed_y: float, observed_area: float) -> None:
        obs_z = np.array([[observed_x], [observed_y], [np.log(max(observed_area, 1.0))]])
        self._kf.update(obs_z)
        
        # Joseph Form Covariance Update
        I = np.eye(self._kf.dim_x)
        K = self._kf.K
        H = self._kf.H
        I_KH = I - np.dot(K, H)
        self._kf.P = np.dot(np.dot(I_KH, self._kf.P), I_KH.T) + np.dot(np.dot(K, self._kf.R), K.T)

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
        return np.exp(self._kf.x[6, 0])

    @property
    def d_area(self) -> float:
        # Rate of change in real area: d(exp(L))/dt = exp(L) * dL/dt
        return np.exp(self._kf.x[6, 0]) * self._kf.x[7, 0]

    @property
    def dd_area(self) -> float:
        return np.exp(self._kf.x[6, 0]) * (self._kf.x[7, 0]**2 + self._kf.x[8, 0])
