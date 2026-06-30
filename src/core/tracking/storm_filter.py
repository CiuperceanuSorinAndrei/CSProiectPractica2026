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

        dt = 1.0  # timp arbitrar = 1 frame
        gamma = 0.8  # Singer Damped Acceleration Model

        # State: [x, y, vx, vy, ax, ay, area, d_area, dd_area]
        self._kf.x = np.array([
            [initial_x], [initial_y],
            [initial_vx], [initial_vy],
            [0.0], [0.0],  # ax, ay
            [np.log(max(initial_area, 1.0))], [initial_d_area],
            [0.0]  # dd_area
        ])

        self._kf.F = self._build_transition_matrix(dt, gamma)
        self._kf.H = self._build_measurement_matrix()
        self._kf.P *= 10.0  # Covariance / Uncertainty
        self._kf.Q = self._build_process_noise(dt, gamma)
        self._kf.R = self._build_measurement_noise()

    @staticmethod
    def _build_transition_matrix(dt: float, gamma: float) -> np.ndarray:
        """Matricea de tranzitie F (Constant Acceleration + amortizare Singer pe acceleratii)."""
        return np.array([
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

    @staticmethod
    def _build_measurement_matrix() -> np.ndarray:
        """Matricea de observatie H - masuram direct x, y si log-aria."""
        return np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ])

    @staticmethod
    def _build_process_noise(dt: float, gamma: float) -> np.ndarray:
        """Matricea Q (zgomot de proces): bloc analitic exact Singer pe (pos, vel, acc) x2 + (area...)."""
        Q = np.zeros((9, 9))

        def add_singer_q_block(indices, var):
            if gamma >= 1.0:
                # Fallback la Constant Acceleration (White Noise)
                q = var * np.array([
                    [dt**5/20, dt**4/8, dt**3/6],
                    [dt**4/8,  dt**3/3, dt**2/2],
                    [dt**3/6,  dt**2/2, dt    ]
                ])
            else:
                alpha = -np.log(gamma) / dt
                a = alpha * dt
                sigma2 = var

                q11 = (sigma2 / alpha**5) * (1 - np.exp(-2*a) + 2*a + (2*a**3)/3 - 2*a**2 - 4*a*np.exp(-a))
                q12 = (sigma2 / alpha**4) * (np.exp(-2*a) + 1 - 2*np.exp(-a) + 2*a*np.exp(-a) - 2*a + a**2)
                q13 = (sigma2 / alpha**3) * (1 - np.exp(-2*a) - 2*a*np.exp(-a))
                q22 = (sigma2 / alpha**3) * (4*np.exp(-a) - 3 - np.exp(-2*a) + 2*a)
                q23 = (sigma2 / alpha**2) * (np.exp(-2*a) + 1 - 2*np.exp(-a))
                q33 = (sigma2 / alpha) * (1 - np.exp(-2*a))

                q = np.array([
                    [q11, q12, q13],
                    [q12, q22, q23],
                    [q13, q23, q33]
                ])

            for i in range(3):
                for j in range(3):
                    Q[indices[i], indices[j]] = q[i, j]

        add_singer_q_block([0, 2, 4], 0.05)
        add_singer_q_block([1, 3, 5], 0.05)
        add_singer_q_block([6, 7, 8], 0.01)  # log-space area noise
        return Q

    @staticmethod
    def _build_measurement_noise() -> np.ndarray:
        """Matricea R (zgomot de masura) pentru x, y si log-aria (scalata)."""
        return np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.2]  # scaled down for log-space
        ])

    def predict(self) -> None:
        self._kf.predict()
        self._kf.x[6, 0] = np.clip(self._kf.x[6, 0], -5.0, 20.0)

    def update(self, observed_x: float, observed_y: float, observed_area: float) -> None:
        obs_z = np.array([[observed_x], [observed_y], [np.log(max(observed_area, 1.0))]])
        
        try:
            self._kf.update(obs_z)
        except np.linalg.LinAlgError:
            # Fallback la modul predict-only dacă matricea inovației este singulară
            return
            
        self._kf.x[6, 0] = np.clip(self._kf.x[6, 0], -5.0, 20.0)
        
        # Joseph Form Covariance Update
        I = np.eye(self._kf.dim_x)
        K = self._kf.K
        H = self._kf.H
        I_KH = I - np.dot(K, H)
        self._kf.P = np.dot(np.dot(I_KH, self._kf.P), I_KH.T) + np.dot(np.dot(K, self._kf.R), K.T)
        
        # V28: PSD Forcing (Eigen-Decomposition) pentru stabilitate numerica
        self._kf.P = (self._kf.P + self._kf.P.T) / 2.0
        try:
            eigval, eigvec = np.linalg.eigh(self._kf.P)
            eigval = np.maximum(eigval, 1e-8)
            self._kf.P = (eigvec * eigval) @ eigvec.T
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

    @property
    def positional_uncertainty(self) -> float:
        """Trace-ul matricei de covarianta pentru coordonatele cinematice (x, y, vx, vy)."""
        return float(np.trace(self._kf.P[0:4, 0:4]))
