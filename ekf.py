""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np
from soccer_field import Field
from utils import minimized_angle

class ExtendedKalmanFilter:
    
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        x = self.mu
        G = Field.G(self, x, u)  
        V = Field.V(self, x, u)

        M = np.zeros((3, 3))
        M[0, 0] = self.alphas[0] * u[0]**2 + self.alphas[1] * u[1]**2
        M[2, 2] = self.alphas[0] * u[2]**2 + self.alphas[1] * u[1]**2
        M[1, 1] = self.alphas[2] * u[1]**2 + self.alphas[3] * (u[0]**2 + u[2]**2)

        eps = np.zeros((3,))
        eps[0,] = G[1, 2]
        eps[1,] = -G[0, 2]
        eps[2,] = minimized_angle(u[0] + u[2])
        self.mu[0] = self.mu[0] + eps[0,]
        self.mu[1] = self.mu[1] + eps[1,]
        self.mu[2] = minimized_angle(self.mu[2]+ eps[2,])

        self.sigma = G @ self.sigma @ G.T + V @ M @ V.T

        print(env.MARKER_X_POS[marker_id])
        q = (env.MARKER_X_POS[marker_id] - self.mu[0])**2 + (env.MARKER_Y_POS[marker_id] - self.mu[1])**2
        print(q)

        z_hat = np.zeros((2, 1))
        z_hat[0, 0] = np.sqrt(q)
        z_hat[1, 0] = minimized_angle(np.arctan2(env.MARKER_Y_POS[marker_id] - self.mu[1],
                                                 env.MARKER_X_POS[marker_id] - self.mu[0]) - self.mu[2])

        Ht = np.zeros((2, 3))
        Ht[0, 0] = -(env.MARKER_X_POS[marker_id] - self.mu[0]) / np.sqrt(q)
        Ht[0, 1] = -(env.MARKER_Y_POS[marker_id] - self.mu[1]) / np.sqrt(q)
        Ht[1, 0] = (env.MARKER_Y_POS[marker_id] - self.mu[1]) / q
        Ht[1, 1] = -(env.MARKER_X_POS[marker_id] - self.mu[0]) / q
        Ht[1, 2] = -1

        St = Ht @ self.sigma @ Ht.T + self.beta
        Kt = self.sigma @ Ht.T @ np.linalg.inv(St)

        self.mu = self.mu + Kt @ (z - z_hat)
        self.sigma = (np.eye(len(self.mu)) - Kt @ Ht) @ self.sigma

        return self.mu, self.sigma
