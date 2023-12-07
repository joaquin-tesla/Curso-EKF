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
        
        M = Field.noise_from_motion(Field, u, self.alphas)

        eps = np.zeros((3,))
        eps[0] = G[1, 2]
        eps[1] = -G[0, 2]
        eps[2] = minimized_angle(u[0] + u[2])
        self.mu[0] = self.mu[0] + eps[0]
        self.mu[1] = self.mu[1] + eps[1]
        self.mu[2] = minimized_angle(self.mu[2] + eps[2])

        self.sigma = G @ self.sigma @ np.transpose(G) + V @ M @ np.transpose(V)

        q = (env.MARKER_X_POS[marker_id] - self.mu[0])**2 + (env.MARKER_Y_POS[marker_id] - self.mu[1])**2

        z_hat = minimized_angle(np.arctan2(env.MARKER_Y_POS[marker_id] - self.mu[1],
                                                 env.MARKER_X_POS[marker_id] - self.mu[0]) - self.mu[2])

        Ht = Field.H(env,x, marker_id)

        St = Ht @ self.sigma @ np.transpose(Ht) + self.beta
        Kt = self.sigma @ np.transpose(Ht) @ np.linalg.inv(St)

        self.mu = self.mu + Kt * minimized_angle(z - z_hat)
        self.sigma = (np.eye(len(self.mu)) - Kt @ Ht) @ self.sigma

        return self.mu, self.sigma

