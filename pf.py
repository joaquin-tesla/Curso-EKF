from itertools import accumulate
from re import M
import numpy as np

from utils import minimized_angle
from soccer_field import Field
from utils import minimized_angle

class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self.soccer = Field(alphas, beta)
        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """

        theta_p = np.zeros((self.num_particles, 1))
        
        for i in range(self.num_particles):
            self.particles[i,:] = np.transpose(self.soccer.forward(self.particles[i,:], self.soccer.sample_noisy_action(u,self.alphas)))
            theta_p[i, 0] = self.soccer.sample_noisy_observation(self.particles[i, :], marker_id, self.beta)
            self.weights[i] = Field.likelihood(Field, ( theta_p[i] - z), self.beta)

        normal = np.sum(self.weights)
        for k in range(self.num_particles):
           self.weights[k] = self.weights[k]/normal
        
    
        self.particles, self.weights = self.resample(self.particles, self.weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        p = self.num_particles
        c = weights[0]
        new_particles = np.zeros((p, 3))  
        n = np.random.uniform(0, 1 / p)
        new_weights = np.zeros(p)
        i = 0
        for j in range(p):
            U = n + j * 1 / p
            while (U > c):
                i = i + 1
                c = c + weights[i]
            new_particles[j] = self.particles[i]
            new_weights[j] = weights[i]

        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )
        
        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
