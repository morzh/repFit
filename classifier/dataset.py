import threading

import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool


class SkeletonDataset(Dataset):
    """Skeleton joints and PCA"""

    def __init__(
            self,
            epoch_size: int = 100,
            batch_size: int = 1000
    ):

        self.epoch_size = epoch_size
        self.batch_size = batch_size

        # range from under zero values need for more zero
        self.sin_range = (-2, 5)
        self.amplitudes_range = (0, 15)
        self.noise_range = (0, 0.5)
        self.sin_power_range = (1, 2)

        self._x = []
        self._y = []
        self.n_threads = 10

    def __len__(self):
        return self.epoch_size

    def generate_batch(self, batch_size: int):
        sin_range = (np.random.rand(batch_size) * (self.sin_range[1] - abs(self.sin_range[0]))) + self.sin_range[0]
        const_idxs = sin_range < 0
        sin_range[const_idxs] = 0

        amplitudes = (np.random.rand(batch_size) * (self.amplitudes_range[1] - abs(self.amplitudes_range[0]))) + self.amplitudes_range[0]
        half_constants = np.random.randint(0, 2, size=const_idxs.size)
        half_constants[half_constants == 0] = -1
        amplitudes[const_idxs] *= half_constants[const_idxs]

        noise_lvl = np.random.rand(batch_size) * self.noise_range[1] * amplitudes
        noise_lvl[noise_lvl < 0.1] = 0

        power_range = np.ones(batch_size)
        n_powered = int(batch_size * 0.4)
        power_range[:n_powered] = np.random.randint(self.sin_power_range[0], self.sin_power_range[1], n_powered)
        power_range[power_range > 1] = 6

        periods = sin_range / self.discrete
        indexes = np.array([i for i in range(self._dimension)], dtype="float32")
        radians = periods[None].transpose() * indexes
        sins = np.sin(radians)
        sins_power = np.power(sins, power_range[None].transpose())
        sins_power[const_idxs, :] = 1
        signals = sins_power * amplitudes[None].transpose()

        bias = (np.random.rand(sum(~const_idxs)) - 0.5) * 10
        signals[~const_idxs] = signals[~const_idxs] + bias[None].transpose()

        y = []
        for i in range(batch_size):
            y.append(signals[i] + np.random.normal(0, noise_lvl[i], self._dimension))
        return signals, np.array(y, dtype="float32")

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.generate_batch(self.batch_size)
