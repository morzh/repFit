import json
import threading

import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from paths import PROJECT_ROOT, DATASETS_DPATH, RESULTS_DPATH
from pathlib import Path


class SkeletonDataset(Dataset):
    """Skeleton joints and PCA"""

    def __init__(
        self,
        epoch_size: int = 100,
        batch_size: int = 1000
    ):
        self.pca_dpath = DATASETS_DPATH / "PCA_5.07.24" / "joints3d_pca"
        self.skeleton_dpath = DATASETS_DPATH / "PCA_5.07.24" / "results" / "joints3d"
        self.markup_fpath = PROJECT_ROOT / "markup" / "markup.json"

        self.min_data_length = 100

        markup = self.load_markup()
        original_data = self.load_data(list(markup.keys()))
        self.dataset = self.cut_parts(original_data, markup)
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

    def load_data(self, files_list: list) -> dict:
        """
        :param
           files_list: files for witch need to load data

        Load data arrays (PCA + joints). Final result array contains:
            - first row: PCA vector
            - other rows: joints points vectors if format - x1, y1, z1, x2, y2, z2....

        :return:
            list of 2d np.ndarray(float32) with different length
        """

        original_data = {}
        for pca_fpath in self.pca_dpath.glob("*.npy"):
            stem = pca_fpath.stem
            if pca_fpath.stem not in files_list:
                continue
            pca_row = np.load(str(pca_fpath))
            skeleton_array = np.load(self.skeleton_dpath / pca_fpath.name)

            length = min(pca_row.shape[0], skeleton_array.shape[0])
            if length < self.min_data_length:
                continue

            pca_row = pca_row[:length, ...]
            skeleton_array = skeleton_array[:length, ...]

            skeleton_array = np.reshape(skeleton_array, (length, np.dot(*skeleton_array.shape[1:])))
            data_sample = np.hstack((pca_row, skeleton_array))
            data_sample = data_sample.transpose()
            original_data[stem] = data_sample
        return original_data

    def load_markup(self):
        with open(self.markup_fpath, 'r') as file:
            markup = json.load(file)

        markup = {Path(path).stem:markup[path]  for path in markup.keys() if markup[path]}
        return markup

    def cut_parts(self, data: dict, markup: dict) -> list:
        """
        :param data: list of np arrays
        :param markup: dict
        :return: list of np arrays
        """
        dataset = []
        for stem, marks in markup.items():
            if stem not in data:
                continue
            for mark in marks:
                if len(mark) > 2:
                    # if range do not contain gaps between exercise repeats
                    start = mark[0]
                    for stop in mark[1:]:
                        part = data[stem][..., start: stop]
                        if part.size == 0:
                            r=0
                        dataset.append(data[stem][..., start: stop])
                        start = stop
                else:
                    start, stop = mark
                    part = data[stem][..., start: stop]
                    if part.size == 0:
                        #  DOTO: Check the reason
                        r = 0
                    dataset.append(data[stem][..., start: stop])
        return dataset

    def generate_batch(self):
        """
        1. выделить диапазон с размером входа в модель
        2. аугментация:
            а. скорость - сжать или растянуть целиком
            б. масштаб осей


        """



        sin_range = (np.random.rand(self.batch_size) * (self.sin_range[1] - abs(self.sin_range[0]))) + self.sin_range[0]
        const_idxs = sin_range < 0
        sin_range[const_idxs] = 0

        amplitudes = (np.random.rand(self.batch_size) * (self.amplitudes_range[1] - abs(self.amplitudes_range[0]))) + self.amplitudes_range[0]
        half_constants = np.random.randint(0, 2, size=const_idxs.size)
        half_constants[half_constants == 0] = -1
        amplitudes[const_idxs] *= half_constants[const_idxs]

        noise_lvl = np.random.rand(self.batch_size) * self.noise_range[1] * amplitudes
        noise_lvl[noise_lvl < 0.1] = 0

        power_range = np.ones(self.batch_size)
        n_powered = int(self.batch_size * 0.4)
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
        for i in range(self.batch_size):
            y.append(signals[i] + np.random.normal(0, noise_lvl[i], self._dimension))
        return signals, np.array(y, dtype="float32")

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.generate_batch()
