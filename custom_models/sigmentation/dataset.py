import json
import pickle
import threading
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from custom_models.paths import PROJECT_ROOT, DATASETS_DPATH, RESULTS_DPATH
from pathlib import Path
from scipy.interpolate import interp2d


class SegmentationDataset(Dataset):
    """Skeleton joints and PCA"""

    def __init__(
        self,
        sample_length: int = 200,
        min_sample_length: int = 100,
        epoch_size: int = 100,
        batch_size: int = 1000
    ):
        self.pca_dpath = DATASETS_DPATH / "PCA_5.07.24" / "joints3d_pca"
        self.skeleton_dpath = DATASETS_DPATH / "PCA_5.07.24" / "results" / "joints3d"
        self.skeleton_info_dpath = DATASETS_DPATH / "PCA_5.07.24" / "results" / "joints2d_info"
        self.markup_fpath = PROJECT_ROOT / "markup" / "markup.json"

        self.sample_length = sample_length
        self.min_sample_length = min_sample_length
        self.epoch_size = epoch_size
        self.batch_size = batch_size

        self.dataset = self.load_data()

        self.speed_range = (0.8, 1.2)

        self.amplitudes_range = (0, 15)
        self.noise_range = (0, 0.5)
        self.sin_power_range = (1, 2)

        self.n_threads = 10

        # 17 points by 3 axis + pca + y
        self._sample_shape = np.zeros((17*3+2, self.sample_length))

    def __len__(self):
        return self.epoch_size

    def load_data(self) -> list:
        """
        :param
           files_list: files for witch need to load data

        Load data arrays (PCA + joints). Final result array contains:
            - first row: PCA vector
            - other rows: joints points vectors if format - x1, y1, z1, x2, y2, z2....
            - last row: y vector

        :return:
            list of 2d np.ndarray(float32) with different length
        """

        markup = self.load_markup()
        markup_files_list = list(markup.keys())
        original_data = []

        for pca_fpath in self.pca_dpath.glob("*.npy"):
            stem = pca_fpath.stem
            if pca_fpath.stem not in markup_files_list:
                continue
            pca_row = np.load(str(pca_fpath))
            joints = np.load(self.skeleton_dpath / pca_fpath.name)

            length = min(pca_row.shape[0], joints.shape[0])
            if length < self.min_sample_length:
                continue
            start_frame_idx, stop_frame_idx = self.read_frame_range(stem)
            print(f"{stem=}; {start_frame_idx=}; {stop_frame_idx=}; {pca_row.shape[0]=}; {joints.shape[0]=}")
            if start_frame_idx is None:
                continue

            # cut data to same length
            pca_row = pca_row[:length, ...]
            joints = joints[:length, ...]

            # flatten joint 3d to 2d shape
            joints = np.reshape(joints, (length, np.dot(*joints.shape[1:])))

            marks = self.move_markup(markup[stem], start_frame_idx)

            y = self.make_y_sample(joints.shape[0], marks)
            original_data.append(self.join_data_sample(pca_row, joints, y))
        return original_data

    def read_frame_range(self, stem: str) -> (int, int):
        # with open(self.skeleton_info_dpath / (stem + ".pickle"), 'rb') as file:
        #     joints_info = pickle.load(file)

        with open(self.skeleton_info_dpath / (stem + ".json"), 'r') as file:
            joints_info = json.load(file)

        if not joints_info:
            return None, None
        # skeleton_frames = list(list(joints_info.values())[0].keys())
        skeleton_frames = list(joints_info.keys())
        start_frame_idx = int(skeleton_frames[0])
        stop_frame_idx = int(skeleton_frames[-1])
        return start_frame_idx, stop_frame_idx

    def move_markup(self, marks: list, start_frame_idx: int):
        """
        Change coordinate indexes system from video frames to skeleton frames.
        Move point to left on index of first frame with skeleton.
        """
        marks = marks.copy()
        for mark in marks:
            for i in range(len(mark)):
                mark[i] -= start_frame_idx
        return marks

    def make_y_sample(self, y_length: int, marks: list):
        """
        Create vector with segmentation goal values.
        Put '1' for exist sample range and '0' for another positions.
        """

        y = np.zeros(y_length)
        for mark in marks:
            start = mark[0]
            for stop in mark[1:]:
                y[start: stop] = 1
                start = stop
        return y

    def join_data_sample(self, pca, joints, y):
        data_sample = np.hstack((pca, joints, y.reshape((len(y), 1))))
        data_sample = data_sample.transpose()
        return data_sample

    def load_markup(self):
        with open(self.markup_fpath, 'r') as file:
            markup = json.load(file)

        markup = {Path(path).stem: markup[path] for path in markup.keys() if markup[path]}
        return markup

    def generate_batch(self):
        """
        1. выделить диапазон с размером входа в модель
        2. аугментация:
            а. скорость - сжать или растянуть целиком  - done
            б. масштаб осей

        """
        x, y = [], []

        data_indexes = np.random.randint(0, len(self.dataset), self.batch_size)

        for idx in data_indexes:
            data_array = self.dataset[idx]
            sample = self.speed_augmentation(data_array)

            # TODO: add axis augmentation

            sample = self.cut_sample(sample)


            # cut sample by x and y parts
            x.append(sample[:-1, ...])
            y.append(sample[-1:, ...])

        return np.array(x, dtype="float32"), np.array(y, dtype="float32")

    def speed_augmentation(self, data_array):
        speed = np.random.uniform(*self.speed_range)
        y = np.arange(data_array.shape[0])
        x = np.arange(data_array.shape[1])
        x2 = np.arange(data_array.shape[1] * speed) * speed
        sample = interp2d(x, y, data_array, kind='cubic')(x2, y)

        return sample

    def cut_sample(self, input_array: np.ndarray) -> np.ndarray:
        """ Cut sample with shape self._sample_shape """
        max_position = input_array.shape[-1] - self.sample_length
        if max_position > 0:
            start_idx = np.random.randint(input_array.shape[-1] - self.sample_length)
        else:
            start_idx = 0

        sample = input_array[..., start_idx: start_idx + self.sample_length]

        if len(sample) < self.sample_length:
            tmp = self._sample_shape.copy()
            tmp[..., :sample.shape[-1]] = sample
            sample = tmp
        return sample

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.generate_batch()
