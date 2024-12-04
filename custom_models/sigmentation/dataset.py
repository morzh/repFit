from typing import Dict, List
import json
import numpy as np
from torch.utils.data import Dataset
from custom_models.paths import PROJECT_ROOT, DATASETS_DPATH
from pathlib import Path
from scipy.interpolate import interp2d

min_distance_between_samples_frames: int = 3 # cut from both sides, real gab will be *2
sample_length: int = 200 # length of one data sample in frames. Calc it as fpt*seconds

# 17 points by 3 axis + pca + y
sample_shape = np.zeros((17 * 3 + 2, sample_length))


def cut_continuous_mark(sample: list, gap_distance: int) -> List[tuple]:
    points = []
    start = sample[0]
    for point in sample[1:]:
        points.append((start, point - gap_distance))
        start = point + gap_distance
    return points


class SegmentationDataset(Dataset):
    """Skeleton joints and PCA"""

    def __init__(
        self,
        dpath=DATASETS_DPATH / "train_mix_squad",
        min_sample_length: int = 100,
        epoch_size: int = 100,
        batch_size: int = 1000
    ):

        self.pca_dpath = dpath / "joints3d_pca"
        self.skeleton_dpath = dpath / "joints3d_aligned_to_global_frame"
        self.skeleton_info_dpath = dpath / "joints2d_info"
        self.markup_fpath = dpath / "markup.json"

        self.sample_length = sample_length
        self.min_sample_length = min_sample_length
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.speed_range = (0.8, 1.2)
        self.stretch_by_axis_range = (0.8, 1.2)
        self.n_threads = 10
        self.remove_class_labels = [7]
 

        self.dataset = self.load_data()

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
            print(f"{stem=}; {pca_row.shape[0]=}; {joints.shape[0]=}")

            # flatten joint 3d to 2d shape
            joints = np.reshape(joints, (length, np.dot(*joints.shape[1:])))

            y = self.make_y_sample(joints.shape[0], markup[stem])
            original_data.append(self.join_data_sample(pca_row, joints, y))
        return original_data

    def read_frame_range(self, stem: str) -> (int, int):
        # with open(self.skeleton_info_dpath / (stem + ".pickle"), 'rb') as file:
        #     joints_info = pickle.load(file)

        with open(self.skeleton_info_dpath / (stem + ".json"), 'r') as file:
            joints_info = json.load(file)

        if not joints_info:
            return None, None
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

    def make_y_sample(self, y_length: int, marks: List) -> np.ndarray:
        """
        Create vector with segmentation goal values.
        Put '1' for exist sample range and '0' for another positions.
        """

        y = np.zeros(y_length)
        for label, (start, stop) in marks:
            y[start: stop] = 1
        return y

    def join_data_sample(self, pca, joints, y):
        data_sample = np.hstack((pca, joints, y.reshape((len(y), 1))))
        data_sample = data_sample.transpose()
        return data_sample

    def load_markup(self) -> Dict[str, list]:
        with open(self.markup_fpath, 'r') as file:
            markup = json.load(file)

        markup = {Path(path).stem: markup[path] for path in markup.keys() if markup[path]}
        markup = self.remove_classes(markup)
        markup = self.cut_continuous_samples(markup)
        return markup

    def remove_classes(self, markup: Dict[str, list], labels: list = None) -> Dict[str, list]:
        """ Remove samples of classes in labels. """

        if labels is None:
            labels = self.remove_class_labels

        for fname, points in markup.items():
            clear_points = []
            for label, sample in points:
                if label not in labels:
                    clear_points.append((label, sample))
            markup[fname] = clear_points
        return markup

    def cut_continuous_samples(self, markup: Dict[str, list]) -> Dict[str, list]:
        """ Cut samples without gaps (20,31,40,52...) on individual start/stop range. """

        for fname, points in markup.items():
            clear_points = []
            for label, sample in points:
                if len(sample) != 2:
                    for sub_sample in cut_continuous_mark(sample, min_distance_between_samples_frames):
                        clear_points.append((label, sub_sample))
                else:
                    clear_points.append((label, sample))
            markup[fname] = clear_points
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
            sample = self.stretch_by_axis(sample)
            sample = self.cut_sample(sample)

            # cut sample by x and y parts
            x.append(sample[:-1, ...])
            y.append(sample[-1:, ...])

        return np.array(x, dtype="float32"), np.array(y, dtype="float32")

    def speed_augmentation(self, data_array: np.ndarray):
        speed = np.random.uniform(*self.speed_range)
        y = np.arange(data_array.shape[0])
        x = np.arange(data_array.shape[1])
        x2 = np.arange(data_array.shape[1] * speed) * speed
        sample = interp2d(x, y, data_array, kind='cubic')(x2, y)

        return sample

    def stretch_by_axis(self, data_array: np.ndarray):
        data_array = np.copy(data_array)
        for i, k in enumerate(np.random.uniform(*self.stretch_by_axis_range, size=3)):
            data_array[1+i:-1:3, :] = data_array[1+i:-1:3, :] * k
        return data_array

    def cut_sample(self, input_array: np.ndarray) -> np.ndarray:
        """ Cut sample with shape sample_shape from random position inside the input_array """
        max_position = input_array.shape[-1] - self.sample_length
        if max_position > 0:
            start_idx = np.random.randint(input_array.shape[-1] - self.sample_length)
        else:
            start_idx = 0

        sample = input_array[..., start_idx: start_idx + self.sample_length]

        if len(sample) < self.sample_length:
            tmp = sample_shape.copy()
            tmp[..., :sample.shape[-1]] = sample
            sample = tmp
        return sample

    def __iter__(self):
        for i in range(self.epoch_size):
            yield self.generate_batch()


class SegmentationDatasetValidation(Dataset):
    """Skeleton joints and PCA"""

    def __init__(self, sliding_window_length: int = 10):
        assert sliding_window_length < sample_length
        self.pca_dpath = DATASETS_DPATH / "PCA_5.07.24" / "joints3d_pca"
        self.skeleton_dpath = DATASETS_DPATH / "PCA_5.07.24" / "joints3d"
        self.skeleton_info_dpath = DATASETS_DPATH / "PCA_5.07.24" / "results" / "joints2d_info"
        self.markup_fpath = PROJECT_ROOT / "markup" / "markup.json"

        self.sample_length = sample_length
        self.epoch_size = 1
        self.batch_size = 1
        self.sliding_window_length = sliding_window_length
        self.speed_range = (0.8, 1.2)
        self.stretch_by_axis_range = (0.8, 1.2)

        self.dataset = self.load_data()
        self.board_size = self.sample_length - self.sliding_window_length
        self._sample_length = None

        self.sum_k = self.sample_length / self.sliding_window_length # how much point were sum in each position


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
        # start in position:  -self.sample_length + self.sliding_window_length
        # because we need one number of sum values in each position
        
        for pca_fpath in self.pca_dpath.glob("*.npy"):
            stem = pca_fpath.stem
            if pca_fpath.stem not in markup_files_list:
                continue
            pca_row = np.load(str(pca_fpath))
            joints = np.load(self.skeleton_dpath / pca_fpath.name)
            assert len(pca_row) == len(joints), "Something wrong with data. PCA and joints have different length"
            length = min(pca_row.shape[0], joints.shape[0])
            if length < self.sample_length:
                # TODO: add zeros
                continue
            print(f"{stem=}; {pca_row.shape[0]=}; {joints.shape[0]=}")

            # flatten joint 3d to 2d shape
            joints = np.reshape(joints, (length, np.dot(*joints.shape[1:])))

            y = self.make_y_sample(joints.shape[0], markup[stem])
            data_sample = np.hstack((pca_row, joints, y.reshape((len(y), 1))))
            data_sample = data_sample.transpose()
            original_data.append(data_sample)
        return original_data

    def make_y_sample(self, y_length: int, marks: List) -> np.ndarray:
        """
        Create vector with segmentation goal values.
        Put '1' for exist sample range and '0' for another positions.
        """

        y = np.zeros(y_length)
        for mark in marks:
            if len(mark)!=2:
                mark = cut_continuous_mark(mark, min_distance_between_samples_frames)
                for (start, stop) in mark:
                    y[start: stop] = 1
            else:
                y[mark[0]: mark[1]] = 1
        return y

    def load_markup(self):
        with open(self.markup_fpath, 'r') as file:
            markup = json.load(file)

        markup = {Path(path).stem: markup[path] for path in markup.keys() if markup[path]}
        return markup

    def generate_batch(self, idx: int):
        """
        1. выделить диапазон с размером входа в модель
        2. аугментация:
            а. скорость - сжать или растянуть целиком  - done
            б. масштаб осей

        """
        data_array = self.dataset[idx]
        array = self.speed_augmentation(data_array)
        array = self.stretch_by_axis(array)
        
        x, y = self.cut_samples(array)

        return np.array(x, dtype="float32"), np.array(y, dtype="float32")

    def speed_augmentation(self, data_array: np.ndarray):
        speed = np.random.uniform(*self.speed_range)
        y = np.arange(data_array.shape[0])
        x = np.arange(data_array.shape[1])
        x2 = np.arange(data_array.shape[1] * speed) * speed
        sample = interp2d(x, y, data_array, kind='cubic')(x2, y)

        return sample

    def stretch_by_axis(self, data_array: np.ndarray):
        data_array = np.copy(data_array)
        for i, k in enumerate(np.random.uniform(*self.stretch_by_axis_range, size=3)):
            data_array[1+i:-1:3, :] = data_array[1+i:-1:3, :] * k
        return data_array

    def cut_samples(self, input_array: np.ndarray) -> (list, list):
        """ Cut full video by parts with slicing window """
        x, y = [], []

        boarder_array = np.zeros(shape=(sample_shape.shape[0], self.board_size))
        input_array = np.hstack((boarder_array, input_array, boarder_array))
        self._sample_length = input_array.shape[-1]
        for start_idx in range(0, input_array.shape[-1]-self.sample_length, self.sliding_window_length):
            sample = input_array[..., start_idx: start_idx + self.sample_length]

            x.append(sample[:-1, ...])
            y.append(sample[-1:, ...])
        return x, y

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.generate_batch(i)

    def join_results(self, y_train: np.ndarray, y_predicted: np.ndarray) -> (np.ndarray, np.ndarray):
        """ Join split by sample_length samples to original row """
        y_train = y_train.squeeze()
        y_predicted = y_predicted.squeeze()
        y_train_array = np.zeros((self._sample_length))
        y_predicted_array = y_train_array.copy()
        stop_range = self._sample_length-self.sample_length
        for i, start in enumerate(range(0, stop_range, self.sliding_window_length)):
            y_train_array[start: self.sample_length+start] += y_train[i]
            y_predicted_array[start: self.sample_length+start] += y_predicted[i]

        y_train_array = y_train_array/self.sum_k
        y_predicted_array = y_predicted_array/self.sum_k
        return y_train_array, y_predicted_array