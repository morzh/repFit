import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]


class Human36mStatistics:
    r"""
    Description:
        HumanH3.6M statistics for animated skeletons processing. Some terminology:

        #. We consider H3.6M skeleton is an [17, 3] array, In other words, H3.6M skeleton is a set of 3D joints;
        #. H3.6M animated skeleton (joints batch) is [N, 17, 3] array, N - number of frames;
        #. Skeleton's normal vector is the 3rd vector of the root joint coordinate frame (see calculate_skeletons_root_frame());
        #. Root joint is a joint with index zero (joints[0])
        #. Mean value is calculated across set of objects (e.g. spines, legs, etc.)
    """

    @staticmethod
    def mean_spine_length(animated_skeleton: joints_batch) -> float:
        r"""
        Description:
            For Human3.6M dataset let's call spine a chain of joints with indices [0, 7, 8].
            Spine's length defined as:  :math:`S = \| joint_0 - joint_7 \| + \| joint_7 - joint_8 \|`.

        :param animated_skeleton: batch of joints coordinates
        :return: mean of spine lengths across joints batch
        """
        spine_lengths = Human36mStatistics.spine_lengths(animated_skeleton)
        average_spine_length = np.mean(spine_lengths)
        return average_spine_length

    @staticmethod
    def spine_lengths(animated_skeleton: joints_batch) -> np.ndarray:
        r"""
        Description:
            For Human3.6M dataset let's call spine a chain of joints with indices [0, 7, 8].
            Spine's length defined as:  :math:`S = \| joint_0 - joint_7 \| + \| joint_7 - joint_8 \|`.

        :param animated_skeleton: batch of joints coordinates
        :return: spine lengths across joints batch
        """
        joints_0_7_lengths = np.linalg.norm(animated_skeleton[:, 0] - animated_skeleton[:, 7], axis=1)
        joints_7_8_lengths = np.linalg.norm(animated_skeleton[:, 7] - animated_skeleton[:, 8], axis=1)
        return joints_0_7_lengths + joints_7_8_lengths

    @staticmethod
    def mean_legs_length(animated_skeleton: joints_batch) -> float:
        """
        Description:
            Calculate mean length of legs across joints batch. Legs length is the average of left leg length and right leg length.

        :return: mean legs length across joints batch
        """

        legs_lengths = Human36mStatistics.legs_lengths(animated_skeleton)
        return np.mean(legs_lengths)

    @staticmethod
    def legs_lengths(animated_skeleton: joints_batch) -> np.ndarray:
        """
        Description:
            Calculate lengths of legs across joints batch. Legs length is the average of left leg length and right leg length.
            In Human3.6M right leg joints has indices (1, 2, 3), while left leg has indices (4, 5, 6).

        :return: legs lengths across joints batch
        """
        right_low_legs_length = np.linalg.norm(animated_skeleton[:, 2] - animated_skeleton[:, 1], axis=1)
        right_upper_legs_length = np.linalg.norm(animated_skeleton[:, 3] - animated_skeleton[:, 2], axis=1)

        left_low_legs_length = np.linalg.norm(animated_skeleton[:, 5] - animated_skeleton[:, 4], axis=1)
        left_upper_legs_length = np.linalg.norm(animated_skeleton[:, 6] - animated_skeleton[:, 5], axis=1)

        return 0.5 * (right_low_legs_length + right_upper_legs_length + left_low_legs_length + left_upper_legs_length)

    @staticmethod
    def neck_heads_lengths(animated_skeleton: joints_batch) -> np.ndarray:
        """
        Description:
            Calculate neck-heads lengths across batch of skeletons.

        :param animated_skeleton: batch of animated skeletons
        :return: neck-heads lengths across joints batch.
        """
        neck_head_lengths = np.linalg.norm(animated_skeleton[:, 10] - animated_skeleton[:, 8], axis=1)
        return neck_head_lengths

    @staticmethod
    def mean_neck_heads_length(animated_skeleton: joints_batch) -> float:
        """
        Description:
            Calculate mean neck-heads length across batch of skeletons.

        :param animated_skeleton: batch of animated skeletons
        :return: mean neck-heads length across joints batch.
        """
        neck_head_lengths = Human36mStatistics.neck_heads_lengths(animated_skeleton)
        return np.mean(neck_head_lengths)

    @staticmethod
    def skeletons_heights(animated_skeleton: joints_batch) -> np.ndarray:
        """
        Description:
            Calculate skeletons heights as a sum of skeletons spines, legs and neck-head lengths

        :param animated_skeleton: joints batch
        :return: skeletons heights across joints batch.
        """
        spine_lengths = Human36mStatistics.spine_lengths(animated_skeleton)
        legs_lengths = Human36mStatistics.legs_lengths(animated_skeleton)
        neck_head_lengths = Human36mStatistics.neck_heads_lengths(animated_skeleton)

        skeletons_height = legs_lengths + spine_lengths + neck_head_lengths
        return skeletons_height

    @staticmethod
    def mean_skeletons_height(animated_skeleton: joints_batch) -> float:
        """
        Description:
            Calculate skeletons average length as a sum of skeletons average spines, legs and neck-head lengths

        :param animated_skeleton: joints batch
        :return: mean skeleton lengths across joints batch.
        """
        mean_spine_length = Human36mStatistics.mean_spine_length(animated_skeleton)
        mean_legs_length = Human36mStatistics.mean_legs_length(animated_skeleton)
        mean_neck_head_length = Human36mStatistics.mean_neck_heads_length(animated_skeleton)

        mean_skeletons_length = mean_legs_length + mean_spine_length + mean_neck_head_length
        return mean_skeletons_length
