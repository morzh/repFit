import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

from datasets.skeletons_processing.core.h36m_animated_skeleton_tools import H36mAnimatedSkeletonTools


DType = TypeVar("DType", bound=np.float32)
arrayNx17x3 = Annotated[npt.NDArray[DType], Literal["N", 17, 3]]
arrayMxNx17x3 = Annotated[npt.NDArray[DType], Literal["M", "N", 17, 3]]


class AnimatedSkeletonsProcrustes:
    """
    Description:
    """
    @staticmethod
    def align(animated_skeletons_set: list[arrayNx17x3], number_iterations=3, transform_to_origin=True) -> list[arrayNx17x3]:
        """
        Description:

        :param animated_skeletons_set:
        :param number_iterations:
        :param transform_to_origin:
        :return: skeletons batch
        """
        if transform_to_origin:
            number_skeletons = len(animated_skeletons_set)
            for index_skeleton in range(number_skeletons):
                animated_skeletons_set[index_skeleton] = H36mAnimatedSkeletonTools.align_to_global_frame(animated_skeletons_set[index_skeleton])

        for index_iteration in range(number_iterations):
            animated_skeletons_set = AnimatedSkeletonsProcrustes._procrustes_spin(animated_skeletons_set)

        return animated_skeletons_set

    @staticmethod
    def _procrustes_spin(animated_skeletons_set: list[arrayNx17x3]) -> list[arrayNx17x3]:
        """
        Description:

        :param animated_skeletons_set:
        :return:
        """
        number_skeletons = len(animated_skeletons_set)
        skeletons_average_lengths = np.empty(number_skeletons)
        skeletons_factors = np.empty(number_skeletons)

        for index_skeleton in range(number_skeletons):
            skeletons_average_lengths[index_skeleton] = H36mAnimatedSkeletonTools.mean_skeletons_length(animated_skeletons_set[index_skeleton])

        mean_skeletons_length = np.mean(skeletons_average_lengths)
        skeletons_factors = mean_skeletons_length / skeletons_average_lengths

        for index_skeleton in range(number_skeletons):
            animated_skeletons_set[index_skeleton] *= skeletons_factors[index_skeleton]

        return animated_skeletons_set
