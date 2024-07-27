import numpy as np
from sklearn.decomposition import PCA

import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

from datasets.skeletons_processing.core.human36m_alignment_tools import Human36mAlignmentTools

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]
vector3 = Annotated[npt.NDArray[Float32], Literal[3] | Literal[3, 1]]


class Human36mPca:
    def __init__(self):
        self.skeletons_pca = None

    @staticmethod
    def stack_skeletons(skeleton_animation: joints_batch) -> np.ndarray:
        number_animation_frames = skeleton_animation.shape[0]

        root_joint_components = skeleton_animation[:, 0]
        root_joint_components = root_joint_components[:, :2].reshape(-1, 2)

        aligned_skeleton_animation = Human36mAlignmentTools.align_skeleton_with_global_frame(skeleton_animation)
        aligned_skeleton_animation = np.delete(aligned_skeleton_animation, 0, axis=1)

        stacked_skeleton_animation = aligned_skeleton_animation.reshape(number_animation_frames, -1)
        stacked_skeleton_animation = np.hstack((stacked_skeleton_animation, root_joint_components))

        return stacked_skeleton_animation

    def fit(self, skeleton_animation: joints_batch, pca_number_components=1, neutral_poses: np.ndarray | None = None):
        # skeletons_number = skeleton_animation


        self.skeletons_pca = PCA(n_components=pca_number_components)
        self.skeletons_pca.fit(skeletons_3d_stacked.T)

    def transform(self, skeleton_animation: joints_batch):
        pass
