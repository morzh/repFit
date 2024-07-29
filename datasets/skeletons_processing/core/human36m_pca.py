import numpy as np
from sklearn.decomposition import PCA

import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]
vector3 = Annotated[npt.NDArray[Float32], Literal[3] | Literal[3, 1]]


class Human36mPca:
    """
    Description:
        Principal component analysis (PCA) for Human3.6M dataset.

        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space. The input data is centered or just shifted
        but not scaled for each feature before applying the SVD.

    :ivar skeletons_pca: skeletons dimensionality reduction engine;
    :ivar use_neutral_pose: use given skeleton to center data (instead if calculating mean value in classical PCA approach);
    :ivar number_components: number of PCA components.
    """
    def __init__(self, neutral_poses: np.ndarray | None = None, number_components=1):
        self.skeletons_pca = None
        self.use_neutral_pose: bool = neutral_poses
        self.number_components: int = number_components

    def fit(self, skeleton_animation: joints_batch) -> None:
        """
        Description:
            Fit data (skeletons animations) to the model.

        :param skeleton_animation: joints batch

        """
        if not self.use_neutral_pose:
            self.skeletons_pca = PCA(n_components=self.number_components)
            self.skeletons_pca.fit(skeleton_animation)
        else:
            raise NotImplementedError('Code for true use_neutral_pose value is not implemented yet.')

    def transform(self, skeleton_animation: joints_batch) -> np.ndarray:
        """
        Description:
            Apply dimensionality reduction to the joints batch.

        :param skeleton_animation: joints batch

        :return: joints batch PCA

        """
        if not self.use_neutral_pose:
            return self.skeletons_pca.transform(skeleton_animation)
        else:
            raise NotImplementedError('Code for true use_neutral_pose value is not implemented yet.')
