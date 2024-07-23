import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

from datasets.skeletons_processing.core.human36m_statistics import Human36mStatistics

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]
vector3 = Annotated[npt.NDArray[Float32], Literal[3] | Literal[3, 1]]


class Human36mTools:
    r"""
    Description:
        Toolbox for H3.6M animated skeletons (joints batch) processing. Some terminology:

        #. We consider Human3.6M skeleton as an [17, 3] array, In other words, Human3.6M skeleton is a set of 3D joints;
        #. H3.6M animated skeleton (joints batch) is [N, 17, 3] array, N - number of frames;
        #. Skeleton's normal vector is the 3rd vector of the root joint coordinate frame (see calculate_skeletons_root_frame());
        #. Root joint is a joint with index zero (joints[0])
    """

    @staticmethod
    def coordinate_frame(animated_skeleton: joints_batch):
        r"""
        Description:
            Calculates coordinate frame of the root joint. Skeleton root frame calculates as follows:

            #. get vector along spine :math:`x = joint_7 - joint_0`;
            #. get vector form right hip joint to left hip joint  :math:`y = joint_4 - joint_1`;
            #. normalize both vectors :math:`x = \frac{x}{\|x\|}, \  y = \frac{y}{\|y\|}`;
            #. get third vector as :math:`z = x \times y`;
            #. to produce orthonormal basis just do another cross product :math:`y = z \times x`.

        :param animated_skeleton:  input animated skeleton
        :return: orthogonal coordinate frame of a skeleton
        """

        if animated_skeleton.shape[1:] != (17, 3):
            raise ValueError('Joints must be an [N, 17, 3] array')

        axes_x = animated_skeleton[:, 7] - animated_skeleton[:, 0]
        axes_x /= np.linalg.norm(axes_x, axis=1).reshape(-1, 1)
        axes_y = (animated_skeleton[:, 4] - animated_skeleton[:, 1])
        axes_y /= np.linalg.norm(axes_y, axis=1).reshape(-1, 1)
        axes_z = np.cross(axes_x, axes_y, axis=1)
        axes_y = np.cross(axes_z, axes_x, axis=1)

        return np.dstack((axes_x, axes_y, axes_z))

    @staticmethod
    def so3_alignment(reference_batch: np.ndarray, target: np.ndarray) -> np.ndarray:
        r"""
        Description:
            Find rotation matrix (shortest rotation) which aligns vector_1 and vector_2 in 3D space. \n
            https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space \n
            https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        :param reference_batch: batch of reference vectors
        :param target: target vector
        :return: :math:`\mathbb{SO}(3)` rotation matrices batch
        """
        number_frames = reference_batch.shape[0]
        reference_batch_normalized = (reference_batch / np.linalg.norm(reference_batch, axis=1)).reshape(-1, 1)
        target_normalized = (target / np.linalg.norm(target)).reshape(1, 3)

        cross_references_target = np.cross(reference_batch_normalized, target_normalized)
        dot_references_target = np.dot(reference_batch_normalized, target_normalized.reshape(3, 1))
        s = np.linalg.norm(cross_references_target, axis=1).reshape(-1, 1)

        k = np.zeros((number_frames, 3, 3))
        k[:, 0, 1] = -cross_references_target[:, 2]
        # noinspection PyPep8
        k[:, 1, 0] = cross_references_target[:, 2]
        k[:, 0, 2] = cross_references_target[:, 1]
        k[:, 2, 0] = -cross_references_target[:, 1]
        k[:, 1, 2] = -cross_references_target[:, 0]
        k[:, 2, 1] = cross_references_target[:, 0]

        so3_matrices = np.repeat(np.expand_dims(np.eye(3), axis=0), number_frames, axis=0)
        multiplier = ((1 - dot_references_target) / (s ** 2)).reshape(-1, 1, 1)
        addition = k + np.square(k) * multiplier
        addition_mask = np.where(s < 1e-7, 0., 1.).reshape(-1, 1, 1)
        so3_matrices += addition_mask * addition

        return so3_matrices

    @staticmethod
    def align_to_global_frame(animated_skeleton: joints_batch) -> joints_batch:
        r"""
        Description:
            Transforms skeleton in such a way, that root joint coordinate frame and  "global" coordinate frames are coincide.
            This function
                #. Subtracts root joint positions from all other joints positions
                #. Rotates all joints by a certain angle. Angle itself is computed as the shortest angle between 
                    vector (0, 0, 1) of the 'global' coordinate frame and third vector of a root joint coordinate frame.

        :param animated_skeleton: input animated skeleton;
        :return: aligned joints batch
        """
        skeletons_shifted = animated_skeleton - animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)
        h36m_skeleton_frames = Human36mTools.coordinate_frame(skeletons_shifted)
        alignment_rotation = np.linalg.inv(h36m_skeleton_frames)

        # additional_rotation = Rotation.from_euler('z', -0.5 * np.pi, degrees=False)
        # additional_rotation = np.expand_dims(additional_rotation.as_matrix(), axis=0)
        # skeletons_aligned = np.matmul(additional_rotation @ alignment_rotation, np.transpose(skeletons_shifted, axes=(0, 2, 1)))

        skeletons_aligned = np.matmul(alignment_rotation, np.transpose(skeletons_shifted, axes=(0, 2, 1)))
        skeletons_aligned = np.transpose(skeletons_aligned, axes=(0, 2, 1))
        return skeletons_aligned

    @staticmethod
    def shift_to_origin(animated_skeleton: joints_batch) -> joints_batch:
        """
        Description:
            Shifts animated skeleton in a way, when root joint is always at origin position.
            In other words subtracting root joint position from all other joints, including root joint itself.

        :param animated_skeleton:  input animated skeleton.
        :return: skeleton animation with root joint at origin.
        """
        return animated_skeleton - animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)

    @staticmethod
    def features(animated_skeleton: joints_batch, alignment_vector: vector3) -> (vector3, float):
        r"""
        Description:
            This function returns two vectors (features):
                #. Positions of root joints;
                #. Dot product of normal skeleton vector and alignment_vector.

        :param animated_skeleton: animated skeleton joints coordinates
        :param alignment_vector:
        :return: (Feature 1, Feature 2)
        """
        feature_1 = animated_skeleton[:, 0]
        h36m_skeleton_normal_vector = Human36mTools.coordinate_frame(animated_skeleton)[:, 2]
        feature_2 = np.dot(h36m_skeleton_normal_vector, alignment_vector)

        return feature_1, feature_2

    @staticmethod
    def align(animated_skeletons_set: list[joints_batch], number_iterations=3, transform_to_origin=True) -> list[joints_batch]:
        """
        Description:

        :param animated_skeletons_set:
        :param number_iterations:
        :param transform_to_origin:
        :return: joints batch
        """
        if transform_to_origin:
            number_skeletons = len(animated_skeletons_set)
            for index_skeleton in range(number_skeletons):
                animated_skeletons_set[index_skeleton] = Human36mTools.align_to_global_frame(animated_skeletons_set[index_skeleton])

        for index_iteration in range(number_iterations):
            animated_skeletons_set = Human36mTools.procrustes_iteration(animated_skeletons_set)

        return animated_skeletons_set

    @staticmethod
    def procrustes_iteration(animated_skeletons_set: list[joints_batch]) -> list[joints_batch]:
        """
        Description:

        :param animated_skeletons_set:
        :return:
        """
        number_skeletons = len(animated_skeletons_set)
        skeletons_average_lengths = np.empty(number_skeletons)
        skeletons_factors = np.empty(number_skeletons)

        for index_skeleton in range(number_skeletons):
            skeletons_average_lengths[index_skeleton] = Human36mStatistics.mean_skeletons_length(animated_skeletons_set[index_skeleton])

        mean_skeletons_length = np.mean(skeletons_average_lengths)
        skeletons_factors = mean_skeletons_length / skeletons_average_lengths

        for index_skeleton in range(number_skeletons):
            animated_skeletons_set[index_skeleton] *= skeletons_factors[index_skeleton]

        return animated_skeletons_set

    @staticmethod
    def pac_stack_skeletons(skeleton_animation: joints_batch) -> np.ndarray:
        number_animation_frames = skeleton_animation.shape[0]

        root_joint_components = skeleton_animation[:, 0]
        root_joint_components = root_joint_components[:, :2].reshape(-1, 2)

        aligned_skeleton_animation = Human36mTools.align_to_global_frame(skeleton_animation)
        aligned_skeleton_animation = np.delete(aligned_skeleton_animation, 0, axis=1)

        stacked_skeleton_animation = aligned_skeleton_animation.reshape(number_animation_frames, -1)
        stacked_skeleton_animation = np.hstack((stacked_skeleton_animation, root_joint_components))

        return stacked_skeleton_animation

    @staticmethod
    def skeletons_pca(skeleton_animation: np.ndarray, pca_number_components=1):
        current_number_animation_frames = skeleton_animation.shape[0]

        current_root_animation = skeleton_animation[:, 0]
        current_root_animation_vertical_horizontal_component = current_root_animation[:, :2].reshape(-1, 2)

        aligned_skeleton_animation = Human36mTools.align_to_global_frame(skeleton_animation)
        aligned_skeleton_animation = np.delete(aligned_skeleton_animation, 0, axis=1)

        stacked_skeleton_animation = aligned_skeleton_animation.reshape(current_number_animation_frames, -1)
        stacked_skeleton_animation = np.hstack((stacked_skeleton_animation, current_root_animation_vertical_horizontal_component))
        skeletons_3d_stacked = np.hstack((skeletons_3d_stacked, stacked_skeleton_animation.T))
