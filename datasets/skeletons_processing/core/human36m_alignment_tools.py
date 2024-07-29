import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar

from datasets.skeletons_processing.core.human36m_statistics import Human36mStatistics

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]
vector3 = Annotated[npt.NDArray[Float32], Literal[3] | Literal[3, 1]]


class Human36mAlignmentTools:
    r"""
    Description:
        Toolbox for H3.6M animated skeletons (joints batch) processing. Some terminology:

        #. We consider Human3.6M skeleton as an [17, 3] array, In other words, Human3.6M skeleton is a set of 3D joints;
        #. H3.6M animated skeleton (joints batch) is [N, 17, 3] array, N - number of frames;
        #. Skeleton's normal vector is the 3rd vector of the root joint coordinate frame (see calculate_skeletons_root_frame());
        #. Root joint is a joint with index zero (joints[0])
    """

    @staticmethod
    def root_joints_coordinate_frames(animated_skeleton: joints_batch) -> np.ndarray:
        r"""
        Description:
            Calculates coordinate frames of the root joints batch. Skeleton root coordinate frame calculates as follows:

            #. get vector along spine :math:`x = joint_7 - joint_0`;
            #. get vector form right hip joint to left hip joint  :math:`y = joint_4 - joint_1`;
            #. normalize both vectors :math:`x = \frac{x}{\|x\|}, \  y = \frac{y}{\|y\|}`;
            #. get third vector as :math:`z = x \times y`;
            #. to produce orthonormal basis just do another cross product :math:`y = z \times x`.

        :param animated_skeleton: input animated skeleton
        :return: orthogonal coordinate frames of a joints batch
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
    def align_vectors_on_so3(reference_batch: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    def align_skeleton_with_global_frame(animated_skeleton: joints_batch, keep_root_unchanged: bool = True) -> joints_batch:
        r"""
        Description:
            Transforms skeleton in such a way, that root joint coordinate frame and  "global" coordinate frames are coincide.
            This function
                #. Subtracts root joint positions from all other joints positions, root joint itself will be at origin if keep_root_unchanged is False.
                    If keep_root_unchanged is True, root joint coordinates will remain unchanged.
                #. Rotates all joints by a certain angle. Angle itself is computed as the shortest angle between 
                    vector (0, 0, 1) of the global coordinate frame and third vector of a root joint coordinate frame.

        Remarks:
            Since we are aligning third vector of root joint coordinate frame to the third vector of the global frame,
            all we need is to calculate inverse of matrix, obtained from stacked root joint coordinate frame normalized vectors.

        :param animated_skeleton: input animated skeleton;
        :param keep_root_unchanged: root joint coordinates will remain unchanged if True, zeros otherwise.

        :return: aligned joints batch
        """
        animated_root = animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)
        skeletons_shifted_to_origin = animated_skeleton - animated_root
        shifted_skeletons_coordinate_frames = Human36mAlignmentTools.root_joints_coordinate_frames(skeletons_shifted_to_origin)
        alignment_to_global_frame_rotations = np.linalg.inv(shifted_skeletons_coordinate_frames)

        aligned_skeletons = np.matmul(alignment_to_global_frame_rotations, np.transpose(skeletons_shifted_to_origin, axes=(0, 2, 1)))
        aligned_skeletons = np.transpose(aligned_skeletons, axes=(0, 2, 1))

        if keep_root_unchanged:
            aligned_skeletons[:, 0] = animated_root

        return aligned_skeletons

    @staticmethod
    def shift_skeleton_to_origin(animated_skeleton: joints_batch) -> joints_batch:
        """
        Description:
            Shifts animated skeleton in a way, when root joint is always at origin position.
            In other words subtracting root joint position from all other joints, including root joint itself.

        :param animated_skeleton:  input animated skeleton.

        :return: skeleton animation with root joint at origin.
        """
        return animated_skeleton - animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)

    @staticmethod
    def skeleton_features(animated_skeleton: joints_batch, alignment_vector: vector3) -> (vector3, float):
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
        h36m_skeleton_normal_vector = Human36mAlignmentTools.root_joints_coordinate_frames(animated_skeleton)[:, 2]
        feature_2 = np.dot(h36m_skeleton_normal_vector, alignment_vector)

        return feature_1, feature_2

    @staticmethod
    def align_skeletons_heights(animated_skeletons_set: list[joints_batch], in_average: bool = True) -> list[joints_batch]:
        """
        Description:
            Make skeletons of equal heights.

        :param animated_skeletons_set: list of animated skeletons
        :param in_average: if True, scale all skeletons in joints batch using single scale factor value. Otherwise, use per frame scake factors for every
            skeleton in every joint batch.

        :return: list of joints batches
        """
        if in_average:
            return Human36mAlignmentTools.align_skeletons_heights_in_average(animated_skeletons_set)
        else:
            return Human36mAlignmentTools.align_skeletons_heights_per_animation_frame(animated_skeletons_set)

    @staticmethod
    def align_skeletons_heights_in_average(animated_skeletons_set: list[joints_batch]) -> list[joints_batch]:
        """
        Description:
            Align skeletons heights using mean skeleton height across each joints batch.
            In other words each joints batch will be scaled using single factor value across animation frames.

        :param animated_skeletons_set: list of animated skeletons (joints batches).

        :return: list of joints batches
        """
        number_skeletons = len(animated_skeletons_set)
        average_heights = np.empty(number_skeletons)
        scale_factors = np.empty(number_skeletons)

        for index_skeleton in range(number_skeletons):
            average_heights[index_skeleton] = Human36mStatistics.mean_skeletons_height(animated_skeletons_set[index_skeleton])

        mean_height = np.mean(average_heights)
        scale_factors = mean_height / average_heights

        for index_skeleton in range(number_skeletons):
            animated_skeletons_set[index_skeleton] *= scale_factors[index_skeleton]

        return animated_skeletons_set

    @staticmethod
    def align_skeletons_heights_per_animation_frame(animated_skeletons_set: list[joints_batch]) -> list[joints_batch]:
        """
        Description:
            Per frame skeletons heights alignment.
            In other words each skeleton height will coincide with mean of skeletons heights across all joints batches.

        :param animated_skeletons_set: list of animated skeletons (joints batches).

        :return: list of joints batches
        """
        number_skeletons = len(animated_skeletons_set)
        heights_per_frame = [np.ndarray] * number_skeletons

        mean_heights = [np.mean(height) for height in heights_per_frame]
        mean_height = sum(mean_heights) / number_skeletons

        for index_skeleton in range(number_skeletons):
            per_frame_skeleton_heights = Human36mStatistics.skeletons_heights(animated_skeletons_set[index_skeleton])
            per_frame_scale_factors = mean_height / per_frame_skeleton_heights
            animated_skeletons_set[index_skeleton] *= per_frame_scale_factors

        return animated_skeletons_set

    @staticmethod
    def stack_joints_coordinates(skeletons_animations: list[joints_batch], use_root_joint_depth: bool = False) -> np.ndarray:
        """
        Description:
            Stack joints coordinates from list of [N, 17, 3] arrays to [51, M] if use_root_joint_depth is False or [50, M] otherwise.
            Here M equals list length times N.

        :param skeletons_animations: list of skeleton animations (joints batches).
        :param use_root_joint_depth: use third coordinate of the root joint (depth coordinate) in joints stack.

        :return: stacked joints coordinates from all joints batches
        """
        animation_frames_number = len(skeletons_animations)
        stacked_joints = np.empty((0, 51))
        for skeleton_animation in skeletons_animations:
            current_animation_frames_number = skeleton_animation.shape[0]
            current_animation_stacked = skeleton_animation.reshape(current_animation_frames_number, 51)
            stacked_joints = np.vstack((stacked_joints, current_animation_stacked))

        if not use_root_joint_depth:
            stacked_joints = np.delete(stacked_joints, 2, axis=1)

        return stacked_joints
