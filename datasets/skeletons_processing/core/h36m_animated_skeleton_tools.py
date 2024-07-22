import glob
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing import Annotated, Literal, TypeVar

DType = TypeVar("DType", bound=np.float32)
arrayNx17x3 = Annotated[npt.NDArray[DType], Literal["N", 17, 3]]
arrayMxNx17x3 = Annotated[npt.NDArray[DType], Literal["M", "N", 17, 3]]
vector3 = Annotated[npt.NDArray[DType], Literal[3] | Literal[3, 1]]
vector4 = Annotated[npt.NDArray[DType], Literal[4] | Literal[4, 1]]
matrix3x3 = Annotated[npt.NDArray[DType], Literal[3, 3]]


class H36mAnimatedSkeletonTools:
    r"""
    Description:
        Toolbox for H3.6M animated skeletons processing.
        Some terminology:

        #. skeleton is an [N, 2] or [N, 3] array, where N is number of joints. In other words, skeleton is a set of 2D or 3D joints
        #. Animated skeleton is [M, N, 2] or [M, N, 3] array, M - number of animation frames,  N is a number of joints.
        #. Normal skeleton vector is the 3rd vector of the root joint coordinate frame (see calculate_skeletons_root_frame())
        #. Root joint is a joint with index zero (joints[0])

    Terminology:
        #. Mean value is calculated across set of objects (e.g. spines, legs, etc.)
        #. Average value is calculated for a single skeleton (e.g. average legs length).
    """

    @staticmethod
    def coordinate_frame(animated_skeleton: arrayNx17x3):
        r"""
        Description:
            Calculates coordinate frame of the root joint. Skeleton root frame calculates as follows:

            #. get vector along spine :math:`x = joint_7 - joint_0`;
            #. get vector form right hip joint to left hip joint  :math:`y = joint_4 - joint_1`;
            #. normalize both vectors :math:`x = \frac{x}{\|x\|}, y = \frac{y}{\|y\|}`;
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
    def rotation_matrix_from_vectors(vector_reference: vector3, vector_target: vector3) -> matrix3x3:
        """
        Description:
            Find rotation matrix (shortest rotation) between vector_1 and vector_2. \n
            https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space \n
            https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        :param vector_reference: reference vector
        :param vector_target: target vector
        :return: 3 by 3 rotation matrix which when applied to vector_1, aligns it with vector_2.
        """
        number_frames = vector_reference.shape[0]
        a = vector_reference / np.linalg.norm(vector_reference, axis=1).reshape(-1, 1)
        b = (vector_target / np.linalg.norm(vector_target)).reshape(1, 3)
        v = np.cross(a, b)
        c = np.dot(a, b.reshape(3, 1))
        s = np.linalg.norm(v, axis=1).reshape(-1, 1)

        k = np.zeros((number_frames, 3, 3))
        k[:, 0, 1] = -v[:, 2]
        k[:, 1, 0] = v[:, 2]
        k[:, 0, 2] = v[:, 1]
        k[:, 2, 0] = -v[:, 1]
        k[:, 1, 2] = -v[:, 0]
        k[:, 2, 1] = v[:, 0]

        rotation_matrix = np.repeat(np.expand_dims(np.eye(3), axis=0), number_frames, axis=0)
        multiplier = ((1 - c) / (s ** 2)).reshape(-1, 1, 1)
        addition = k + np.square(k) * multiplier
        mask = np.where(s < 1e-7, 0., 1.).reshape(-1, 1, 1)
        rotation_matrix += mask * addition

        return rotation_matrix

    @staticmethod
    def align_to_global_frame(animated_skeleton: arrayNx17x3) -> arrayNx17x3:
        """
        Description:
            Transforms skeleton in such a way, that root joint coordinate frame and  "global" coordinate frames are coincide.
            This function
                1. Subtracts root joint positions from all other joints positions
                2. Rotates all joints by a certain angle. Angle itself is computed as the shortest angle between
                    vector (0, 0, 1) of the 'global' coordinate frame and third vector of a root joint coordinate frame.

        :param animated_skeleton: input animated skeleton;
        :return: aligned animated skeleton.
        """
        skeletons_shifted = animated_skeleton - animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)
        h36m_skeleton_frames = H36mAnimatedSkeletonTools.coordinate_frame(skeletons_shifted)
        alignment_rotation = np.linalg.inv(h36m_skeleton_frames)

        # additional_rotation = Rotation.from_euler('z', -0.5 * np.pi, degrees=False)
        # additional_rotation = np.expand_dims(additional_rotation.as_matrix(), axis=0)
        # skeletons_aligned = np.matmul(additional_rotation @ alignment_rotation, np.transpose(skeletons_shifted, axes=(0, 2, 1)))

        skeletons_aligned = np.matmul(alignment_rotation, np.transpose(skeletons_shifted, axes=(0, 2, 1)))
        skeletons_aligned = np.transpose(skeletons_aligned, axes=(0, 2, 1))
        return skeletons_aligned

    @staticmethod
    def shift_to_origin(animated_skeleton: arrayNx17x3) -> arrayNx17x3:
        """
        Description:
            Shifts skeleton animation in a way, when root joint is always at origin position.
            In other words subtracting root joint position from all other joints, including root joint.

        :param animated_skeleton:  input animated skeleton.
        :return: skeleton animation with root joint at origin.
        """
        return animated_skeleton - animated_skeleton[:, 0].reshape(animated_skeleton.shape[0], 1, 3)

    @staticmethod
    def features(animated_skeleton: arrayNx17x3, alignment_vector: vector3) -> (vector3, float):
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
        h36m_skeleton_normal_vector = H36mAnimatedSkeletonTools.coordinate_frame(animated_skeleton)[:, 2]
        feature_2 = np.dot(h36m_skeleton_normal_vector, alignment_vector)

        return feature_1, feature_2

    @staticmethod
    def mean_spines_length(animated_skeleton: arrayNx17x3) -> float:
        r"""
        Description:
            in H3.6M dataset let's call spine a chain of joints with indices [0, 7, 8].
            Spine's length defined as:  :math:`S = \| joint_0 - joint_7 \| + \| joint_7 - joint_8 \|`.

        :param animated_skeleton:  animated skeleton joints coordinates
        :return: mean value across animation frames spine length
        """
        joints_0_7_lengths = np.linalg.norm(animated_skeleton[:, 0] - animated_skeleton[:, 7], axis=1)
        joints_7_8_lengths = np.linalg.norm(animated_skeleton[:, 7] - animated_skeleton[:, 8], axis=1)
        average_spine_length = np.mean(joints_0_7_lengths + joints_7_8_lengths)
        return average_spine_length

    @staticmethod
    def mean_legs_length(animated_skeleton: arrayNx17x3) -> float:
        """
        Description:

        :return: average legs length
        """
        right_low_leg_average_length = np.linalg.norm(animated_skeleton[:, 2] - animated_skeleton[:, 1], axis=1)
        right_upper_leg_average_length = np.linalg.norm(animated_skeleton[:, 3] - animated_skeleton[:, 2], axis=1)
        average_right_leg_length = np.mean(right_low_leg_average_length + right_upper_leg_average_length)

        left_low_leg_average_length = np.linalg.norm(animated_skeleton[:, 5] - animated_skeleton[:, 4], axis=1)
        left_upper_leg_average_length = np.linalg.norm(animated_skeleton[:, 6] - animated_skeleton[:, 5], axis=1)
        average_left_leg_length = np.mean(left_low_leg_average_length + left_upper_leg_average_length)

        return 0.5 * (average_right_leg_length + average_left_leg_length)

    @staticmethod
    def mean_neck_heads_length(animated_skeleton: arrayNx17x3) -> float:
        """
        Description:
            Calculate mean neck-heads length across batch of skeletons.

        :param animated_skeleton: batch of animated skeletons
        :return: mean neck-heads length across batch of skeletons.
        """
        mean_neck_head_length = np.mean(np.linalg.norm(animated_skeleton[:, 10] - animated_skeleton[:, 8], axis=1))
        return mean_neck_head_length

    @staticmethod
    def mean_skeletons_length(animated_skeleton: arrayNx17x3) -> float:
        """
        Description:
            Calculate skeletons average length as a sum of skeletons average spines, legs and neck-head lengths

        :param animated_skeleton: batch of animated skeletons
        :return: mean skeleton lengths across batch of skeletons.
        """
        mean_spine_length = H36mAnimatedSkeletonTools.mean_spines_length(animated_skeleton)
        mean_legs_length = H36mAnimatedSkeletonTools.mean_legs_length(animated_skeleton)
        mean_neck_head_length = H36mAnimatedSkeletonTools.mean_neck_heads_length(animated_skeleton)

        mean_skeletons_length = mean_spine_length + mean_legs_length + mean_neck_head_length
        return mean_skeletons_length


def select_main_skeleton_animation(skeletons: list[arrayNx17x3]) -> np.ndarray:
    skeletons_bounding_boxes_volumes = []
    skeletons_bounding_boxes_frames = []
    for skeleton in skeletons:
        current_bbox_min_corner = np.min(skeleton, axis=(0, 1))
        current_bbox_max_corner = np.max(skeleton, axis=(0, 1))
        current_bbox_size = current_bbox_max_corner - current_bbox_min_corner
        current_bbox_volume = np.cumprod(current_bbox_size)[-1]
        skeletons_bounding_boxes_volumes.append(current_bbox_volume)
        skeletons_bounding_boxes_frames.append(skeleton.shape[0])

    maximum_volume_index = np.argmax(skeletons_bounding_boxes_volumes)
    maximum_frames_segment_index = np.argmax(skeletons_bounding_boxes_frames)
    '''
    TODO: improve the following heuristics of bounding box selection
    '''
    if maximum_volume_index == maximum_frames_segment_index:
        ''' largest bounding box which appeared in maximum frames '''
        return skeletons[maximum_frames_segment_index]
    else:
        ''' just simple heuristics,  select character which stays longer in video'''
        return skeletons[maximum_frames_segment_index]


def participants_3d_skeletons(video_files_path: str, motion_bert_tracked_path: str) -> dict:
    tracked_3d_skeletons_filenames = glob.glob(motion_bert_tracked_path + '/*.npy')
    tracked_3d_skeletons_filenames_set = set([f.split('.mp4_')[0] + '.mp4' for f in tracked_3d_skeletons_filenames])

    skeletons_3d_map = {}
    for skeleton_3d_filename in tracked_3d_skeletons_filenames_set:
        current_tracked_3d_skeletons_filenames = [f for f in tracked_3d_skeletons_filenames
                                                  if f.startswith(skeleton_3d_filename)]
        # print(current_tracked_3d_skeletons_filenames)
        current_tracked_3d_skeletons = []
        for participant_skeleton_filename in current_tracked_3d_skeletons_filenames:
            current_participant_skeleton = np.load(participant_skeleton_filename)
            current_tracked_3d_skeletons.append(current_participant_skeleton)

        skeletons_3d_map[skeleton_3d_filename] = current_tracked_3d_skeletons

    return skeletons_3d_map


'''
if __name__ == "__main__":
    dataset_root_folder = \
        '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/dataset_aggregation/squats_2022_skeletons_02'
    
    alpha_pose_videos_folder = 'AlphaPose/video'
    motion_bert_tracked_skeletons_folder = 'MotionBERT_track'
    motion_bert_selected_skeletons_folder = 'MotionBERT_selected'
    
    vector_to_align = np.array([0, 0, 1])
    
    alpha_pose_videos_path = os.path.join(dataset_root_folder, alpha_pose_videos_folder)
    tracked_3d_skeletons_path = os.path.join(dataset_root_folder, motion_bert_tracked_skeletons_folder)
    selected_3d_skeletons = os.path.join(dataset_root_folder, motion_bert_selected_skeletons_folder)
    
    tracked_3d_skeletons_map = participants_3d_skeletons(alpha_pose_videos_path, tracked_3d_skeletons_path)
    
    for video_file_pathname, skeletons_animations in tracked_3d_skeletons_map.items():
        number_skeletons_in_video = len(skeletons_animations)
        main_skeleton_animation = select_main_skeleton_animation(skeletons_animations)
        skeleton_animation_aligned = align_h36m_skeletons(main_skeleton_animation, vector_to_align)
        skeleton_animation_root, skeleton_animation_rotation = features_from_h36m_skeletons(main_skeleton_animation,
                                                                                            vector_to_align)
        # skeleton_animation_aligned[:, 0, :] = skeleton_animation_root
        video_filename = os.path.basename(video_file_pathname)
        selected_skeletons_file_pathname = os.path.join(selected_3d_skeletons, video_filename)
    
        np.save(selected_skeletons_file_pathname + '.selected_skeleton.npy', skeleton_animation_aligned)
        np.save(selected_skeletons_file_pathname + '.selected_skeleton_angle.npy', skeleton_animation_rotation)
        np.save(selected_skeletons_file_pathname + '.selected_skeleton_root.npy', skeleton_animation_root)
'''