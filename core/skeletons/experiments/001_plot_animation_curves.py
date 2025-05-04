import glob
import os.path

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt
import numpy as np

from datasets.datasets_preporocessing.identifying_periodic_signal_patterns.similarity import similarity


def auto_correlation(input_array, shift_value, weights=None):
    """
    auto correlation function implementation
    :param input_array: assuming input array  has [number_channels, number_frames] format
    :param weights: weights for each channel
    :param shift_value: integer shift
    :return: auto correlation at shift_value
    """
    if not isinstance(shift_value, int):
        raise ValueError('shift_value should be an integer')

    if shift_value == 0:
        return 0.0
    else:
        input_array_shifted = np.delete(input_array, range(shift_value), axis=1)
        input_array_cropped = input_array[:, :-shift_value]
        absolute_differences = np.abs(input_array_cropped - input_array_shifted)
        if weights is not None:
            absolute_differences *= weights.reshape(-1, 1)
        value = np.mean(absolute_differences)
        return value


def visualize_skeleton_animation(joints_animation: np.ndarray, show_joints_indices=False):
    number_samples = joints_animation.shape[0]
    for index in range(number_samples):
        joints = joints_animation[index]
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
        if show_joints_indices:
            for joint_index in range(joints.shape[0]):
                ax.text(joints[joint_index, 0], joints[joint_index, 1], joints[joint_index, 2], str(joint_index))
        plt.show()


def calculate_skeleton_root_frame(joints: np.ndarray, show_frame=False, show_joints_indices=False):
    if joints.shape != (17, 3):
        raise ValueError('Joints must be an [17, 3] array')

    axis_x = joints[7] - joints[0]
    axis_x /= np.linalg.norm(axis_x)
    axis_y = (joints[4] - joints[1])
    axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_y = np.cross(axis_z, axis_x)

    if show_frame:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
        ax.quiver(joints[0, 0], joints[0, 1], joints[0, 2], axis_x[0], axis_x[1], axis_x[2], color='r')
        ax.quiver(joints[0, 0], joints[0, 1], joints[0, 2], axis_y[0], axis_y[1], axis_y[2], color='g')
        ax.quiver(joints[0, 0], joints[0, 1], joints[0, 2], axis_z[0], axis_z[1], axis_z[2], color='b')
        if show_joints_indices:
            for joint_index in range(joints.shape[0]):
                ax.text(joints[joint_index, 0], joints[joint_index, 1], joints[joint_index, 2], str(joint_index))
        ax.set_aspect('equal', 'box')
        plt.show()

    return np.array([axis_x, axis_y, axis_z])


def calculate_skeleton_animation_root_frame(joints_animation: np.ndarray, show_root_frame=False, show_joints_indices=False):
    number_animation_frames = joints_animation.shape[0]
    root_frames = np.zeros((number_animation_frames, 3, 3))
    for index in range(number_animation_frames):
        root_frames[index] = calculate_skeleton_root_frame(joints_animation[index], show_root_frame, show_joints_indices)

    return root_frames


def parent_joints_to_root(joints: np.ndarray, show_plot=False) -> np.ndarray:
    if joints.shape != (17, 3):
        raise ValueError(f'Joints must be an [17, 3] array, but {joints.shape} provided')

    root_coordinate_frame = calculate_skeleton_root_frame(joints)
    # transform = np.array([])
    joints_transformed = joints.copy()
    joints_transformed[1:] = np.transpose(root_coordinate_frame.T @ (joints[1:].T - joints[0].reshape(-1, 1)))
    if show_plot:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
        ax.scatter(joints_transformed[:, 0], joints_transformed[:, 1], joints_transformed[:, 2])
        plt.show()
    return joints_transformed


def parent_joints_to_root_animation(joints_animation: np.ndarray, show_plot=False) -> np.ndarray:
    if joints_animation.shape[1:] != (17, 3):
        raise ValueError(f'Joints animation must be an [N, 17, 3] array, but {joints_animation.shape} provided')

    number_frames = joints_animation.shape[0]
    for index in range(number_frames):
        joints_animation[index] = parent_joints_to_root(joints_animation[index], show_plot)

    return joints_animation


def flip_sibling_joints_around_root(joints: np.ndarray) -> np.ndarray:
    root_joint = joints[0].reshape(-1, 1)
    root_coordinate_frame = calculate_skeleton_root_frame(joints)

    sibling_leg_joints = joints[4:7]
    sibling_arm_joints = joints[11:14]

    sibling_leg_joints = np.transpose(root_coordinate_frame.T @ (sibling_leg_joints.T - root_joint))
    sibling_leg_joints[:, 1] *= -1.0
    sibling_leg_joints = np.transpose(root_coordinate_frame @ sibling_leg_joints.T + root_joint)

    sibling_arm_joints = np.transpose(root_coordinate_frame.T @ (sibling_arm_joints.T - root_joint))
    sibling_arm_joints[:, 1] *= -1.0
    sibling_arm_joints = np.transpose(root_coordinate_frame @ sibling_arm_joints.T + root_joint)

    joints[4:7] = sibling_leg_joints
    joints[11:14] = sibling_arm_joints

    return joints


def flip_sibling_joints_animation_around_root(joints_animation: np.ndarray, show_plot=False) -> np.ndarray:
    number_frames = joints_animation.shape[0]
    number_joints = joints_animation.shape[1]
    for index in range(number_frames):
        joints_animation[index] = flip_sibling_joints_around_root(joints_animation[index])
        if show_plot:
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(joints_animation[index, :, 0], joints_animation[index, :, 1], joints_animation[index, :, 2])
            for joint_index in range(number_joints):
                ax.text(joints_animation[index, joint_index, 0],
                        joints_animation[index, joint_index, 1],
                        joints_animation[index, joint_index, 2],
                        str(joint_index))
            ax.set_aspect('equal', 'box')
            plt.show()

    return joints_animation


draw_plots = True

dataset_videos_folder = ''
dataset_skeletons_folder = ''

root_folder = \
    '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/dataset_aggregation/squats_2022_skeletons/data/MotionBERT'

video_files = [x for x in glob.glob(root_folder)]
# video_filename = 'Squat Bodyweight-zawXRtyOTuA.mp4'
# video_filename = 'stephanie doing kettlebell clean and squat to press-YduwRC64IQo.mp4'
# video_filename = 'zercher squats-Nr3FcuYJJBI.mp4'
# video_filename = 'Приседания ноги узко 200_15-GyPdjhQF3lk.mp4'
# video_filename = 'Squat Training, 195x8-RxWaDZd8bp8.mp4'
# video_filename = 'Split Squat-aH7CsxXDj5w.mp4'
video_filename = 'R Sindelar ohead squat-PrX15kDXhQ8.mp4'
joint_animation_filename = video_filename + '.json.npy'

file_pathname = os.path.join(root_folder, joint_animation_filename)
print(joint_animation_filename)

skeleton_animation_curves = np.load(file_pathname)
number_frames = skeleton_animation_curves.shape[0]
print(skeleton_animation_curves.shape)

# visualize_skeleton_animation(skeleton_animation_curves, show_joints_indices=True)
# calculate_skeleton_animation_root_frame(skeleton_animation_curves, show_root_frame=True, show_joints_indices=False)
# skeleton_animation_curves = parent_joints_to_root_animation(skeleton_animation_curves)


# skeleton_animation_curves = skeleton_animation_curves.reshape((skeleton_animation_curves.shape[0], -1)).T
skeleton_animation_curves = savgol_filter(skeleton_animation_curves, 51, 2, axis=0)
# for index in range(skeleton_animation_curves.shape[0]):
#     skeleton_animation_curves[index] = medfilt(skeleton_animation_curves[index], kernel_size=21)
skeleton_animation_curves = gaussian_filter1d(skeleton_animation_curves, sigma=3, radius=9, axis=0)
# skeleton_animation_curves = flip_sibling_joints_animation_around_root(skeleton_animation_curves, show_plot=False)
# skeleton_animation_curves = parent_joints_to_root_animation(skeleton_animation_curves)

# skeleton_animation_curves = skeleton_animation_curves[:, :7]

curves_mean = np.mean(skeleton_animation_curves, axis=0)
curves_std = np.std(skeleton_animation_curves, axis=0)
curves_std_percentile = np.percentile(curves_std, 50)

if draw_plots:
    plt.figure('Animation cures', figsize=(30, 20))
    plt.plot(range(number_frames), skeleton_animation_curves[:, 0, :])
    # plt.plot(range(number_frames), skeleton_animation_curves.reshape(number_frames, -1))
    plt.tight_layout()
    plt.show()

skeleton_animation_curves -= curves_mean

if draw_plots:
    plt.figure('Animation cures', figsize=(30, 20))
    plt.plot(range(number_frames), skeleton_animation_curves.reshape(number_frames, -1))
    plt.tight_layout()
    plt.show()

animation_curves_mask = (curves_std > curves_std_percentile)
skeleton_animation_curves = skeleton_animation_curves[animation_curves_mask, :]
curves_mean = np.mean(skeleton_animation_curves, axis=1)
curves_std = np.std(skeleton_animation_curves, axis=1)
curves_colors_denominator = np.maximum(np.max(curves_std) - np.min(curves_std), 0.01)
curves_colors = (curves_std - np.min(curves_std)) / curves_colors_denominator
curves_colors *= 0.9
curves_colors += 0.2
curves_colors /= 1.2

if draw_plots:
    plt.figure('Animation cures', figsize=(30, 20))
    for i in range(len(curves_colors)):
        plt.plot(range(skeleton_animation_curves.shape[1]), skeleton_animation_curves[i], color=str(1 - curves_colors[i]))
    plt.tight_layout()
    plt.show()


'''
shift_values = [i for i in range(animation_curves.shape[1])]
auto_correlation_values = np.ones(len(shift_values))
for index, shift in enumerate(shift_values):
    auto_correlation_values[index] = auto_correlation(animation_curves, shift, weights=curves_std)

plt.figure('Auto Correlation', figsize=(30, 10))
plt.plot(range(len(auto_correlation_values)), auto_correlation_values)
plt.tight_layout()
plt.show()
'''

skeleton_animation_curves *= curves_std.reshape(-1, 1)
animation_curves_averaged = np.mean(skeleton_animation_curves, axis=0)
animation_curves_averaged = medfilt(animation_curves_averaged, kernel_size=31)
animation_curves_averaged = gaussian_filter1d(animation_curves_averaged, sigma=3, radius=9)

plt.figure('Animation cures', figsize=(30, 20))
plt.plot(range(len(animation_curves_averaged)), animation_curves_averaged)
plt.tight_layout()
plt.show()
plt.close()

skeleton_animation_curves_ss = similarity(animation_curves_averaged)


plt.figure('Similarity Matrix', figsize=(30, 20))
plt.matshow(skeleton_animation_curves_ss)
plt.tight_layout()
plt.show()

