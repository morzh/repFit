import os.path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datasets.skeletons_processing.core.h36m_animated_skeleton_tools import H36mAnimatedSkeletonTools
from datasets.skeletons_processing.core.set_animated_skeletons_procrustes import AnimatedSkeletonsProcrustes


def transform_to_stacked_root_vertical_component_only(skeleton_animation):
    number_animation_frames = skeleton_animation.shape[0]

    root_animation = skeleton_animation[:, 0]
    root_animation_vertical_component = root_animation[:, 1].reshape(-1, 1)

    aligned_skeleton_animation = H36mAnimatedSkeletonTools.align_to_global_frame(skeleton_animation)
    aligned_skeleton_animation = np.delete(aligned_skeleton_animation, 0, axis=1)

    stacked_skeleton_animation = aligned_skeleton_animation.reshape(number_animation_frames, -1)
    stacked_skeleton_animation = np.hstack((stacked_skeleton_animation, root_animation_vertical_component))

    return stacked_skeleton_animation


def transform_to_stacked_root_vertical_horizontal_components(skeleton_animation):
    number_animation_frames = skeleton_animation.shape[0]

    root_animation = skeleton_animation[:, 0]
    root_animation_horizontal_vertical_component = root_animation[:, :2].reshape(-1, 2)

    aligned_skeleton_animation = H36mAnimatedSkeletonTools.align_to_global_frame(skeleton_animation)
    aligned_skeleton_animation = np.delete(aligned_skeleton_animation, 0, axis=1)

    stacked_skeleton_animation = aligned_skeleton_animation.reshape(number_animation_frames, -1)
    stacked_skeleton_animation = np.hstack((stacked_skeleton_animation, root_animation_horizontal_vertical_component))

    return stacked_skeleton_animation


if __name__ == '__main__':
    do_procrustes_alignment = True
    save_stacked_skeletons = False
    save_skeletons_pca = True
    show_skeleton_animation_pca = False

    root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_skeletons/results_base_video_mp3'

    input_videos_folder = os.path.join(root_folder, 'filtered_final_video')
    input_joints_3d_folder = os.path.join(root_folder, 'joints3d')
    output_joints_3d_stacked_folder = os.path.join(root_folder, 'joints3d_stacked_procrustes')
    output_joints_3d_pca_folder = os.path.join(root_folder, 'joints3d_pca')

    video_files_extensions = ['.mp4', '.mkv', '.webm']
    video_filenames = [f for f in os.listdir(input_videos_folder) if os.path.splitext(f)[1] in video_files_extensions]
    video_filenames.sort()

    os.makedirs(output_joints_3d_stacked_folder, exist_ok=True)
    os.makedirs(output_joints_3d_pca_folder, exist_ok=True)
    skeletons_3d_data = [np.ndarray] * len(video_filenames)
    skeletons_3d_stacked = np.empty((16 * 3 + 2, 0), dtype=np.float32)

    for index, video_filename in enumerate(video_filenames):
        video_filename_base = os.path.splitext(video_filename)[0]
        joints_3d_filename = video_filename_base + '.npy'
        joints_3d_filepath = os.path.join(input_joints_3d_folder, joints_3d_filename)
        skeletons_3d_data[index] = np.load(joints_3d_filepath)
        if video_filename == 'Arg6JZg4UdA_1_0.mp4':
            skeletons_3d_data[index] = skeletons_3d_data[index][:390]

    if do_procrustes_alignment:
        skeletons_3d_data = AnimatedSkeletonsProcrustes.align(skeletons_3d_data, transform_to_origin=False)

    for index, video_filename in enumerate(video_filenames):
        current_filename_base = os.path.splitext(video_filename)[0]
        current_skeleton_animation = skeletons_3d_data[index]
        current_number_animation_frames = current_skeleton_animation.shape[0]

        current_root_animation = current_skeleton_animation[:, 0]
        current_root_animation_vertical_horizontal_component = current_root_animation[:, :2].reshape(-1, 2)

        current_aligned_skeleton_animation = H36mAnimatedSkeletonTools.align_to_global_frame(current_skeleton_animation)
        current_aligned_skeleton_animation = np.delete(current_aligned_skeleton_animation, 0, axis=1)

        current_stacked_skeleton_animation = current_aligned_skeleton_animation.reshape(current_number_animation_frames, -1)
        current_stacked_skeleton_animation = np.hstack((current_stacked_skeleton_animation, current_root_animation_vertical_horizontal_component))
        skeletons_3d_stacked = np.hstack((skeletons_3d_stacked, current_stacked_skeleton_animation.T))

        if save_stacked_skeletons:
            np.save(os.path.join(output_joints_3d_stacked_folder, current_filename_base), current_stacked_skeleton_animation)

    skeletons_pca = PCA(n_components=1)
    skeletons_pca.fit(skeletons_3d_stacked.T)

    for index, video_filename in enumerate(video_filenames):
        current_filename_base = os.path.splitext(video_filename)[0]
        current_skeleton_animation = skeletons_3d_data[index]
        # current_skeleton_animation_stacked = transform_to_stacked_root_vertical_component_only(current_skeleton_animation)
        current_skeleton_animation_stacked = transform_to_stacked_root_vertical_horizontal_components(current_skeleton_animation)
        current_skeleton_animation_pca = skeletons_pca.transform(current_skeleton_animation_stacked)

        if save_skeletons_pca:
            np.save(os.path.join(output_joints_3d_pca_folder, current_filename_base), current_skeleton_animation_pca)

        if show_skeleton_animation_pca:
            plt.figure(video_filename, figsize=(25, 10))
            # plt.suptitle(video_filename)
            plt.subplot(211)
            plt.plot(range(current_skeleton_animation_pca.shape[0]), current_skeleton_animation_pca[:, 0])
            plt.subplot(212)
            plt.plot(range(current_skeleton_animation_pca.shape[0]), current_skeleton_animation_pca[:, 1])
            plt.tight_layout()
            plt.show()
