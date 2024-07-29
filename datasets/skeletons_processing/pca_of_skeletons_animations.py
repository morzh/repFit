import os.path
import numpy as np
from datasets.skeletons_processing.core.human36m_alignment_tools import Human36mAlignmentTools
from datasets.skeletons_processing.core.human36m_pca import Human36mPca


def run_skeletons_pca(root_directory: str):
    input_videos_folder = os.path.join(root_directory, 'filtered_final_video')
    input_joints_folder = os.path.join(root_directory, 'joints3d')
    output_joints_stacked_folder = os.path.join(root_directory, 'joints3d_stacked')
    output_joints_pca_folder = os.path.join(root_directory, 'joints3d_stacked')

    os.makedirs(output_joints_stacked_folder, exist_ok=True)
    os.makedirs(output_joints_pca_folder, exist_ok=True)

    video_files_extensions = ['.mp4', '.mkv', '.webm']
    video_filenames = [f for f in os.listdir(input_videos_folder) if os.path.splitext(f)[1] in video_files_extensions]
    video_filenames.sort()

    os.makedirs(output_joints_stacked_folder, exist_ok=True)
    skeletons_animations = [np.ndarray] * len(video_filenames)

    skeletons_animation = np.empty((0, 17, 3))

    for filename_index, video_filename in enumerate(video_filenames):
        video_filename_base = os.path.splitext(video_filename)[0]
        skeleton_animation_filename = video_filename_base + '.npy'
        skeleton_animation_filepath = os.path.join(input_joints_folder, skeleton_animation_filename)
        skeletons_animations[filename_index] = np.load(skeleton_animation_filepath)

    skeletons_animations = Human36mAlignmentTools.align_skeletons_heights(skeletons_animations)
    skeletons_animations = Human36mAlignmentTools.align_skeleton_with_global_frame(skeletons_animations)
    stacked_skeletons_animations = Human36mAlignmentTools.stack_joints_coordinates(skeletons_animations, use_root_joint_depth=False)

    skeletons_pca = Human36mPca()
    skeletons_pca.fit(stacked_skeletons_animations)

    for video_filename, skeleton_animation in zip(video_filenames, skeletons_animations):
        current_skeleton_animation_pca = skeletons_pca.transform(skeleton_animation)
        current_output_filepath = os.path.join(output_joints_pca_folder, video_filename + '.npy')
        np.save(current_output_filepath, current_skeleton_animation_pca)


if __name__ == '__main__':
    root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_skeletons/results_base_video_mp3'
    run_skeletons_pca(root_folder)
