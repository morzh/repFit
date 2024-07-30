import os.path
import numpy as np
from datasets.skeletons_processing.core.human36m_alignment_tools import Human36mAlignmentTools
from datasets.skeletons_processing.core.human36m_pca import Human36mPca


def run_skeletons_pca(root_directory: str, save_intermediate_data: bool = False):
    input_videos_folder = os.path.join(root_directory, 'filtered_final_video')
    input_joints_folder = os.path.join(root_directory, 'joints3d')

    output_joints_aligned_heights_folder = os.path.join(root_directory, 'joints3d_heights_aligned')
    output_joints_aligned_aligned_to_global_folder = os.path.join(root_directory, 'joints3d_aligned_to_global_frame')
    output_joints_stacked_folder = os.path.join(root_directory, 'joints3d_stacked')
    output_joints_pca_folder = os.path.join(root_directory, 'joints3d_pca')

    os.makedirs(output_joints_aligned_heights_folder, exist_ok=True)
    os.makedirs(output_joints_aligned_aligned_to_global_folder, exist_ok=True)
    os.makedirs(output_joints_stacked_folder, exist_ok=True)
    os.makedirs(output_joints_pca_folder, exist_ok=True)

    video_files_extensions = ['.mp4', '.mkv', '.webm']
    video_filenames = [f for f in os.listdir(input_videos_folder) if os.path.splitext(f)[1] in video_files_extensions]
    video_filenames.sort()

    skeletons_animations = [np.ndarray] * len(video_filenames)
    for filename_index, video_filename in enumerate(video_filenames):
        video_filename_base = os.path.splitext(video_filename)[0]
        skeleton_animation_filename = video_filename_base + '.npy'
        skeleton_animation_filepath = os.path.join(input_joints_folder, skeleton_animation_filename)
        skeletons_animations[filename_index] = np.load(skeleton_animation_filepath)

    aligned_height_skeletons_animations = Human36mAlignmentTools.align_skeletons_heights(skeletons_animations, in_average=True, verbose=True)
    aligned_to_global_frame_skeletons_animations = Human36mAlignmentTools.align_animated_skeletons_to_global_frame(aligned_height_skeletons_animations)
    stacked_skeletons_animations = Human36mAlignmentTools.stack_joints_coordinates(aligned_to_global_frame_skeletons_animations, use_root_joint_depth=False)

    skeletons_pca = Human36mPca()
    skeletons_pca.fit(stacked_skeletons_animations)

    for video_filename, skeleton_animation in zip(video_filenames, skeletons_animations):
        current_stacked_skeleton_animation = Human36mAlignmentTools.stack_joints_coordinates([skeleton_animation], use_root_joint_depth=False)
        current_skeleton_animation_pca = skeletons_pca.transform(current_stacked_skeleton_animation)
        current_output_filepath = os.path.join(output_joints_pca_folder, video_filename + '.npy')
        np.save(current_output_filepath, current_skeleton_animation_pca)

    if save_intermediate_data:
        for filename_index, video_filename in enumerate(video_filenames):
            current_output_aligned_height_filepath = os.path.join(output_joints_aligned_heights_folder, video_filename + '.npy')
            current_output_aligned_to_global_frame_filepath = os.path.join(output_joints_aligned_aligned_to_global_folder, video_filename + '.npy')
            current_output_stacked_filepath = os.path.join(output_joints_stacked_folder, video_filename + '.npy')

            np.save(current_output_aligned_height_filepath, aligned_height_skeletons_animations[filename_index])
            np.save(current_output_aligned_to_global_frame_filepath, aligned_to_global_frame_skeletons_animations[filename_index])
            np.save(current_output_stacked_filepath, stacked_skeletons_animations[filename_index])


if __name__ == '__main__':
    root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_skeletons/results_base_video_mp3'
    run_skeletons_pca(root_folder, save_intermediate_data=True)
