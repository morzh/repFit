import os.path
import numpy as np
from datasets.skeletons_processing.core.h36m_animated_skeleton_tools import H36mAnimatedSkeletonTools


root_folder = '/media/anton/4c95a564-35ea-40b5-b747-58d854a622d0/home/anton/work/fitMate/datasets/squats_2022_skeletons/results_base_video_mp3'
input_videos_folder = os.path.join(root_folder, 'filtered_final_video')
input_joints_3d_folder = os.path.join(root_folder, 'joints3d')
output_joints_3d_stacked_folder = os.path.join(root_folder, 'joints3d_stacked')
video_files_extensions = ['.mp4', '.mkv', '.webm']
video_filenames = [f for f in os.listdir(input_videos_folder) if os.path.splitext(f)[1] in video_files_extensions]
video_filenames.sort()

os.makedirs(output_joints_3d_stacked_folder, exist_ok=True)
joints_3d_data = {}

for video_filename in video_filenames:
    video_filename_base = os.path.splitext(video_filename)[0]
    joints_3d_filename = video_filename_base + '.npy'
    joints_3d_filepath = os.path.join(input_joints_3d_folder, joints_3d_filename)
    joints_3d_data[video_filename] = np.load(joints_3d_filepath)


joints_3d_data['Arg6JZg4UdA_1_0.mp4'] = joints_3d_data['Arg6JZg4UdA_1_0.mp4'][:390]
for video_filename, skeleton_animation in joints_3d_data.items():
    current_number_animation_frames = skeleton_animation.shape[0]

    current_root_animation = skeleton_animation[:, 0]
    current_root_animation_vertical_component = current_root_animation[:, 1].reshape(-1, 1)

    current_aligned_skeleton_animation = H36mAnimatedSkeletonTools.align_to_global_frame(skeleton_animation)
    current_aligned_skeleton_animation = np.delete(current_aligned_skeleton_animation, 0, axis=1)

    current_stacked_skeleton_animation = current_aligned_skeleton_animation.reshape(current_number_animation_frames, -1)
    current_stacked_skeleton_animation = np.hstack((current_stacked_skeleton_animation, current_root_animation_vertical_component))
    np.save(os.path.join(output_joints_3d_stacked_folder, video_filename), current_stacked_skeleton_animation)
