import glob
import os
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar
from datasets.skeletons_processing.core.human36m_alignment_tools import Human36mAlignmentTools

Float32 = TypeVar("Float32", bound=np.float32)
joints_batch = Annotated[npt.NDArray[Float32], Literal["N", 17, 3]]


def select_main_skeleton_animation(skeletons: list[joints_batch]) -> np.ndarray:
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
