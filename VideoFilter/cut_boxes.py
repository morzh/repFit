from typing import List
import cv2
from pathlib import Path
import numpy as np

from paths import YOLO_BBOXES_DPATH, STEADY_VIDEO_DPATH, RESULTS_ROOT, STABLE_FILTER_DPATH, FILTERED_VIDEO_DPATH
from utils.file_reader import read_pickle
from cv_tools.video_reader import VideoReader
from constants import min_fragment_n_frames

camera_steady = read_pickle(RESULTS_ROOT/"camera_steady_segments.pickle")
video_seg = {yid: seg for fname, _, yid, seg in zip(*list(camera_steady.values()))}


def cut_video_by_yolo_boxes(video_fpath: Path):
    video_fname = video_fpath.stem
    video_youtube_id = video_fname[-11:]
    if video_youtube_id not in video_seg:
        print(f"skip video {video_fname} by steady filter")
        return

    bboxes_fpath = (YOLO_BBOXES_DPATH / video_fpath.name).with_suffix('.pickle')
    tracks = read_pickle(bboxes_fpath)

    for track_row_n, (track_idx, track) in enumerate(tracks.items()):
        cut_video_fpath = (STEADY_VIDEO_DPATH / (video_fname[-11:] + f"_{track_row_n}_0")).with_suffix(".mp4")
        if cut_video_fpath.is_file():
            print(f"skip processed track {str(cut_video_fpath)}")
            continue

        bbox_array = found_bbox_array(track, track_idx)

        fragment_with_bboxes = get_fragments(bbox_array, video_youtube_id, min_fragment_n_frames)
        if not fragment_with_bboxes:
            print(f"skip track {str(cut_video_fpath)} by steady filter")
            continue

        bbox_min_h, bbox_max_h, bbox_min_w, bbox_max_w = found_bbox_boarders(bbox_array)
        for seg_n, seg in enumerate(fragment_with_bboxes):
            video_reader = VideoReader(video_fpath)

            fragment_fname = video_fname[-11:] + f"_{track_row_n}_{seg_n}.mp4"
            cut_video_fpath = STEADY_VIDEO_DPATH / fragment_fname
            video_writer = cv2.VideoWriter(
                str(cut_video_fpath),
                cv2.VideoWriter_fourcc(*'MP4V'),
                video_reader.fps,
                (bbox_max_w-bbox_min_w, bbox_max_h-bbox_min_h)
            )

            for frame in video_reader.frame_generator():
                if video_reader.progress > seg[0] and video_reader.progress < seg[1]:
                    bbox_img = frame[bbox_min_h:bbox_max_h, bbox_min_w:bbox_max_w, :]
                    video_writer.write(bbox_img)
                elif video_reader.progress > seg[1]:
                    break
            video_writer.release()
            print(f"Save cut video {str(cut_video_fpath)}")


def cut_videos_by_filters(videos: List[Path], filter_result: dict):
    for video_fpath in videos:
        video_fname = video_fpath.stem
        if video_fname not in filter_result:
            continue
        bboxes_fpath = STABLE_FILTER_DPATH / (video_fname+".pickle")
        tracks = read_pickle(bboxes_fpath)
        for track_row_n, (track_idx, track) in enumerate(tracks.items()):
            cut_video_fpath = (FILTERED_VIDEO_DPATH / (bboxes_fpath.stem + f"_{track_row_n}")).with_suffix(".mp4")
            if cut_video_fpath.is_file():
                print(f"skip processed track {str(cut_video_fpath)}")
                continue
            bbox_array = {i: frame_data['bbox'] for i, frame_data in track.items()}
            bbox_min_h, bbox_max_h, bbox_min_w, bbox_max_w = found_bbox_boarders(bbox_array)
            frames_with_bbox = list(bbox_array.keys())
            video_reader = VideoReader(video_fpath)
            video_writer = cv2.VideoWriter(
                str(cut_video_fpath),
                cv2.VideoWriter_fourcc(*'MP4V'),
                video_reader.fps,
                (bbox_max_w-bbox_min_w, bbox_max_h-bbox_min_h)
            )

            for frame in video_reader.frame_generator():
                if video_reader.progress > frames_with_bbox[0] and video_reader.progress < frames_with_bbox[-1]:
                    bbox_img = frame[bbox_min_h:bbox_max_h, bbox_min_w:bbox_max_w, :]
                    video_writer.write(bbox_img)
                elif video_reader.progress > frames_with_bbox[-1]:
                    break
            video_writer.release()
            print(f"Save cut video {str(cut_video_fpath)}")


def get_fragments(bbox_array: dict, video_id: str, min_length: int) -> list:
    bbox_indexes = list(bbox_array.keys())
    bbox_start_idx = bbox_indexes[0]
    bbox_stop_idx = bbox_indexes[-1]
    fragment_with_bboxes = []
    for seg in video_seg[video_id]:
        new_seg = (max(seg[0], bbox_start_idx), min(seg[1], bbox_stop_idx))
        if new_seg[1] > new_seg[0] and (new_seg[1] - new_seg[0]) > min_length:
            fragment_with_bboxes.append(new_seg)
    return fragment_with_bboxes


def found_bbox_array(track: dict, track_idx: int) -> dict:
    bbox_array = {}
    for idx, bboxes in track.items():
        for _box in bboxes:
            if _box[4] == track_idx:
                bbox_array[idx] = np.array(_box[:4], dtype="int16")
                break
    return bbox_array


def found_bbox_max_wh(bbox_array: dict) -> (int, int):
    bbox_max_h = 0
    bbox_max_w = 0
    for idx, bbox in bbox_array.items():
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > bbox_max_w:
            bbox_max_w = w
        if h > bbox_max_h:
            bbox_max_h = h
    return int(bbox_max_h) + 2, int(bbox_max_w) + 2


def found_bbox_boarders(bbox_array: dict) -> (int, int, int, int):
    bbox_min_h = np.inf
    bbox_max_h = 0
    bbox_min_w = np.inf
    bbox_max_w = 0
    for idx, bbox in bbox_array.items():
        if bbox[0] < bbox_min_w:
            bbox_min_w = bbox[0]
        if bbox[2] > bbox_max_w:
            bbox_max_w = bbox[2]

        if bbox[1] < bbox_min_h:
            bbox_min_h = bbox[1]
        if bbox[3] > bbox_max_h:
            bbox_max_h = bbox[3]

    return int(bbox_min_h), int(bbox_max_h), int(bbox_min_w), int(bbox_max_w)

