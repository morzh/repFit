import cv2
import json
import pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
import ffmpeg
from paths import INFO_DPATH, VIDEO_DPATH, CUT_VIDEO_DPATH

n_processes = 12



with open(base_dir/"camera_steady_segments.pickle", 'rb') as file:
    camera_steady = pickle.load(file)
video_seg = {yid: seg for fname, _, yid, seg in zip(*list(camera_steady.values()))}

def run_trim():
    video_fname = '140kg Squat 6 reps-viZUvjS0RGY.mkv'
    video_fpath = VIDEO_DPATH/video_fname
    with open((INFO_DPATH/video_fname).with_suffix('.json'), 'r') as file:
        video_info = json.load(file)

    cap = cv2.VideoCapture(str(video_fpath))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result_fpath = CUT_VIDEO_DPATH / video_fname
    trim(str(video_fpath), str(result_fpath), video_info['1'][0]*fps, video_info['1'][-1]*fps)


def trim(in_file: str, out_file: str, start: int, end: int):
    if os.path.exists(out_file):
        os.remove(out_file)

    in_file_probe_result = ffmpeg.probe(in_file)
    in_file_duration = in_file_probe_result.get("format", {}).get("duration", None)
    print(in_file_duration)

    input_stream = ffmpeg.input(in_file)

    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start, end=end).setpts(pts)
    output = ffmpeg.output(video, out_file, format="mp4")
    output.run()


def cut_video_on_boxes(video_fpath):
    video_fname = video_fpath.stem
    video_youtube_id = video_fname[-11:]
    if video_youtube_id not in video_seg:
        print(f"skip video {video_fname} by steady filter")
        return

    bboxes_fpath = (INFO_DPATH / video_fpath.name).with_suffix('.pickle')
    with open(bboxes_fpath, 'rb') as file:
        tracks = pickle.load(file)
    # bbox_idxs = list(tracks.keys())
    for track_row_n, (track_idx, track) in enumerate(tracks.items()):

        cut_video_fpath = (CUT_VIDEO_DPATH / (video_fname[-11:] + f"_{track_row_n}")).with_suffix(".mp4")
        if cut_video_fpath.is_file():
            print(f"skip processed track {str(cut_video_fpath)}")
            continue

        bbox_array = found_bbox_array(track, track_idx)
        bbox_min_h, bbox_max_h, bbox_min_w, bbox_max_w = found_bbox_boarders(bbox_array)

        bbox_indexes = list(bbox_array.keys())
        bbox_start_idx = bbox_indexes[0]
        bbox_stop_idx = bbox_indexes[-1]

        seg_with_bboxes = []
        for seg in video_seg[video_youtube_id]:
            new_seg = (max(seg[0], bbox_start_idx), min(seg[1], bbox_stop_idx))
            if new_seg[1] > new_seg[0] and (new_seg[1] - new_seg[0]) > 120:
                seg_with_bboxes.append(new_seg)
        if not seg_with_bboxes:
            print(f"skip track {str(cut_video_fpath)} by steady filter")
            continue
        for seg_n, seg in enumerate(seg_with_bboxes):
            cut_video_fpath = (CUT_VIDEO_DPATH / (video_fname[-11:] + f"_{track_row_n}_{seg_n}")).with_suffix(".mp4")

            video_reader = cv2.VideoCapture(str(video_fpath))
            n_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
            success, frame = video_reader.read()
            fps = int(video_reader.get(cv2.CAP_PROP_FPS))
            video_writer = cv2.VideoWriter(str(cut_video_fpath), cv2.VideoWriter_fourcc(*'MP4V'), fps, (bbox_max_w-bbox_min_w, bbox_max_h-bbox_min_h))
            progress = tqdm(range(n_frames))
            while success:
                if success:
                    if progress.n > seg[0] and progress.n < seg[1]:
                        bbox_img = frame[bbox_min_h:bbox_max_h, bbox_min_w:bbox_max_w, :]
                        video_writer.write(bbox_img)
                    elif progress.n > seg[1]:
                        break
                    progress.update()
                success, frame = video_reader.read()
            video_writer.release()


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

def cut_by_bbox(image: np.ndarray, bbox: np.ndarray, w, h) -> np.ndarray:
    image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    boarder_w = w - (bbox[2] - bbox[0])
    left = boarder_w // 2
    right = boarder_w - left

    boarder_h = h - (bbox[3] - bbox[1])
    top = boarder_h // 2
    buttom = boarder_h - top

    bbox_img = cv2.copyMakeBorder(image, top, buttom, left, right,
                                  borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return bbox_img


def run_multiprocess():
    videos = list(video_dpath.glob('*'))
    cut_video_on_boxes(videos[28])
    with Pool(n_processes) as p:
        p.map(cut_video_on_boxes, videos)


if __name__ == '__main__':
    run_multiprocess()
