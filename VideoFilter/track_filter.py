import cv2
import json
import pickle
from pathlib import Path
from matplotlib import pyplot
from multiprocessing import Pool
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import os
import ffmpeg
from paths import JOINTS2d_DPATH, one_person_filtered, CUT_VIDEO_DPATH, JOINTS2d_YOLO_BBOXES_DPATH, JOINTS_DPATH, RESULTS_DPATH, credibility_filtered, JOINTS2d_TRACK_DPATH
from tools.video_reader import VideoReader
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
import shutil

# model = YOLO('yolov8n-pose.yaml')
# results = model(frame)

k = 0.9
show = False
n_processes = 1


def filter_by_joints_credibility():
    creds = {}
    for joints_fpath in JOINTS_DPATH.glob("*.npy"):
        joints = np.load(joints_fpath)
        with open(joints_fpath.with_suffix('.txt'), 'r') as file:
            frames_with_joints = json.load(file)

        if len(joints):
            creds[joints_fpath.with_suffix('.mp4').name] = joints[:, :, 3].sum() / (frames_with_joints[-1] * 33)
        else:
            creds[joints_fpath.with_suffix('.mp4').name] = 0
    creds = {k: v for k, v in reversed(sorted(creds.items(), key=lambda item: item[1]))}
    with open(RESULTS_DPATH/'cut_video_stats.json', 'w', encoding='utf8') as file:
        json.dump(creds, file, ensure_ascii=True)

    for video_name, k in creds.items():
        if k > 0.6:
            shutil.copyfile(CUT_VIDEO_DPATH / video_name, credibility_filtered/video_name)


def mark_as_steady():
    base_dir = Path("/home/ubuntu/PycharmProjects/FitMate/AlphaPose/data_cut")
    joints_dir = base_dir/'joints2d'
    track_joints_dir = base_dir/'track_joints2d'
    out_dir = base_dir/'joints3d'

    with open(base_dir/"camera_steady_segments.pickle", 'rb') as file:
        camera_steady = pickle.load(file)
    #
    # e = []
    # for n in list(JOINTS_DPATH.glob("*.npy")):
    #     d = n.stem.split("_")
    #     if len(d) > 1:
    #         n = "_".join(d[:-1])
    #     else:
    #         n = n.stem
    #
    #     e.append(n)
    # e2 = set(e)

    video_seg = {Path(fname).stem: seg for fname, _, _, seg in zip(*list(camera_steady.values()))}

    with open(RESULTS_DPATH/'cut_video_stats.json', 'r') as file:
        creds = json.load(file)

    fragments = {}
    for joints_fpath in JOINTS_DPATH.glob("*.npy"):
        stemsplit = joints_fpath.stem.split("_")
        if len(stemsplit) > 1:
            stem = "_".join(stemsplit[:-1])
        else:
            stem = joints_fpath.stem
        with open(joints_fpath.with_suffix('.txt'), 'r') as file:
            frames_with_joints = json.load(file)
        fragments[stem] = frames_with_joints

    for joints_fpath in out_dir.glob("*.npy"):
        stem_split = joints_fpath.stem.split("_")
        if len(stem_split) > 2:
            stem = "_".join(stem_split[:-2])
        else:
            stem = joints_fpath.stem
        if stem in fragments and len(fragments[stem]) < 150:
            continue
        if stem in video_seg:
            for (start, stop) in video_seg[stem]:
                start_idx = max(start, fragments[stem][0])
                stop_idx = min(fragments[stem][-1], stop)
                if (stop_idx - start_idx)/len(fragments[stem]) > 0.97:
                    shutil.copyfile(joints_fpath, joints_fpath.parents[1]/'joints3d_steady'/joints_fpath.name)
        else:
            continue

    f=0


def filter_by_joints_count():
    for joints_info_fpath in JOINTS2d_YOLO_BBOXES_DPATH.glob("*.json"):
        with open(joints_info_fpath, 'r') as file:
            joints_info = json.load(file)
        frames_with_only_one_person = sum([len(j) == 1 for n, j in joints_info.items()])
        if frames_with_only_one_person/len(joints_info) < 0.95:
            # if there is more than 1 extra body for each 20 frames - skip the video
            continue
        else:
            shutil.copyfile(JOINTS2d_DPATH /(joints_info_fpath.stem+".mp4.json") , one_person_filtered / joints_info_fpath.name)


def extract_one_track():
    for joints_fpath in one_person_filtered.glob("*.json"):
        with open(joints_fpath, 'r') as file:
            joints_data = json.load(file)
        tracks = {}
        for j in joints_data:
            idxs = j['idx'] if isinstance(j['idx'], list) else [j['idx']]

            for idx in idxs:
                if idx not in tracks:
                    tracks[idx] = []
                tracks[idx].append(j)

        min_track_length = 150
        min_fill_percent = 0.5
        stable_tracks = {}
        for idx, t in tracks.items():
            if len(t) > min_track_length:
                start_idx = int(t[0]['image_id'][:-4])
                stop_idx = int(t[-1]['image_id'][:-4])
                if len(t) / (stop_idx - start_idx) > min_fill_percent:
                    stable_tracks[idx] = t
        if len(stable_tracks)>1:
            continue
        for idx, t in stable_tracks.items():
            save_joints_idx_fpath = (JOINTS2d_TRACK_DPATH / joints_fpath.stem).with_suffix(".json")
            with open(save_joints_idx_fpath, 'w') as file:
                json.dump(t, file)


if __name__ == '__main__':
    extract_one_track()
