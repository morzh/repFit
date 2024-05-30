import json
import numpy as np
from paths import (
    ONE_PERSON_FILTERED_VIDEO_DPATH,
    CUT_VIDEO_DPATH,
    JOINTS2d_EXTRA_INFO_DPATH,
    JOINTS_MP_DPATH,
    CRED_FILTERED_VIDEO_DPATH,
    JOINTS2d_TRACK_DPATH,
    video_stats_fpath
)
import shutil
from constants import credibility_threshold
from utils.file_reader import write_json


def filter_video_by_joints_credibility(threshold: float = credibility_threshold):
    creds = {}
    for joints_fpath in JOINTS_MP_DPATH.glob("*.npy"):
        joints = np.load(joints_fpath)
        with open(joints_fpath.with_suffix('.txt'), 'r') as file:
            frames_with_joints = json.load(file)

        if len(joints):
            creds[joints_fpath.with_suffix('.mp4').name] = joints[:, :, 3].sum() / (frames_with_joints[-1] * 33)
        else:
            creds[joints_fpath.with_suffix('.mp4').name] = 0
    creds = {k: v for k, v in reversed(sorted(creds.items(), key=lambda item: item[1]))}

    write_json(creds, video_stats_fpath)

    for video_name, k in creds.items():
        if k > threshold:
            shutil.copyfile(
                CUT_VIDEO_DPATH / video_name,
                CRED_FILTERED_VIDEO_DPATH / video_name
            )


def filter_by_joints_count():
    for joints_info_fpath in JOINTS2d_EXTRA_INFO_DPATH.glob("*.json"):
        with open(joints_info_fpath, 'r') as file:
            joints_info = json.load(file)
        frames_with_only_one_person = sum([len(j) == 1 for n, j in joints_info.items()])
        if frames_with_only_one_person / len(joints_info) < 0.95:
            # if there is more than 1 extra body for each 20 frames - skip the video
            continue
        else:
            shutil.copyfile(
                JOINTS2d_TRACK_DPATH / joints_info_fpath.name,
                ONE_PERSON_FILTERED_VIDEO_DPATH / joints_info_fpath.name
            )


def found_video_with_only_one_track_id():
    for joints_fpath in ONE_PERSON_FILTERED_VIDEO_DPATH.glob("*.json"):
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
        if len(stable_tracks) > 1:
            continue
        for idx, t in stable_tracks.items():
            save_joints_idx_fpath = (JOINTS2d_TRACK_DPATH / joints_fpath.stem).with_suffix(".json")
            write_json(t, save_joints_idx_fpath)


if __name__ == '__main__':
    found_video_with_only_one_track_id()
