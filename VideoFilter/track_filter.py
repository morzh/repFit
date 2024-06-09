import json
import numpy as np
from itertools import combinations
from paths import (
    ONE_PERSON_FILTERED_VIDEO_DPATH,
    CUT_VIDEO_DPATH,
    JOINTS2d_EXTRA_INFO_DPATH,
    STABLE_FILTER_DPATH,
    CRED_FILTERED_VIDEO_DPATH,
    JOINTS2d_TRACK_DPATH,
    video_stats_fpath
)
import shutil
from constants import credibility_threshold
from utils.file_reader import write_json, read_pickle


def get_joints_credibility():
    creds = {}
    for bboxes_fpath in STABLE_FILTER_DPATH.glob("*.pickle"):
        tracks = read_pickle(bboxes_fpath)
        creds[bboxes_fpath.name] = {}
        for i, (_id, track_data) in enumerate(tracks.items()):
            frames_with_joints = np.array(list(track_data.keys()))
            frames_with_joints -= frames_with_joints[0]
            joints_probs = np.zeros((frames_with_joints[-1]+1, 17))
            for i, (_idx, frame_data) in enumerate(track_data.items()):
                joints_probs[frames_with_joints[i]] = frame_data['keypoints'][..., -1]
            creds[bboxes_fpath.name][_idx] = joints_probs.sum() / joints_probs.size
    return creds


def filter_by_joints_count():
    """"""
    video_ids = set([p.stem[:-4] for p in list(JOINTS_MP_DPATH.glob("*.npy"))])
    video_joints = {_id: [] for _id in video_ids}
    for joints_fpath in JOINTS_MP_DPATH.glob("*.npy"):
        video_joints[joints_fpath.stem[:-4]].append(joints_fpath)

    for joints_fpath in JOINTS_MP_DPATH.glob("*.npy"):
        joints = np.load(joints_fpath)
        with open(joints_fpath.with_suffix('.txt'), 'r') as file:
            frames_with_joints = json.load(file)

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
    for joints_fpath in JOINTS_MP_DPATH.glob("*.npy"):
        joints = np.load(joints_fpath)
        with open(joints_fpath.with_suffix('.txt'), 'r') as file:
            frames_with_joints = json.load(file)

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
        min_fill_percent = 0.95
        stable_tracks = {}
        for idx, t in tracks.items():
            if len(t) > min_track_length:
                start_idx = int(t[0]['image_id'][:-4])
                stop_idx = int(t[-1]['image_id'][:-4])
                if len(t) / (stop_idx - start_idx) > min_fill_percent:
                    stable_tracks[idx] = t
        if len(stable_tracks) > 1:
            continue
            track_frames = {_id: extract_track_frames(track) for _id, (_, track) in enumerate(stable_tracks.items())}
            n_tracks = len(track_frames)
            for first_idx, second_idx in combinations(range(n_tracks), 2):
                intersection = set(track_frames[first_idx]).intersection(track_frames[second_idx])
                if intersection:
                    #TODO: need improve tracking quolity
                    pass
        for idx, t in stable_tracks.items():
            save_joints_idx_fpath = (JOINTS2d_TRACK_DPATH / joints_fpath.stem+f"_{idx}").with_suffix(".json")
            write_json(t, save_joints_idx_fpath)



def filter_by_joints_count_by_alphapose():
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


def found_video_with_only_one_track_id_by_alphapose():
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
        min_fill_percent = 0.95
        stable_tracks = {}
        for idx, t in tracks.items():
            if len(t) > min_track_length:
                start_idx = int(t[0]['image_id'][:-4])
                stop_idx = int(t[-1]['image_id'][:-4])
                if len(t) / (stop_idx - start_idx) > min_fill_percent:
                    stable_tracks[idx] = t
        if len(stable_tracks) > 1:
            continue
            track_frames = {_id: extract_track_frames(track) for _id, (_, track) in enumerate(stable_tracks.items())}
            n_tracks = len(track_frames)
            for first_idx, second_idx in combinations(range(n_tracks), 2):
                intersection = set(track_frames[first_idx]).intersection(track_frames[second_idx])
                if intersection:
                    #TODO: need improve tracking quolity
                    pass
        for idx, t in stable_tracks.items():
            save_joints_idx_fpath = (JOINTS2d_TRACK_DPATH / joints_fpath.stem+f"_{idx}").with_suffix(".json")
            write_json(t, save_joints_idx_fpath)

def extract_track_frames(track: list) -> list:
    return [int(bbox_data['image_id'][:-4]) for bbox_data in track]


if __name__ == '__main__':
    found_video_with_only_one_track_id()
