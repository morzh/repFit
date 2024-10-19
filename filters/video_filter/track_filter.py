import json
import numpy as np
from itertools import combinations
from paths import STABLE_FILTER_DPATH, JOINTS2d_TRACK_DPATH
from constants import credibility_threshold, multi_person_threshold
from utils.io.file_read_write import write_json, read_pickle


def filter_by_joints_credibility(threshold: float = credibility_threshold) -> dict:
    result = {}
    for bboxes_fpath in STABLE_FILTER_DPATH.glob("*.pickle"):
        tracks = read_pickle(bboxes_fpath)
        result[bboxes_fpath.stem] = []
        for _, (_id, track_data) in enumerate(tracks.items()):
            frames_with_joints = np.array(list(track_data.keys()))
            frames_with_joints -= frames_with_joints[0]
            joints_probs = np.zeros((frames_with_joints[-1] + 1, 17))

            for i, (_idx, frame_data) in enumerate(track_data.items()):
                joints_probs[frames_with_joints[i]] = frame_data['keypoints'][..., -1]
            average_cred = joints_probs.sum() / joints_probs.size
            if average_cred > threshold:
                result[bboxes_fpath.stem].append(_id)
                # debug mode.
                # result[bboxes_fpath.stem].append((_id, average_cred))
    return result


def filter_by_joints_count() -> dict:
    """"""
    result = {}
    # intersection_percents = {}
    for bboxes_fpath in STABLE_FILTER_DPATH.glob("*.pickle"):
        # intersection_percents[bboxes_fpath.stem] = []
        tracks = read_pickle(bboxes_fpath)
        result[bboxes_fpath.stem] = []

        if len(tracks) == 1:
            # if video contains only one stable track, just mark it good and continue
            result[bboxes_fpath.stem].append(list(tracks)[0])
            # intersection_percents[bboxes_fpath.stem].append(0)
            continue

        track_frames = {_id: list(track.keys()) for _id, track in tracks.items()}
        track_ids = list(tracks)
        drop_tracks = set()
        for n_first, n_second in combinations(range(len(track_frames)), 2):
            first_idx = track_ids[n_first]
            second_idx = track_ids[n_second]
            intersection = set(track_frames[first_idx]).intersection(track_frames[second_idx])
            for idx in [first_idx, second_idx]:
                intersection_percent = len(intersection) / len(track_frames[idx])
                # intersection_percents[bboxes_fpath.stem].append((idx, intersection_percent))
                if intersection_percent > multi_person_threshold:
                    drop_tracks.add(idx)
        result[bboxes_fpath.stem].extend(list(set(track_ids).difference(drop_tracks)))
    # return result, intersection_percents
    return result


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

def extract_track_frames(track: list) -> list:
    return [int(bbox_data['image_id'][:-4]) for bbox_data in track]


if __name__ == '__main__':
    found_video_with_only_one_track_id()
