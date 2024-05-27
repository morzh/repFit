import cv2
import json
import pickle
from multiprocessing import Pool
from ultralytics import YOLO
from tqdm import tqdm
from paths import INFO_DPATH, VIDEO_DPATH
from pathlib import Path

min_occurrence_rate = 0.5
min_track_length_sec = 4
n_processes = 12


def run():
    videos = list(VIDEO_DPATH.glob('*'))
    with Pool(n_processes) as p:
        p.map(process_video, videos)


def run_with_subdir():
    for video_subdir in VIDEO_DPATH.glob('*'):
        if not video_subdir.is_dir():
            continue
        videos = list(video_subdir.glob('*'))
        with Pool(n_processes) as p:
            p.map(process_video, videos)


def process_video(video_fpath):
    video_fname = video_fpath.name
    try:
        result_fpath = (INFO_DPATH / video_fname).with_suffix('.pickle')
        if result_fpath.is_file():
            print(f"Skip processed video {video_fname}. Result fpath: {str(result_fpath)}")
            return
        result_fpath.parents[0].mkdir(exist_ok=True, parents=True)
        print(f"Start processing of video {video_fname}")

        model = YOLO('yolov8x.pt')
        cap = cv2.VideoCapture(str(video_fpath))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-2
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        min_track_length_frames = min_track_length_sec * fps

        tracks = {}
        success, frame = cap.read()

        progress = tqdm(range(n_frames))
        while success:
            base_frame_size = frame.shape
            if success:
                if frame.shape != base_frame_size:
                    print(f"Found frame with not the same shape {frame.shape}")
                else:
                    results = model.track(frame, classes=0, persist=True, conf=0.7, iou=0.7, show=False, tracker='tracker_conf.yaml', verbose=False)
                    idxs = results[0].boxes.id
                    if idxs is not None:
                        for idx in results[0].boxes.id:
                            idx = int(idx)
                            if idx not in tracks:
                                tracks[idx] = {}
                            tracks[idx][progress.n] = results[0].boxes.data.numpy()
                progress.update()

            success, frame = cap.read()

        stable_tracks = {}
        for idx, t in tracks.items():
            if len(t) > min_track_length_frames:
                idxs = list(t.keys())
                start_idx = int(idxs[0])
                stop_idx = int(idxs[-1])
                if len(t) / (stop_idx - start_idx) > min_occurrence_rate:
                    stable_tracks[idx] = t

        with open(result_fpath, 'wb') as file:
            pickle.dump(stable_tracks, file)
            print(f"Save results {str(result_fpath)}")
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    run()
