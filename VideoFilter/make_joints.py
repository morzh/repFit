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
from paths import YOLO_BBOXES_DPATH, video_dpath, CUT_VIDEO_DPATH, FILTERED_VIDEO_DPATH, JOINTS_DPATH, RESULTS_DPATH
from tools.video_reader import VideoReader
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# model = YOLO('yolov8n-pose.yaml')
# results = model(frame)

k = 0.9
show = False
n_processes = 12


def make_joints(video_fpath: Path):
    # Load a model
    save_joints_fpath = JOINTS_DPATH / video_fpath.name
    if save_joints_fpath.is_file():
        print(f"Skip processed video {str(save_joints_fpath)}")
        return
    video_reader = VideoReader(video_fpath)
    results = []
    stats = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
        for frame in video_reader.frame_generator():
            result = pose.process(frame)
            joints = to_np(result)
            if joints is None:
                pass
            else:
                results.append(joints)
                stats.append(video_reader._progress.n-1)

            if show:
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_list=result.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS
                )
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key == 27:  # if ESC is pressed, exit loop
                    cv2.destroyAllWindows()
                    break

    np.save(save_joints_fpath.with_suffix('.npy'), results, allow_pickle=True)
    with open(save_joints_fpath.with_suffix('.txt'), 'w') as file:
        json.dump(stats, file)

    # if sum(mean)/len(mean) > k:
    #     video_fpath.rename(FILTERED_VIDEO_DPATH/video_fpath.name)


def to_np(landmarks):
    if landmarks.pose_landmarks is None:
        return
    return np.array([(l.x, l.y, l.z, l.visibility) for l in landmarks.pose_landmarks.landmark])


def multiprocess_make_joints():
    videos = list(CUT_VIDEO_DPATH.glob('*'))
    with Pool(n_processes) as p:
        p.map(make_joints, videos)


if __name__ == '__main__':
    # make_joints(CUT_VIDEO_DPATH/'sentadillas y desplantes-v1ToL_2IBac_0.mp4')
    multiprocess_make_joints()
