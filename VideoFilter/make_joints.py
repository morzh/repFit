import cv2
import json
from pathlib import Path
import numpy as np

from paths import JOINTS_DPATH
from cv_tools.video_reader import VideoReader
from utils.file_reader import write_json

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


show = False


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
    write_json(stats, save_joints_fpath.with_suffix('.txt'))


def to_np(landmarks):
    if landmarks.pose_landmarks is None:
        return
    return np.array([(l.x, l.y, l.z, l.visibility) for l in landmarks.pose_landmarks.landmark])


