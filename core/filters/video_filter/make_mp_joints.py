from pathlib import Path
import numpy as np

from paths import JOINTS_MP_DPATH
from core.utils.cv.video_reader import VideoReader
from core.utils.io.files_operations import write_json
from constants import model_complexity

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


show = False


def make_mp_joints(video_fpath: Path):
    # Load a model
    save_joints_fpath = JOINTS_MP_DPATH / video_fpath.name
    if save_joints_fpath.is_file():
        print(f"Skip processed video {str(save_joints_fpath)}")
        return
    video_reader = VideoReader(video_fpath)
    mp_joints = []
    frames_with_joints = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity) as pose:
        for frame in video_reader.frame_generator():
            result = pose.process(frame)
            joints = to_np(result)
            if joints is not None:
                mp_joints.append(joints)
                frames_with_joints.append(video_reader.progress-1)

            if show:
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_list=result.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS
                )
                video_reader.imshow(frame)

    mp_joints_fpath = save_joints_fpath.with_suffix('.npy')
    np.save(mp_joints_fpath, mp_joints, allow_pickle=True)
    print(f"Save mediapipe joint in {str(mp_joints_fpath)}")

    write_json(frames_with_joints, save_joints_fpath.with_suffix('.txt'))
    print(f"Save position number of frames which contains "
          f"joint in {str(save_joints_fpath.with_suffix('.txt'))}")


def to_np(landmarks):
    if landmarks.pose_landmarks is None:
        return
    return np.array([(l._x, l._y, l.z, l.visibility) for l in landmarks.pose_landmarks.landmark])


