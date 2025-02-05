# Check difference in difficult screens between models
# 1) yolo8
# 2) mediapipe
# 3) Alphapose + MotionBert

import json
from paths import RESULTS_ROOT, DATASETS_DPATH

from pathlib import Path
import numpy as np
import cv2



img_dpath = DATASETS_DPATH / 'images'
result_dpath = RESULTS_ROOT / 'compare_joint_models'
result_dpath.mkdir(exist_ok=True, parents=True)


def join_images_to_video(images_dpath, video_fpath: Path):
    video_writer = cv2.VideoWriter(str(video_fpath), cv2.VideoWriter_fourcc(*'MP4V'), 10, (1920, 1080))

    for frame_fpath in images_dpath.glob("*.png"):
        frame = cv2.imread(str(frame_fpath))
        frame = cv2.resize(frame, (1920, 1080))
        video_writer.write(frame)
    video_writer.release()
    print(f"Save cut video {str(video_fpath)}")

def save_json(data, fpath):
    with open(fpath, 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def read_json(fpath):
    with open(fpath, 'r') as file:
        return json.load(file)

def make_mp_joints():
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    model_complexity = 2
    results = {}
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity) as pose:
        for frame_fpath in img_dpath.glob("*.png"):
            frame =  cv2.imread(str(frame_fpath))
            landmarks = pose.process(frame)
            if landmarks.pose_landmarks is None:
                continue
            joints =  [(l.x, l.y, l.z, l.visibility) for l in landmarks.pose_landmarks.landmark]
            results[frame_fpath.stem] = joints
            mp_drawing.draw_landmarks(
                frame,
                landmark_list=landmarks.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS
            )
            cv2.imwrite(str(result_dpath / (frame_fpath.stem + "_mp.png")), frame)
    save_json(results, result_dpath / "mp.json")


def make_yolo8_joints(model_name: str = '../models/yolov8x-pose.pt'):
    from ultralytics import YOLO
    detector_model = YOLO(model_name)
    detector_params = dict(
        classes=0,
        persist=True,
        conf=0.7,
        iou=0.7,
        show=False,
        verbose=False
    )
    results = {}
    for i, frame_fpath in enumerate(img_dpath.glob("*.png")):
        frame = cv2.imread(str(frame_fpath))
        result = detector_model.track(frame, **detector_params)[0]

        idxs = result.boxes.id
        if idxs is not None:
            for bbox, keypoints in zip(result.boxes.data.numpy(), result.keypoints.data.cpu().numpy()):
                try:
                    idx = int(bbox[4])
                except Exception as ex:
                    continue
                results[frame_fpath.stem] = {
                    "bbox": bbox.tolist(),
                    "keypoints": keypoints.tolist()
                }

    save_json(results, result_dpath / "yolo.json")


def convert_motion_bert_joints():
    #TODO: dosn't finished
    return
    joints = np.load(result_dpath/'MotionBert.npy')
    # data1 = read_json(result_dpath / 'AlphaPose.json')
    # data2 = read_json(result_dpath / 'AlphaPose_extra.json')
    results = {}
    for i, frame_fpath in enumerate(img_dpath.glob("*.png")):
        results[frame_fpath.stem] = joints[i, ...].tolist()

    save_json(results, result_dpath / "motion_bert.json")


def yolo_nas_pose():
    # python3.10
    # https://habr.com/ru/articles/772558/
    # https://github.com/Deci-AI/super-gradients

    import torch
    import os
    import pathlib
    from super_gradients.training import models
    from super_gradients.common.object_names import Models

    model = models.get(
        "yolo_nas_pose_l",
                       # checkpoint_path=
                       # "file://home/ubuntu/PycharmProjects/FitMate/repFit/models/yolo_nas_pose_l_coco_pose.pth",
                       pretrained_weights="coco_pose")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    confidence = 0.6

    for i, frame_fpath in enumerate(img_dpath.glob("*.png")):
        frame = cv2.imread(str(frame_fpath))
        res = model.predict(frame, conf=confidence)
        res.save("./results/"+frame_fpath.name)
        p=0



if __name__ == '__main__':
    yolo_nas_pose()
    # make_mp_joints()
    # make_yolo8_joints()
    # convert_motion_bert_joints()



