import cv2
import numpy as np
import ffmpeg


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


def run_trim():
    # cut part of a video by time
    video_fname = '140kg Squat 6 reps-viZUvjS0RGY.mkv'
    video_fpath = VIDEO_DPATH/video_fname
    with open((YOLO_BBOXES_DPATH/video_fname).with_suffix('.json'), 'r') as file:
        video_info = json.load(file)

    cap = cv2.VideoCapture(str(video_fpath))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result_fpath = CUT_VIDEO_DPATH / video_fname
    trim(str(video_fpath), str(result_fpath), video_info['1'][0]*fps, video_info['1'][-1]*fps)


def trim(in_file: str, out_file: str, start: int, end: int):
    in_file_probe_result = ffmpeg.probe(in_file)
    in_file_duration = in_file_probe_result.get("format", {}).get("duration", None)
    print(in_file_duration)

    input_stream = ffmpeg.input(in_file)

    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start, end=end).setpts(pts)
    output = ffmpeg.output(video, out_file, format="mp4")
    output.run()