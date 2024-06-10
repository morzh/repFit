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


def trim(in_file: str, out_file: str, start: int, end: int):
    in_file = str(in_file)
    input_stream = ffmpeg.input(in_file)
    probe = ffmpeg.probe(in_file)
    _fps = probe['streams'][0]['r_frame_rate'].split('/')
    fps = float(_fps[0]) / float(_fps[1])
    video = input_stream.trim(start=start/fps, end=end/fps).setpts("PTS-STARTPTS")
    output = ffmpeg.output(video, str(out_file), format="mp4")
    output.run()
