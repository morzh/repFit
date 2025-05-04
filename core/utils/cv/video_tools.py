import os

import cv2
import numpy as np
import ffmpeg

from core.utils.cv.video_file_segments import VideoFileSegments
from core.utils.cv.video_writer import VideoWriter
from core.utils.io.files_operations import extract_name_extension_from_filepath


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


def video_resolution_check(video_filepath: str, minimum_dimension_size: int = 360) -> bool:
    """
    Description:
        Check if video size is greater than a given threshold.

    :param video_filepath: filepath of the video
    :param minimum_dimension_size: minimum(video width, video height) threshold

    :return: True if (width, height) >  minimum_dimension_size, False otherwise
    """
    video_capture = cv2.VideoCapture(video_filepath)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maximum_dimension = max(video_width, video_height)

    if maximum_dimension > minimum_dimension_size:
        return True

    return False


def write_segments_values(video_segments: VideoFileSegments, input_filepath: os.PathLike, output_folder: os.PathLike, filter_name: str = 'steady') -> None:
    """
    Description:
        Write segments values. This feature is for debug purposes

    :param video_segments: video segments
    :param input_filepath:
    :param output_folder:
    :param filter_name: filter name (e.g. steady or non-steady)
    """
    video_filename_base, _ = extract_name_extension_from_filepath(input_filepath)
    segments_values_filename = f'{video_filename_base}__{filter_name}__.npy'
    segments_values_filepath = os.path.join(output_folder, segments_values_filename)

    video_segments.write(segments_values_filepath)
