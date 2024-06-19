import os.path
import warnings
import numpy as np
import cv2
import yaml

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from filters.steady_camera_filter.core.ocr.craft import Craft
from filters.steady_camera_filter.core.ocr.easy_ocr import EasyOcr
from filters.steady_camera_filter.core.ocr.tesseract_ocr import TesseractOcr
from filters.steady_camera_filter.core.steady_camera_coarse_filter import SteadyCameraCoarseFilter
from cv_utils.video_segments_writer import VideoSegmentsWriter
from filters.steady_camera_filter.core.video_segments import VideoSegments

segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


def yaml_parameters(filepath: str) -> dict:
    """
    Description:
        Read yaml file
    :param filepath: filepath to .yaml file
    :return: dictionary with yaml data
    """
    parameters = None
    if not os.path.exists(filepath):
        raise FileNotFoundError('Parameters YAML file does not exist')

    with open('steady_camera_filter_parameters.yaml') as f:
        try:
            parameters = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    if parameters is None:
        raise ValueError('Something wrong with the YAML file')

    return parameters


def video_resolution_check(video_filepath: str, minimum_dimension_size: int = 360) -> bool:
    """
    Description:
        Check if video size is greater than a given threshold.
    :return: True if (width, height) >  minimum_dimension_size, False otherwise
    """
    video_capture = cv2.VideoCapture(video_filepath)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maximum_dimension = max(video_width, video_height)

    if maximum_dimension > minimum_dimension_size:
        return True
    return False


def extract_coarse_steady_camera_filter_video_segments(video_filepath: str, parameters: dict) -> VideoSegments:
    """
    Description:
        Extract segments from video in frames, where camera is steady (meets steadiness criteria of coarse steady camera filter).
    :param video_filepath: filepath of the video
    :param parameters: parameters for steady camera filter
    """
    if parameters['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        print(video_filename)

    image_registration_parameters = parameters['image_registration']
    number_frames_to_average = image_registration_parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    match image_registration_parameters['ocr_model']:
        case 'craft':
            craft_parameters = parameters['text_mask']['craft']
            ocr_model = Craft(use_cuda=craft_parameters['use_cuda'],
                              use_refiner=craft_parameters['use_refiner'],
                              use_float16=craft_parameters['use_float_16'])
        case 'easy_ocr':
            easyocr_parameters = parameters['text_mask']['easy_ocr']
            ocr_model = EasyOcr(minimum_ocr_confidence=easyocr_parameters['minimum_ocr_confidence'],
                                minimal_resolution=easyocr_parameters['minimal_resolution'])
        case 'tesseract':
            tesseract_parameters = parameters['text_mask']['tesseract']
            ocr_model = TesseractOcr()
        case _:
            raise ValueError('Models for masking text other than Craft, EasyOCR or Tesseract are not provided.')

    camera_filter = SteadyCameraCoarseFilter(video_filepath,
                                             ocr_model,
                                             number_frames_to_average=number_frames_to_average,
                                             maximum_shift_length=image_registration_parameters['maximum_shift_length'],
                                             poc_maximum_image_dimension=image_registration_parameters['poc_maximum_dimension'],
                                             registration_minimum_confidence=image_registration_parameters['poc_minimum_confidence'])

    camera_filter.process(parameters['poc_show_averaged_frames_pair'])
    steady_segments = camera_filter.calculate_steady_camera_ranges()
    steady_segments = camera_filter.filter_segments_by_time(steady_segments, parameters['minimum_steady_camera_time_segment'])

    if parameters['poc_registration_verbose']:
        camera_filter.print_registration_results()
    if parameters['verbose_segments']:
        print(steady_segments)

    return steady_segments


def write_video_segments(video_filepath, output_folder, video_segments: VideoSegments, parameters: dict) -> None:
    """
    Description:
        Cuts input video according video_segments information.
    :param video_filepath: input video filepath
    :param output_folder: output folder for trimmed videos
    :param video_segments information about video segments to trim
    :param parameters: parameters to write videos
    """
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f'File {video_filepath} does not exist')

    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=video_segments.video_fps,
                                                scale_factor=parameters['scale_factor'])
    video_segments_writer.write(video_segments, write_gaps=parameters['use_segments_gaps'])


def extract_and_write_steady_camera_segments(video_source_filepath, videos_target_folder, parameters) -> None:
    """
    Description:
        Convenient function for multiprocessing. It violates single responsibility principle, but who cares.
    :param video_source_filepath: source video filepath
    :param videos_target_folder: output folder for segmented videos
    :param parameters: extraction parameters
    """
    minimum_resolution = parameters['video_segments_extraction']['resolution_filter']['minimum_dimension_resolution']
    if not video_resolution_check(video_source_filepath, minimum_dimension_size=minimum_resolution):
        return

    video_segments = extract_coarse_steady_camera_filter_video_segments(video_source_filepath, parameters['video_segments_extraction'])
    write_video_segments(video_source_filepath, videos_target_folder, video_segments, parameters['video_segments_output'])
