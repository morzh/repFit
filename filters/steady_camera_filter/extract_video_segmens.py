import os.path
import shutil
import warnings
from typing import Annotated, Literal

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from cv_utils.video_segments_writer import VideoSegmentsWriter
from filters.steady_camera_filter.core.ocr.craft import Craft
from filters.steady_camera_filter.core.ocr.easy_ocr import EasyOcr
from filters.steady_camera_filter.core.ocr.tesseract_ocr import TesseractOcr
from filters.steady_camera_filter.core.steady_camera_coarse_filter import \
    SteadyCameraCoarseFilter
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
    :raises ValueError: when trying to use text masking with neural network models other than CRAFT, EasyOCR or Tesseract.
    """
    if parameters['verbose_filename']:
        video_filename = os.path.basename(video_filepath)
        print(video_filename)

    steady_camera_coarse_parameters = parameters['steady_camera_coarse_filter']
    number_frames_to_average = steady_camera_coarse_parameters['number_frames_to_average']
    if number_frames_to_average < 5:
        warnings.warn(f'Value {number_frames_to_average} of number_frames_to_average is low, results could be non applicable')

    match steady_camera_coarse_parameters['text_mask_model']:
        case 'craft':
            craft_parameters = steady_camera_coarse_parameters['text_mask_models']['craft']
            ocr_model = Craft(use_cuda=craft_parameters['use_cuda'],
                              use_refiner=craft_parameters['use_refiner'],
                              use_float16=craft_parameters['use_float_16'])
        case 'easy_ocr':
            easyocr_parameters = steady_camera_coarse_parameters['text_mask_models']['easy_ocr']
            ocr_model = EasyOcr(confidence_threshold=easyocr_parameters['confidence_threshold'],
                                minimal_resolution=easyocr_parameters['minimal_resolution'])
        case 'tesseract':
            tesseract_parameters = steady_camera_coarse_parameters['text_mask_models']['tesseract']
            ocr_model = TesseractOcr(confidence=tesseract_parameters['confidence_threshold'])
        case _:
            raise ValueError('Models for masking text other than Craft, EasyOCR or Tesseract are not provided.')

    camera_filter = SteadyCameraCoarseFilter(video_filepath, ocr_model, **steady_camera_coarse_parameters)
    camera_filter.process(steady_camera_coarse_parameters['poc_show_averaged_frames_pair'])
    steady_segments = camera_filter.calculate_steady_camera_ranges()

    # steady_segments = camera_filter.filter_segments_by_time(steady_segments, parameters['minimum_steady_camera_time_segment'])
    steady_segments.filter_by_time_duration(parameters['minimum_steady_camera_time_segment'])

    if steady_camera_coarse_parameters['poc_registration_verbose']:
        camera_filter.print_registration_results()
    if steady_camera_coarse_parameters['verbose_segments']:
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
    :raises FileNotFoundError: when video_filepath does not exist
    """
    if not os.path.exists(video_filepath):
        raise FileNotFoundError(f'File {video_filepath} does not exist')

    video_segments_writer = VideoSegmentsWriter(input_filepath=video_filepath,
                                                output_folder=output_folder,
                                                fps=video_segments.video_fps,
                                                scale_factor=parameters['scale_factor'])

    video_segments_writer.write(video_segments, filter_name='steady')

    if parameters['use_segments_gaps']:
        video_segments_complement = video_segments.segments_complement()
        video_segments_writer.write(video_segments_complement, filter_name='nonsteady')


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


def differentiate_steady_non_steady_to_subfolders(root_folder: str,
                                                  steady_entry: str,
                                                  non_steady_entry: str,
                                                  subfolder_steady: str,
                                                  subfolder_non_steady: str) -> None:
    """
    Description:
        Move cut steady and non-steady video segments to different folders. If filename has steady_entry, it will bw moved to subfolder_steady subfolder of
        the root_folder. If filename has non_steady_entry, it will bw moved to subfolder_non_steady subfolder of the root_folder
    :param root_folder: folder with source  steady anf non-steady video files;
    :param steady_entry: filename entry which that indicates video is (considered as) steady;
    :param non_steady_entry: filename entry which that indicates video is (considered as) non-steady;
    :param subfolder_steady: subfolder of the root folder to move steady videos to;
    :param subfolder_non_steady: subfolder of the root folder to move non-steady videos to;
    """
    source_filepaths = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if os.path.isfile(os.path.join(root_folder, f))]
    target_steady_folder = os.path.join(root_folder, subfolder_steady)
    target_non_steady_folder = os.path.join(root_folder, subfolder_non_steady)
    os.makedirs(target_steady_folder, exist_ok=True)
    os.makedirs(target_non_steady_folder, exist_ok=True)

    for source_filepath in source_filepaths:
        filename = os.path.basename(source_filepath)
        if non_steady_entry in filename:
            target_filepath = str(os.path.join(root_folder, subfolder_non_steady, filename))
            shutil.move(source_filepath, target_filepath)
        elif steady_entry in filename:
            target_filepath = str(os.path.join(root_folder, subfolder_steady, filename))
            shutil.move(source_filepath, target_filepath)
