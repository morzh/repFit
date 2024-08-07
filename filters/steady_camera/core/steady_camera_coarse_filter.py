import os.path
from collections import deque

import cv2
import numpy as np
from beautifultable import BeautifulTable
from loguru import logger
from typing import Annotated, Literal
from numpy.typing import NDArray

from utils.cv.video_reader_frames_batch import VideoReaderFramesBatch
from filters.steady_camera.core.image_registration.image_registration_poc import ImageSequenceRegistrationPoc
from filters.steady_camera.core.persons_mask.persons_mask_base import PersonsMaskBase
from filters.steady_camera.core.ocr.ocr_base import OcrBase
from filters.steady_camera.core.video_segments import VideoSegments

image_grayscale = Annotated[NDArray[np.uint8], Literal["N", "M"]]
image_color = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class SteadyCameraCoarseFilter:
    """
    Description:
        Steady camera filter. This filter extracts steady camera segments in video. Coarse in the name of the filter means, it uses averaged frames
        to speedup video processing. The downside of the averaging is the filter mey not be that precise.
    Remarks:
        As this approach uses frames averaging, at the begging and at the end of a video segment camera could be slightly non-steady.
        Also, when camera angle changes fast enough (in the period of one or several frames), this algorithm may not catch it.
        There also could be some issues  with sidebars videos (videos originally vertical with added sidebars to produce horizontal ones).
        Text regions smooth masking is mandatory to produce correct results.
    """
    def __init__(self, video_filepath: str, ocr_detector: OcrBase = None, persons_detector: PersonsMaskBase = None, **kwargs):
        """
        :param video_filepath: video file pathname;
        :param ocr_detector: ocr model to use for text masking;
        :param kwargs: see below.
        :keyword number_frames_to_average: -- number of frames to average before registration
        :keyword maximum_shift_length: pixel shift length threshold. If norm(pixel_shift) < maximum_shift_length, camera considered as steady
        between respective frames.
        :keyword registration_minimum_confidence: registration confidence threshold. If registration confidence less than
        registration_minimum_confidence, then camera is not considered as steady.
        """
        self.video_frames_batch = VideoReaderFramesBatch(video_filepath, kwargs.get('number_frames_to_average', 20))
        self.ocr_detector = ocr_detector
        self.persons_detector = persons_detector
        # Image registration section
        self.maximum_shift_length = kwargs.get('maximum_shift_length', 2.0)
        # self.steady_camera_frames_ranges: list = []
        self.registration_shifts: np.ndarray = np.empty((0, 2))
        self.registration_confidence: np.ndarray = np.empty((0,))
        self.averaged_frames_pair_deque = deque(maxlen=2)
        # Phase only correlation section
        self.poc_resolution: tuple[int, int]
        self.poc_maximum_image_dimension = kwargs.get('poc_maximum_dimension', 1024)
        self._calculate_poc_resolution()
        self.poc_engine = ImageSequenceRegistrationPoc(self.poc_resolution)
        self.poc_minimum_confidence = kwargs.get('poc_minimum_confidence', 0.2)

    def _calculate_poc_resolution(self) -> None:
        """
        Description:
            Calculates factor for image resolution before using images registration. For computational reasons, all images fed to
            image registration procedure will have the same maximum resolution along some dimension.
        """
        poc_scale_factor = self.poc_maximum_image_dimension / max(self.video_frames_batch.video_reader.resolution)
        original_resolution = self.video_frames_batch.video_reader.resolution
        poc_resolution = (int(original_resolution[0] * poc_scale_factor + 0.5), int(original_resolution[1] * poc_scale_factor + 0.5))
        self.poc_resolution = poc_resolution[0], poc_resolution[1]

    def process(self, verbose=False) -> None:
        """
        Description:
            Register images sequence using phase only correlation. Before performing registration all text regions are masked
            by a soft blurred masks, cause POC is sensitive to a high gradients regions.
        Remarks:
            Sometimes number of frames, given by cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT) is not precise.
            So, accumulated VideoFramesBatch.VideoReader.current_frame_index used as a video frames number (after all frames had been read).

        :param verbose: show reference and target image pair with registration results (for debug purposes)
        """

        for current_image_frames_batch in self.video_frames_batch:
            current_averaged_frames = np.mean(current_image_frames_batch, axis=0).astype(np.uint8)  # averaging images across batch dimension
            # logger.info(f'Read batch of frames with shape {current_averaged_frames.shape}')

            if self.ocr_detector is not None:
                current_text_mask = self.ocr_detector.pixel_mask(current_averaged_frames, self.poc_resolution)
                current_averaged_frames = cv2.resize(current_averaged_frames, self.poc_resolution)
                current_averaged_frames = self._apply_mask(current_averaged_frames, current_text_mask)
                # logger.info(f'Applied text mask with shape {current_text_mask.shape} to averaged frame with shape {current_averaged_frames.shape}')
            if self.persons_detector is not None:
                current_persons_mask = self.persons_detector.pixel_mask(current_averaged_frames, self.poc_resolution)
                # logger.info(f'Current persons mask shape is {current_persons_mask.shape}')
                current_averaged_frames = self._apply_mask(current_averaged_frames, current_persons_mask)
                # logger.info(f'Applied persons mask with shape {current_persons_mask.shape} to averaged frame with shape {current_averaged_frames.shape}')

            current_averaged_frames = np.mean(current_averaged_frames, axis=2).astype(np.uint8)  # making grayscale image from color one
            self.poc_engine.update_deque(current_averaged_frames)
            self.averaged_frames_pair_deque.append(current_averaged_frames)

            if len(self.poc_engine.fft_deque) == 2:
                current_registration_result = self.poc_engine.register()
                self.registration_shifts = np.vstack((self.registration_shifts, np.array(current_registration_result.shift)))
                self.registration_confidence = np.append(self.registration_confidence, current_registration_result.confidence)

                if verbose:
                    # logger.info(f'Registration value: {current_registration_result.shift}, confidence: {current_registration_result.confidence:1.2f}.')
                    registration_result = f'Shift: {current_registration_result.shift}, peak value: {current_registration_result.confidence:1.2f}'
                    reference_target_image = np.hstack((self.averaged_frames_pair_deque[0].astype(np.uint8),
                                                        self.averaged_frames_pair_deque[1].astype(np.uint8)))
                    text_font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    text_origin = (50, 50)
                    reference_target_image = cv2.putText(reference_target_image, registration_result, text_origin, text_font,
                                                         font_scale, (10, 10, 10), 5, cv2.LINE_AA)
                    reference_target_image = cv2.putText(reference_target_image, registration_result, text_origin, text_font,
                                                         font_scale, (255, 100, 100), 1, cv2.LINE_AA)
                    cv2.imshow('POC', reference_target_image)
                    cv2.waitKey(10)

    def steady_camera_video_segments(self) -> VideoSegments:
        """
        Description:
            Calculate video segments in frames at which camera is steady
        :return: video segments
        """
        # self.print_registration_results()
        shifts_norms = np.linalg.norm(self.registration_shifts, axis=1)
        shifts_norms_mask = shifts_norms < self.maximum_shift_length
        confidence_mask = self.registration_confidence > self.poc_minimum_confidence
        mask = np.logical_and(shifts_norms_mask, confidence_mask)

        number_frames_to_average = self.video_frames_batch.batch_size
        video_frames_number = self.video_frames_batch.video_reader.current_frame_index
        number_bins = mask.shape[0]
        base_segment_range = np.array([0, 2 * number_frames_to_average - 1])
        segments_bins = np.linspace(0, (number_bins - 1) * number_frames_to_average, number_bins, dtype=np.int32).reshape(-1, 1) + base_segment_range
        segments_bins[-1, 1] = video_frames_number - 1
        segments_bins = segments_bins[mask]
        segments = self.unite_overlapping_ranges(segments_bins)

        video_filename = os.path.basename(self.video_frames_batch.video_filepath)
        video_segments = VideoSegments(video_filename=video_filename,
                                       video_width=self.video_frames_batch.video_reader.width,
                                       video_height=self.video_frames_batch.video_reader.height,
                                       frames_number=self.video_frames_batch.video_reader.current_frame_index,
                                       video_fps=self.video_frames_batch.video_reader.fps,
                                       segments=segments)
        return video_segments

    @staticmethod
    def unite_overlapping_ranges(segments: segments_list) -> segments_list:
        """
        Description:
            If two or more segments overlaps, this function unties them into one.
            Resulted number of segments will be less or equal to input number of segments.
        :param segments: input segments
        :return: unified segments
        """
        for index in range(len(segments) - 1):
            if segments[index, 1] > segments[index + 1, 0]:
                segments[index + 1, 0] = segments[index, 0]
                segments[index] = -1
        mask = segments[:, 0] >= 0
        segments = segments[mask]
        return segments

    def log_registration_results(self) -> None:
        """
        Description:
            Print registration results for all images in averaged images batch
        """
        table = BeautifulTable()
        table.columns.header = ['Shifts', 'Confidence']
        for index in range(self.registration_confidence.shape[0]):
            table.rows.append([self.registration_shifts[index], self.registration_confidence[index]])
        table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
        logger.info(table)

    def save_registration_results(self, filepath: str) -> None:
        pass

    @staticmethod
    def _apply_mask(image: image_color | image_grayscale, mask: image_grayscale, sigma=7, mask_extend_threshold=0.1) -> image_color | image_grayscale:
        """
        Description:
            Apply mask to the image. Given a mask it will be:
                #. Extended to cover more space
                #. Blurred using Gaussian blur with the given sigma
            After that, grayscale image and monochrome image (whose color is the mean color of the whole image with the same shape)
            will be mixed up by calculated mask.
        :param image: input grayscale image
        :param mask: mask with float values in [0, 1] range
        :return: masked image
        """
        blurred_mask = cv2.GaussianBlur(mask, (sigma * 3, sigma * 3), sigma)
        blurred_mask = blurred_mask > mask_extend_threshold
        blurred_mask = blurred_mask.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(blurred_mask, (sigma * 3, sigma * 3), sigma)

        image_channels = image.shape[2]
        if image_channels == 1:
            image_masked = image * (1 - blurred_mask) + image.mean() * blurred_mask
        elif image_channels == 3:
            blurred_mask = np.expand_dims(blurred_mask, axis=2)
            blurred_mask = np.repeat(blurred_mask, image_channels, axis=2)
            image_masked = image * (1 - blurred_mask) + image.mean(axis=(0, 1)) * blurred_mask

        return image_masked
