import os.path

import cv2
import numpy as np

from collections import deque
from beautifultable import BeautifulTable
from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase
from filters.steady_camera_filter.core.ocr.easy_ocr import EasyOcr

from typing import Annotated, Literal
from numpy.typing import NDArray


from filters.steady_camera_filter.core.video_segments import VideoSegments
from cv_utils.video_frames_batch import VideoFramesBatch
from filters.steady_camera_filter.core.image_registration.image_registration_poc import ImageSequenceRegistrationPoc

image_grayscale = Annotated[NDArray[np.uint8], Literal["N", "M"]]
image_color = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class SteadyCameraCoarseFilter:
    def __init__(self,
                 video_filepath: str,
                 ocr: OcrBase,
                 number_frames_to_average=20,
                 poc_maximum_image_dimension=512,
                 maximum_shift_length=1.5,
                 poc_minimum_confidence=0.4,
                 ):
        self.video_frames_batch = VideoFramesBatch(video_filepath, number_frames_to_average)
        self.ocr = ocr
        # Image registration section
        self.maximum_shift_length = maximum_shift_length
        self.steady_camera_frames_ranges: list[range] = []
        self.registration_shifts: np.ndarray = np.empty((0, 2))
        self.registration_confidence: np.ndarray = np.empty((0,))
        self.averaged_frames_pair_deque = deque(maxlen=2)
        # Phase only correlation section
        self.poc_resolution: tuple[int, int]
        self.poc_maximum_image_dimension = poc_maximum_image_dimension
        self._calculate_poc_resolution()
        self.poc_engine = ImageSequenceRegistrationPoc(self.poc_resolution)
        self.poc_minimum_confidence = poc_minimum_confidence

    def _calculate_poc_resolution(self) -> None:
        poc_scale_factor = float(self.poc_maximum_image_dimension) / max(self.video_frames_batch.video_reader.resolution)
        original_resolution = self.video_frames_batch.video_reader.resolution
        poc_resolution = (round(original_resolution[0] * poc_scale_factor), round(original_resolution[1] * poc_scale_factor))
        self.poc_resolution = int(poc_resolution[0]), int(poc_resolution[1])

    def process(self, verbose=False) -> None:
        """
        Description:
            Register images sequence using phase only correlation. Before performing registration all text is softly masked,
            cause POC is sensitive to high gradients regions.
        Remarks:
            Sometimes number of frames, given by cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT) is not precise.
            So, accumulated VideoFramesBatch.VideoReader.current_frame_index used as a video frames number (after all frames had been read).

        @verbose: show reference and target image pair with registration results (for debug purposes)
        """

        for current_image_frames_batch in self.video_frames_batch:
            current_averaged_frames = np.mean(current_image_frames_batch, axis=(0, 3)).astype(np.uint8)
            # print(f'Mean of empty slice, Filename {self.video_frames_batch.video_filepath}, {current_averaged_frames.shape=}')

            current_text_mask = self.ocr.pixel_mask(current_averaged_frames, self.poc_resolution)
            current_averaged_frames = cv2.resize(current_averaged_frames, self.poc_resolution)
            current_averaged_frames = self._apply_text_mask(current_averaged_frames, current_text_mask)
            self.poc_engine.update_deque(current_averaged_frames)

            if verbose:
                self.averaged_frames_pair_deque.append(current_averaged_frames)

            if len(self.poc_engine.fft_deque) == 2:
                current_registration_result = self.poc_engine.register()
                self.registration_shifts = np.vstack((self.registration_shifts, np.array(current_registration_result.shift)))
                self.registration_confidence = np.append(self.registration_confidence, current_registration_result.peak_value)

                if verbose:
                    registration_result = f'Shift: {current_registration_result.shift}, peak value: {current_registration_result.peak_value:1.2f}'
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

    def filter_segments_by_time(self, video_segments: VideoSegments, time_threshold: float) -> VideoSegments:
        """
        Description:
            Filter video segments by duration. If segment duration is less than time_threshold, it will be deleted.
        @video_segments: input video segments
        @time_threshold: time threshold in seconds
        @return filtered by time video segments
        """
        fps = self.video_frames_batch.video_reader.fps
        for segment_index, segment in enumerate(video_segments.segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / fps) < time_threshold:
                video_segments.segments[segment_index] = np.array([-1, -1])
        mask = video_segments.segments[:, 0] >= 0
        video_segments.segments = video_segments.segments[mask]
        return video_segments

    def calculate_steady_camera_ranges(self) -> VideoSegments:
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
        @segments: input segments
        @return: unified segments
        """
        for index in range(len(segments) - 1):
            if segments[index, 1] > segments[index + 1, 0]:
                segments[index + 1, 0] = segments[index, 0]
                segments[index] = -1
        nans_mask = segments[:, 0] >= 0
        segments = segments[nans_mask]
        return segments

    def print_registration_results(self) -> None:
        """
        Description:
            Print registration results for all images in averaged images batch
        @return: None
        """
        table = BeautifulTable()
        table.columns.header = ['Shifts', 'Confidence']
        for index in range(self.registration_confidence.shape[0]):
            table.rows.append([self.registration_shifts[index], self.registration_confidence[index]])
        table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
        print(table)

    @staticmethod
    def _apply_text_mask(image: image_grayscale, mask: image_grayscale, sigma=7) -> image_grayscale:
        """
        Apply mask to the image. Given a mask it will be:
            1. Extended to cover more space
            2. Blurred using Gaussian blur with the given sigma
        After that, grayscale image and monochrome image (whose color is the mean color of the whole image with the same shape)
        will be mixed up by calculated mask.
        @image: input grayscale image
        @mask: mask with float values in [0, 1] range
        @return: masked image
        """
        blurred_mask = cv2.GaussianBlur(mask, (sigma * 3, sigma * 3), sigma)
        blurred_mask = blurred_mask > 0.1
        blurred_mask = blurred_mask.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(blurred_mask, (sigma * 3, sigma * 3), sigma)
        return image * (1 - blurred_mask) + image.mean() * blurred_mask
