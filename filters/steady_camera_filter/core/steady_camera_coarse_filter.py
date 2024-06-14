import cv2
import numpy as np
import easyocr

from beautifultable import BeautifulTable

from typing import Annotated, Literal
from numpy.typing import NDArray

from cv_utils.video_frames_batch import VideoFramesBatch
from filters.steady_camera_filter.core.image_registration_engine.image_registration_poc import ImageSequenceRegistrationPoc

image_grayscale = Annotated[NDArray[np.uint8], Literal["N", "M"]]
image_color = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
segments_list = Annotated[NDArray[np.int32], Literal["N", 2]]


class SteadyCameraCoarseFilter:
    def __init__(self,
                 video_filepath: str,
                 number_frames_to_average=20,
                 poc_maximum_dimension=512,
                 minimum_ocr_confidence=0.4,
                 maximum_shift_length=1.5,
                 poc_minimum_confidence=0.4):
        self.poc_maximum_image_dimension = poc_maximum_dimension
        self.video_frames_batch = VideoFramesBatch(video_filepath, number_frames_to_average)
        # OCR mask section
        self.ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
        self.model_ocr = easyocr.Reader(self.ocr_lang_list)
        self.ocr_confidence = minimum_ocr_confidence
        # Filter stuff section
        self.maximum_shift_length = maximum_shift_length
        self.steady_camera_frames_ranges: list[range] = []
        self.registration_shifts: np.ndarray = np.empty((0, 2))
        self.registration_confidence: np.ndarray = np.empty((0,))
        # Phase only correlation parameters
        self.poc_resolution = self._calculate_poc_resolution()
        self.poc_engine = ImageSequenceRegistrationPoc(self.poc_resolution)
        self.poc_minimum_confidence = poc_minimum_confidence

    def _calculate_poc_resolution(self) -> tuple[int, int]:
        poc_scale_factor = float(self.poc_maximum_image_dimension) / max(self.video_frames_batch.video_reader.resolution)
        original_resolution = self.video_frames_batch.video_reader.resolution
        poc_resolution = (round(original_resolution[0] * poc_scale_factor), round(original_resolution[1] * poc_scale_factor))
        return int(poc_resolution[0]), int(poc_resolution[1])

    def process(self, verbose=False):
        image_frames_batch = next(self.video_frames_batch)
        averaged_frames = np.mean(image_frames_batch, axis=(0, 3)).astype(np.uint8)
        averaged_frames = cv2.resize(averaged_frames, self.poc_resolution)
        # text_mask = self._text_mask(cv2.cvtColor(averaged_frames, cv2.COLOR_GRAY2RGB))
        # averaged_frames = self._apply_text_mask(averaged_frames, text_mask)
        self.poc_engine.update_deque(averaged_frames)

        if verbose:
            reference_image = averaged_frames

        for current_image_frames_batch in self.video_frames_batch:
            current_averaged_frames_rgb = np.mean(current_image_frames_batch, axis=0).astype(np.uint8)
            current_averaged_frames_grayscale = np.mean(current_averaged_frames_rgb, axis=2).astype(np.uint8)
            current_averaged_frames = cv2.resize(current_averaged_frames_grayscale, self.poc_resolution)
            current_text_mask = self._text_mask(current_averaged_frames_rgb, self.poc_resolution)
            current_averaged_frames = self._apply_text_mask(current_averaged_frames, current_text_mask)
            self.poc_engine.update_deque(current_averaged_frames)
            current_registration_result = self.poc_engine.register()
            self.registration_shifts = np.vstack((self.registration_shifts, np.array(current_registration_result.shift)))
            self.registration_confidence = np.append(self.registration_confidence, current_registration_result.peak_value)

            if verbose:
                target_image = current_averaged_frames
                registration_result = f'Shift: {current_registration_result.shift}, peak value: {current_registration_result.peak_value:1.2f}'
                reference_target_image = np.hstack((reference_image.astype(np.uint8), target_image.astype(np.uint8)))
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                text_origin = (50, 50)
                reference_target_image = cv2.putText(reference_target_image, registration_result, text_origin, text_font,
                                                     font_scale, (10, 10, 10), 5, cv2.LINE_AA)
                reference_target_image = cv2.putText(reference_target_image, registration_result, text_origin, text_font,
                                                     font_scale, (255, 100, 100), 1, cv2.LINE_AA)
                cv2.imshow('POC', reference_target_image)
                cv2.waitKey(10)
                reference_image = target_image

    def filter_segments_by_time(self, segments, time_threshold) -> segments_list:
        fps = self.video_frames_batch.video_reader.fps
        for segment_index, segment in enumerate(segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / fps) < time_threshold:
                segments[segment_index] = np.array([-1, -1])
        mask = segments[:, 0] >= 0
        segments = segments[mask]
        return segments

    def calculate_steady_camera_ranges(self) -> segments_list:
        # self.print_registration_results()
        shifts_norms = np.linalg.norm(self.registration_shifts, axis=1)
        shifts_norms_mask = shifts_norms < self.maximum_shift_length
        confidence_mask = self.registration_confidence > self.poc_minimum_confidence
        mask = np.logical_and(shifts_norms_mask, confidence_mask)

        number_frames_to_average = self.video_frames_batch.batch_size
        video_frames_number = self.video_frames_batch.video_reader.current_frame_index
        # video_frames_number = self.video_frames_batch.video_reader.n_frames
        number_bins = mask.shape[0]  # int(video_frames_number / number_frames_to_average)
        base_segment_range = np.array([0, 2 * number_frames_to_average - 1])
        segments_bins = np.linspace(0, (number_bins - 1) * number_frames_to_average, number_bins, dtype=np.int32).reshape(-1, 1) + base_segment_range
        segments_bins[-1, 1] = video_frames_number - 1
        segments_bins = segments_bins[mask]
        segments = self.unite_overlapping_ranges(segments_bins)

        return segments

    @staticmethod
    def unite_overlapping_ranges(segments: segments_list) -> segments_list:
        for index in range(len(segments) - 1):
            if segments[index, 1] > segments[index + 1, 0]:
                segments[index + 1, 0] = segments[index, 0]
                segments[index] = -1
        nans_mask = segments[:, 0] >= 0
        segments = segments[nans_mask]
        return segments

    def print_registration_results(self):
        table = BeautifulTable()
        table.columns.header = ['Shifts', 'Confidence']
        for index in range(self.registration_confidence.shape[0]):
            table.rows.append([self.registration_shifts[index], self.registration_confidence[index]])
        table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
        print(table)

    def _text_mask(self, image: image_color, output_resolution: tuple[int, int]) -> image_grayscale:
        current_ocr_result = self.model_ocr.readtext(image)  # , decoder='beamsearch', beamWidth=10)
        current_text_mask = np.zeros(image.shape[:2])
        if len(current_ocr_result):
            for ocr_box in current_ocr_result:
                if ocr_box[2] > self.ocr_confidence:
                    ocr_box = np.array(ocr_box[0]).reshape(-1, 2).astype(np.int32)
                    current_text_mask = cv2.fillPoly(current_text_mask, [ocr_box], color=(1, 1, 1))
        current_text_mask = cv2.resize(current_text_mask, output_resolution)
        return current_text_mask

    @staticmethod
    def _apply_text_mask(image: image_grayscale, mask: image_grayscale, sigma=7) -> image_grayscale:
        blurred_mask = cv2.GaussianBlur(mask, (sigma * 3, sigma * 3), sigma)
        blurred_mask = blurred_mask > 0.2
        blurred_mask = blurred_mask.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(blurred_mask, (sigma * 3, sigma * 3), sigma)
        return image * (1 - blurred_mask) + image.mean() * blurred_mask
