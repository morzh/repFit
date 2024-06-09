import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

from typing import Annotated, Literal, TypeVar, Optional
from numpy.typing import NDArray

from cv_utils.video_frames_batch import VideoFramesBatch
from filters.steady_camera_filter.core.image_registration_poc import ImageSequenceRegistrationPoc

ndimageNxMx3 = Annotated[NDArray[np.uint8], Literal["N", "M", 3]]
ndimageNxM = Annotated[NDArray[np.uint8], Literal["N", "M"]]


class SteadyCameraCoarseFilter:
    def __init__(self, video_filepath: str, number_frames_to_average=20, poc_maximum_dimension=512, ocr_confidence=0.4):
        # self.number_frames_to_average = number_frames_to_average
        self.poc_maximum_image_dimension = poc_maximum_dimension
        self.video_frames_batch = VideoFramesBatch(video_filepath, number_frames_to_average)
        self.video_frames_batch.batch_size = number_frames_to_average
        # OCR mask section
        self.ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
        self.model_ocr = easyocr.Reader(self.ocr_lang_list)
        self.ocr_confidence = ocr_confidence
        # Filter stuff section
        self.steady_camera_frames_ranges: list[range] = []
        self.registration_results: list = []
        self.poc_resolution = self._calculate_poc_resolution()
        self.poc = ImageSequenceRegistrationPoc(self.poc_resolution)

    def _calculate_poc_resolution(self) -> tuple[int, int]:
        poc_scale_factor = float(self.poc_maximum_image_dimension) / max(self.video_frames_batch.video_reader.resolution)
        original_resolution = self.video_frames_batch.video_reader.resolution
        poc_resolution = (round(original_resolution[0] * poc_scale_factor), round(original_resolution[1] * poc_scale_factor))
        return int(poc_resolution[0]), int(poc_resolution[1])

    def process(self):
        image_frames_batch = next(self.video_frames_batch)
        averaged_frames = np.mean(image_frames_batch, axis=(0, 3)).astype(np.uint8)
        averaged_frames = cv2.resize(averaged_frames, self.poc_resolution)
        text_mask = self._text_mask(cv2.cvtColor(averaged_frames, cv2.COLOR_GRAY2RGB))
        averaged_frames = self._apply_text_mask(averaged_frames, text_mask)
        self.poc.update_deque(averaged_frames)

        if __debug__:
            reference_image = averaged_frames

        while True:
            current_image_frames_batch = next(self.video_frames_batch)
            current_averaged_frames = np.mean(current_image_frames_batch, axis=(0, 3)).astype(np.uint8)
            current_averaged_frames = cv2.resize(current_averaged_frames, self.poc_resolution)
            current_text_mask = self._text_mask(cv2.cvtColor(current_averaged_frames, cv2.COLOR_GRAY2RGB))
            current_averaged_frames = self._apply_text_mask(current_averaged_frames, current_text_mask)
            self.poc.update_deque(current_averaged_frames)
            current_registration_result = self.poc.register()
            self.registration_results.append(current_registration_result)

            if __debug__:
                target_image = current_averaged_frames
                plt.figure(figsize=(25, 15))
                plt.suptitle(str(current_registration_result.shift) + ' ' + str(current_registration_result.peak_value))
                plt.subplot(121)
                plt.imshow(reference_image, cmap='gray')
                plt.subplot(122)
                plt.imshow(target_image, cmap='gray')
                plt.tight_layout()
                plt.show()
                reference_image = target_image

    def _text_mask(self, image: ndimageNxMx3) -> ndimageNxM:
        current_ocr_result = self.model_ocr.readtext(image)
        current_text_mask = np.zeros(image.shape[:2])
        if len(current_ocr_result):
            for ocr_box in current_ocr_result:
                if ocr_box[2] > self.ocr_confidence:
                    ocr_box = np.array(ocr_box[0]).reshape(-1, 2).astype(np.int32)
                    current_text_mask = cv2.fillPoly(current_text_mask, [ocr_box], color=(1, 1, 1))
        return current_text_mask

    @staticmethod
    def _apply_text_mask(image: ndimageNxM, mask: ndimageNxM, sigma=11) -> ndimageNxM:
        blurred_mask = cv2.GaussianBlur(mask, (sigma * 3, sigma * 3), sigma)
        return image * (1 - blurred_mask) + image.mean() * blurred_mask
