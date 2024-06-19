import numpy as np
import cv2
import easyocr

from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase


class EasyOcr(OcrBase):
    """
    Jaided EasyOCR for text recognition in images.
    """
    def __init__(self, confidence_threshold, minimal_resolution: int = 512):
        """
        @confidence_threshold: minimum confidence for detected text
        @minimal resolution: if one of image's dimension less than minimal_resolution, it will be increased by a 1.5 factor.
        """
        self.ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
        self.model_ocr = easyocr.Reader(self.ocr_lang_list)
        self.ocr_confidence = confidence_threshold
        self.minimum_resolution = minimal_resolution

    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        if max(image.shape) < self.minimum_resolution:
            image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        current_ocr_result = self.model_ocr.readtext(image)
        current_text_mask = np.zeros(image.shape[:2])
        if len(current_ocr_result):
            for ocr_box in current_ocr_result:
                if ocr_box[2] > self.ocr_confidence:
                    ocr_box = np.array(ocr_box[0]).reshape(-1, 2).astype(np.int32)
                    current_text_mask = cv2.fillPoly(current_text_mask, [ocr_box], color=(1, 1, 1))
        current_text_mask = cv2.resize(current_text_mask, output_resolution)
        return current_text_mask
