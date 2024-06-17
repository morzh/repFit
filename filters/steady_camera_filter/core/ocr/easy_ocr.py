import numpy as np
import cv2
from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase
import easyocr


class EasyOcr(OcrBase):
    def __init__(self, minimum_ocr_confidence, minimal_resolution: int = 512):
        self.ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
        self.model_ocr = easyocr.Reader(self.ocr_lang_list)
        self.ocr_confidence = minimum_ocr_confidence
        self.minimum_resolution = minimal_resolution

    def pixel_mask(self, image, output_resolution) -> cv2.typing.MatLike:
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
