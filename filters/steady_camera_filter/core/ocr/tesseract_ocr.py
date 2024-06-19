import pytesseract
import numpy as np
import cv2
from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase


class TesseractOcr(OcrBase):
    """
    Google Tesseract text recognition engine for text masking
    """
    def __init__(self, confidence=0.1):
        self.confidence = 100 * confidence

    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        number_boxes = len(d['level'])
        image_resolution = (image.shape[0], image.shape[1])
        mask = np.zeros(image_resolution, dtype=np.float32)
        for i in range(number_boxes):
            if d['conf'][i] > self.confidence:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
        mask = cv2.resize(mask, output_resolution)
        return mask
