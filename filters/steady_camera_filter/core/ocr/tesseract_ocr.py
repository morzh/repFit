import pytesseract
import numpy as np
import cv2
from filters.steady_camera_filter.core.ocr.ocr_base import OcrBase


class TesseractOcr(OcrBase):
    """
    Google Tesseract text recognition engine for text masking
    """
    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])
        image_resolution = (image.shape[0], image.shape[1])
        mask = np.zeros(image_resolution)
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 1, 2)
        return image
