from ocr_base import OcrBase
import pytesseract


class TesseractOcr(OcrBase):
    def pixel_mask(self, image, output_resolution):
        current_ocr_result = pytesseract.image_to_boxes(image)
        if current_ocr_result:
            print(current_ocr_result)
