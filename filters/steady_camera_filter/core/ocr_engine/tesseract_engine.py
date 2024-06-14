from ocr_engine_base import OcrEngineBase
import pytesseract


class TesseractOcrEngine(OcrEngineBase):
    def pixel_mask(self, image, output_resolution):
        current_ocr_result = pytesseract.image_to_boxes(image)
        if current_ocr_result:
            print(current_ocr_result)
