from ocr_engine_base import OcrEngineBase
import easyocr


class EasyOcrEngine(OcrEngineBase):
    def __init__(self, minimum_ocr_confidence):
        self.ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
        self.model_ocr = easyocr.Reader(self.ocr_lang_list)
        self.ocr_confidence = minimum_ocr_confidence

    def pixel_mask(self, image, output_resolution):
        current_ocr_result = self.model_ocr.readtext(image)  # , decoder='beamsearch', beamWidth=10)
        # current_ocr_result = pytesseract.image_to_boxes(image)
        # if current_ocr_result:
        #     print(current_ocr_result)
        current_text_mask = np.zeros(image.shape[:2])
        if len(current_ocr_result):
            for ocr_box in current_ocr_result:
                if ocr_box[2] > self.ocr_confidence:
                    ocr_box = np.array(ocr_box[0]).reshape(-1, 2).astype(np.int32)
                    current_text_mask = cv2.fillPoly(current_text_mask, [ocr_box], color=(1, 1, 1))
        current_text_mask = cv2.resize(current_text_mask, output_resolution)
        return current_text_mask
