import easyocr
import cv2
import time

test_image = '/home/anton/work/fitMate/repFit/filters/steady_camera_filter/experiments/cafe_sign.jpg'

languages_1 = ['ch_tra', 'en']
ocr_lang_list = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
reader = easyocr.Reader(ocr_lang_list, quantize=False)
img = cv2.imread(test_image)
start_time = time.time()
result = reader.readtext(img)
end_time = time.time()
print(f'processing toolk {end_time - start_time} seconds')