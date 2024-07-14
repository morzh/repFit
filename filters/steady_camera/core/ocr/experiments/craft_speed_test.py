import time

from PIL import Image
from CRAFT import CRAFTModel, draw_polygons

"""
https://github.com/boomb0om/CRAFT-text-detection
"""

craft_weights_folder = '/home/anton/work/fitMate/repFit/3rd_party/weights/craft'
test_image = '/home/anton/work/fitMate/repFit/filters/steady_camera/experiments/cafe_sign.jpg'

model = CRAFTModel(craft_weights_folder, 'cuda', use_refiner=False, fp16=False)
img = Image.open(test_image)
start_time = time.time()
polygons = model.get_polygons(img)
end_time = time.time()
result = draw_polygons(img, polygons)
# result.show()
print(f'processing toolk {end_time - start_time} seconds')
