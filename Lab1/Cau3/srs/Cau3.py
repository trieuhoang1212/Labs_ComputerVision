# Cắt và thay đổi kích thước ảnh sử dụng PIL
from PIL import Image
import os

script_dir = os.path.dirname(__file__)
img_path = os.path.join(script_dir, '..', 'input.jpg')

img = Image.open(img_path)
resized_img = img.resize((600, 500))

resized_img.save('resized_image_PIL.jpg')

resized_img.show()