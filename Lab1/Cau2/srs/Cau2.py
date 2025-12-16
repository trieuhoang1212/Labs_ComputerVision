import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# Đọc ảnh bằng Pillow
img = Image.open('../anh_mau.jpg')

img_array = np.array(img)

# Chuyển sang định dạng khác:
img_change = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

# Hiển thị ảnh
Image.fromarray(img_change)

# Lưu ảnh
cv2.imwrite('../anh_chuyen_doi.jpg', img_change)

img = Image.open('../anh_mau.jpg')
img_change = Image.open('../anh_chuyen_doi.jpg')

# Hiển thị 2 ảnh cạnh nhau
fig, axes = plt.subplots(1,2)
axes[0].imshow(img)
axes[0].set_title('Ảnh gốc')
axes[0].axis('off')

axes[1].imshow(img_change)
axes[1].set_title('Ảnh chuyển đổi màu sắc')
axes[1].axis('off')
plt.show()