import cv2
import numpy as np

# Đọc ảnh gốc
img = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)

# image1: ảnh gốc
cv2.imwrite("image1.jpg", img)

# image2: tăng sáng
image2 = cv2.convertScaleAbs(img, alpha=1.4, beta=40)
cv2.imwrite("image2.jpg", image2)

# image3: thêm nhiễu Gaussian
noise = np.random.normal(0, 90, img.shape)
image3 = np.clip(img + noise, 0, 255).astype(np.uint8)
cv2.imwrite("image3.jpg", image3)

# image4: blur mạnh + nhiễu mạnh
image4 = cv2.GaussianBlur(img, (101,101), 80)
image4 = cv2.GaussianBlur(img, (101,101), 80)
image4 = cv2.GaussianBlur(img, (101,101), 80)
noise = np.random.normal(0, 200, img.shape)
image4 = np.clip(image4 + noise, 0, 255).astype(np.uint8)
cv2.imwrite("image4.jpg", image4)

# image5: crop + resize
h, w = img.shape
crop = img[h//4:3*h//4, w//4:3*w//4]
image5 = cv2.resize(crop, (w, h))
cv2.imwrite("image5.jpg", image5)

#image6: rotate
M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
image6 = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("image6.jpg", image6)