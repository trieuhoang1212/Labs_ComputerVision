import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

# Đường thẳng màu xanh dương với độ dày 5 pixel
cv.line(img, (384,330), (128,300), (255, 0, 0), 5)

# Hình chữ nhật màu xanh lá cây với độ dày 3 pixel
cv.rectangle(img, (10,20), (300,200), (0,255,0), 3)

# Hình tròn màu đỏ đặc với bán kính 63 pixel
cv.circle(img, (400,200), 20, (0,0,255), 3)

cv.imwrite('Hinhve.png', img)
cv.imshow('Hinh ve', img)

cv.waitKey(0)
cv.destroyAllWindows()
