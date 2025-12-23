import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread(r"C:\Users\PC\Downloads\images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kernel làm sắc nét
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Áp dụng lọc
sharpen = cv2.filter2D(gray, -1, kernel)

cv2.imshow("Original", gray)
cv2.imshow("Sharpen", sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()
