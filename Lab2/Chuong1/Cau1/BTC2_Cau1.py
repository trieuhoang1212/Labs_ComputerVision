import cv2
import numpy as np

img = cv2.imread(r"Chuong1/Cau1/anh_mau.jpg")  

if img is None:
    print("Không đọc được ảnh!")
    exit()

value = 50  

bright_img = cv2.add(img, np.ones(img.shape, dtype="uint8") * value)

dark_img = cv2.subtract(img, np.ones(img.shape, dtype="uint8") * value)

cv2.imshow("Anh goc", img)
cv2.imshow("Tang do sang", bright_img)
cv2.imshow("Giam do sang", dark_img)
cv2.waitKey(0)
cv2.destroyAllWindows()