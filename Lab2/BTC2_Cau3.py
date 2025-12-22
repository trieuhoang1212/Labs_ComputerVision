import cv2
import numpy as np

img = cv2.imread(r"D:\XLH.a\BT_C2_c123\anh_mau.jpg")  
if img is None:
    print("Không đọc được ảnh")
    exit()

negative_img = 255 - img

cv2.imshow("Anh goc", img)
cv2.imshow("Anh am ban", negative_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
