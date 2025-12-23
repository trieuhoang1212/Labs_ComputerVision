import cv2
import numpy as np

img = cv2.imread(r"Chuong1\Cau2\anh_mau.jpg") 
if img is None:
    print("Không đọc được ảnh")
    exit()

alpha = 0.1 

contrast_img = img.astype(np.float32) * alpha

contrast_img = np.clip(contrast_img, 0, 255).astype(np.uint8)

cv2.imshow("Anh goc", img)
cv2.imshow("Anh sau khi thay doi do tuong phan", contrast_img)
cv2.waitKey(0)
cv2.destroyAllWindows()