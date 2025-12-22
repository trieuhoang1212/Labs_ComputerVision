import cv2
import numpy as np

img = cv2.imread(r"D:\XLH.a\BT_C2_c123\anh_mau.jpg")  
if img is None:
    print("Không đọc được ảnh")
    exit()

img = cv2.resize(img, (600, 400))

def update_contrast(val):
    alpha = val / 50.0  
    if alpha == 0:
        alpha = 0.01

    h, w, c = img.shape
    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                v = int(img[i, j, k] * alpha)
                if v > 255:
                    v = 255
                elif v < 0:
                    v = 0
                result[i, j, k] = v

    cv2.imshow("Dieu chinh do tuong phan", result)

cv2.namedWindow("Dieu chinh do tuong phan")
cv2.createTrackbar("Tuong phan", "Dieu chinh do tuong phan", 50, 150, update_contrast)

cv2.imshow("Anh goc", img)
update_contrast(50)
cv2.waitKey(0)
cv2.destroyAllWindows()
