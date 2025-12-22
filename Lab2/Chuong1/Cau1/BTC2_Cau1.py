import cv2
import numpy as np

img = cv2.imread(r"D:\XLH.a\BT_C2_c123\anh_mau.jpg")
if img is None:
    print("Không đọc được ảnh")
    exit()

brightness = 0   

while True:
    result = img.copy()
    h, w, c = img.shape

    for y in range(h):
        for x in range(w):
            for k in range(c):
                val = int(img[y, x, k]) + brightness
                if val > 255:
                    val = 255
                elif val < 0:
                    val = 0
                result[y, x, k] = val

    cv2.imshow("Dieu chinh do sang thu cong (+ / -)", result)

    key = cv2.waitKey(0)

    if key == 27:      # ESC
        break
    elif key == ord('+'):
        brightness += 10
    elif key == ord('-'):
        brightness -= 10

cv2.destroyAllWindows()
