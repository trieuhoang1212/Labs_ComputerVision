# cắt và thay dổi kích thước ảnh sử dụng OpenCV
import cv2

img = cv2.imread('../input.jpg')

# thay đổi kích thước
(h, w, d) = img.shape

# tính tỷ lệ mới và kích thước mới
r = 300.0 / w 
dim = (300, int(h * r))

resized = cv2.resize(img, dim)
cv2.imwrite('resized_image.jpg', resized)

cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()