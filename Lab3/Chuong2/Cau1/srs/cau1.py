# sử dụng thuật toán cv2.canny để phát hiện cạnh trong hình ảnh
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('Lab3/assets/images/anh_mau1.jpg', cv2.IMREAD_GRAYSCALE)

# ngưỡng phổ biến hay sử dụng nhất và giúp ảnh rõ nét.
t_lower = 50;
t_upper = 150;

# Giải thích: img là ảnh đầu vào, t_lower và t_upper là hai ngưỡng để phát hiện cạnh
edges = cv2.Canny(img, t_lower, t_upper,apertureSize=3, L2gradient=True)
cv2.imwrite('Lab3/Chuong2/Cau1/input_cau1.jpg', edges)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc')
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Ảnh cạnh')
plt.show()