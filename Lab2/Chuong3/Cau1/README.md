## **Sobel**

- là phương pháp phổ biến hơn. Nó kết hợp giữa việc làm mịn ảnh (Gaussian smoothing) và tính đạo hàm.Đặc điểm: Do có tích hợp yếu tố làm mịn (trọng số ở giữa cao hơn), Sobel có khả năng chống nhiễu tốt hơn.Ma trận Kernel (3x3):Kernel Sobel gán trọng số cao hơn cho các điểm ảnh ở gần tâm của hướng đạo hàm.

Theo hướng ngang (X - phát hiện cạnh dọc):
$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$
Theo hướng dọc (Y - phát hiện cạnh ngang):
$$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$
**Cách dùng trong OpenCV:** img là ảnh đầu vào (grayscale)

**CV_64F:** Kiểu dữ liệu đầu ra (float) để giữ các giá trị âm (đạo hàm có thể âm)

```python
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # Đạo hàm theo x
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # Đạo hàm theo y 2.
```

## **Prewitt**

- là phiên bản đơn giản hơn của Sobel. Nó tính toán sự chênh lệch cường độ trung bình mà không nhấn mạnh vào các điểm ảnh ở trung tâm.Đặc điểm: Không thực hiện làm mịn như Sobel, nên Prewitt nhạy cảm với nhiễu hơn (dễ nhận diện sai nhiễu là cạnh). Tuy nhiên, tính toán đơn giản hơn (mặc dù với phần cứng hiện đại, sự khác biệt về tốc độ là không đáng kể).Ma trận Kernel (3x3):Các hệ số là 1, không có sự gia tăng trọng số ở giữa.

Theo hướng ngang (X):$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{bmatrix}$$

Theo hướng dọc (Y):
$$G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{bmatrix}$$

Cách dùng trong OpenCV:OpenCV không có hàm cv2.Prewitt trực tiếp.
Bạn phải tạo kernel thủ công và sử dụng hàm `cv2.filter2D().Pythonimport cv2`
và
`import numpy as np`

### Tạo kernel thủ công

```python
kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Áp dụng bộ lọc
prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
```
