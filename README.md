# Computer Vision Labs

Tổng hợp các bài thực hành về Thị giác máy tính (Computer Vision) sử dụng OpenCV, PIL và các thư viện xử lý ảnh.

## Cấu trúc Labs

### **Lab 1** - Cơ bản về xử lý ảnh

- Đọc, hiển thị và lưu ảnh với OpenCV và PIL
- Thay đổi kích thước và cắt ảnh
- Các thao tác cơ bản trên ảnh

### **Lab 2** - Biến đổi ảnh và lọc

- **Chương 1**: Điều chỉnh độ sáng, độ tương phản
- **Chương 2**: Các bộ lọc (Mean, Gaussian, làm sắc nét)
- **Chương 3**: Phát hiện biên với Sobel, Prewitt, Canny

### **Lab 3** - Phát hiện biên nâng cao

- **Lý thuyết**: Nghiên cứu chi tiết thuật toán Canny, so sánh Sobel, Laplacian
- **Chương 2**: Thực hành thuật toán Canny với các tham số (apertureSize, L2Gradient)

### **Lab 4** - Trích xuất đặc trưng và so khớp

- **Chương 1**: Giới thiệu về trích xuất đặc trưng
- **Chương 2**: Trích xuất đặc trưng Wavelet (Haar, Daubechies), so sánh độ tương đồng ảnh
- **Chương 3**: Bài toán nâng cao về feature matching
- **Chương 5**: Triển khai ứng dụng thực tế

### **Lab 5** - Nhận diện khuôn mặt

**FaceNet** và **MTCNN** cho nhận diện khuôn mặt

- Phát hiện khuôn mặt thời gian thực qua webcam
- Chụp và lưu ảnh khuôn mặt để huấn luyện (`capture_face.py`)
- Nhận diện khuôn mặt từ cơ sở dữ liệu (`detect_face.py`)
- Cập nhật và quản lý database khuôn mặt (`update_facenet.py`)thực qua webcam
- Cập nhật và quản lý cơ sở dữ liệu khuôn mặt

## Công nghệ sử dụng

- **OpenCV**: Xử lý ảnh và computer vision
- **PIL/Pillow**: Thao tác ảnh cơ bản
- **PyWavelets**: Biến đổi Wavelet
- **FaceNet, MTCNN**: Nhận diện khuôn mặt
- **PyTorch**: Deep learning cho face recognition
