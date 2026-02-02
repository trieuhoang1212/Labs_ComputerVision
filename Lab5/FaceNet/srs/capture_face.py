import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

# --- CONSTANTS ---
NUM_IMAGES_TO_CAPTURE = 20
FRAME_SKIP = 2  # Chụp mỗi 2 frames
MTCNN_MARGIN = 20
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- DEVICE SETUP ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.abspath(os.path.join(script_dir, '..', 'assets', 'images'))

# Tạo thư mục assets/images nếu chưa tồn tại
os.makedirs(IMG_PATH, exist_ok=True)

# --- USER INPUT ---
usr_name = input("Nhập tên của bạn: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)

# Tạo thư mục cho user
os.makedirs(USR_PATH, exist_ok=True)
print(f"Thư mục lưu ảnh: {USR_PATH}")

# --- INITIALIZATION ---
mtcnn = MTCNN(margin=MTCNN_MARGIN, keep_all=False, post_process=False, device=device)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("✗ Không thể mở camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print(f"\n{'='*50}")
print(f"Bắt đầu chụp {NUM_IMAGES_TO_CAPTURE} ảnh cho: {usr_name}")
print(f"Nhấn ESC để dừng")
print(f"{'='*50}\n")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
count = NUM_IMAGES_TO_CAPTURE
frame_counter = 0

while cap.isOpened() and count > 0:
    isSuccess, frame = cap.read()
    if not isSuccess:
        print("✗ Lỗi đọc frame")
        break
    
    frame_counter += 1
    
    # Tránh chụp ảnh quá giống nhau 
    # Chỉ chụp mỗi FRAME_SKIP frames để có đa dạng góc độ/biểu cảm
    if frame_counter % FRAME_SKIP == 0:
        # MTCNN detect khuôn mặt
        face_detected = mtcnn(frame)
        if face_detected is not None:
            # === LƯU ẢNH: MTCNN tự động crop và align khuôn mặt ===
            img_filename = f"{usr_name}_{timestamp}_{NUM_IMAGES_TO_CAPTURE - count + 1:02d}.jpg"
            save_path = os.path.join(USR_PATH, img_filename)
            mtcnn(frame, save_path=save_path)  # Save cropped face
            count -= 1
            print(f"✓ Đã chụp: {NUM_IMAGES_TO_CAPTURE - count}/{NUM_IMAGES_TO_CAPTURE}")
    
    # Hiển thị progress trên frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Captured: {NUM_IMAGES_TO_CAPTURE - count}/{NUM_IMAGES_TO_CAPTURE}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Face Capturing', display_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        print("\n⚠ Đã hủy bởi người dùng")
        break

cap.release()
cv2.destroyAllWindows()

if count == 0:
    print(f"\n✓ Hoàn thành! Đã chụp {NUM_IMAGES_TO_CAPTURE} ảnh")
else:
    print(f"\n⚠ Chỉ chụp được {NUM_IMAGES_TO_CAPTURE - count} ảnh")
