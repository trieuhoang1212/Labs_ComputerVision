import cv2
from facenet_pytorch import MTCNN
import torch

# --- CONSTANTS ---
DETECTION_THRESHOLDS = [0.7, 0.7, 0.8]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
BOX_COLOR = (0, 255, 0)  # Green
BOX_THICKNESS = 2

# --- DEVICE SETUP ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- INITIALIZATION ---
mtcnn = MTCNN(thresholds=DETECTION_THRESHOLDS, keep_all=True, device=device)
cap = cv2.VideoCapture(0)

# Set video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("\nBắt đầu phát hiện khuôn mặt. Nhấn ESC để thoát.\n")

# Read video frame-by-frame
while cap.isOpened():
    isSuccess, frame = cap.read()
    if not isSuccess:
        print("Lỗi đọc frame")
        break
    
    # Phát hiện tất cả khuôn mặt trong frame
    boxes, probs = mtcnn.detect(frame)
    face_count = 0
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            # Chuyển bounding box coordinates sang integer
            bbox = list(map(int, box.tolist()))
            # MTCNN trả về probability (độ chắc chắn là khuôn mặt)
            confidence = probs[i] if probs is not None else 0.0
            
            # VẼ BOUNDING BOX xung quanh khuôn mặt
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         BOX_COLOR, BOX_THICKNESS)
            
            # HIỂN THỊ CONFIDENCE SCORE
            # Score càng cao = MTCNN càng chắc đây là khuôn mặt
            label = f"{confidence*100:.1f}%"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)
            face_count += 1
    
    # Hiển thị số lượng faces
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Đã đóng camera")

