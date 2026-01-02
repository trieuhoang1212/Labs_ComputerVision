import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Cố định đường dẫn đến thư mục assets/images
script_dir = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(script_dir, '..', 'assets', 'images') 
IMG_PATH = os.path.abspath(IMG_PATH)

# Tạo thư mục nếu chưa tồn tại assets/images/<username>
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
    print(f"Created directory: {IMG_PATH}")

# Số lượng ảnh chụp cho người dùng là 20
count = 20
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)

# Tạo thư mục nếu chưa tồn tại assets/images/<username>
if not os.path.exists(USR_PATH):
    os.makedirs(USR_PATH)
    print(f"Created directory: {USR_PATH}")

leap = 1

# Khởi tạo MTCNN và mở camera
mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

print(f"Capturing {count} images. Press ESC to stop.")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if mtcnn(frame) is not None and leap%2:
        path = os.path.join(USR_PATH, f'{timestamp}_img{21-count:02d}.jpg')
        face_img = mtcnn(frame, save_path = path)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
