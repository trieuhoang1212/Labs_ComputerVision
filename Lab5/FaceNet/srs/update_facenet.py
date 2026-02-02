import glob
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from datetime import datetime
from collections import deque

#  CONFIGURATION CONSTANTS

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(script_dir, '..', 'assets', 'images')
DATA_PATH = os.path.join(script_dir, '..', 'data')
EMBEDDINGS_FILE = os.path.join(DATA_PATH, 'facelist_facenet.pth')
USERNAMES_FILE = os.path.join(DATA_PATH, 'usernames_facenet.npy')

# Model Configuration
FACENET_MODEL = 'vggface2'  # 'vggface2' hoặc 'casia-webface'
FACE_SIZE = 160  # FaceNet input size
MARGIN = 20

# Recognition Thresholds
RECOGNITION_THRESHOLD = 0.6
QUALITY_THRESHOLD = 0.4
DETECTION_THRESHOLD = 0.7
K_NEAREST = 5

# Data Augmentation
MAX_EMBEDDINGS_PER_PERSON = 10
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.9, 1.1)

# Camera & Display
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FRAME_SKIP = 3
HISTORY_LENGTH = 5

# MTCNN Configuration
MTCNN_THRESHOLDS = [0.7, 0.8, 0.8]
MTCNN_MIN_FACE_SIZE = 40

# Colors (BGR)
COLOR_KNOWN = (0, 255, 0)  # Green
COLOR_UNKNOWN = (0, 0, 255)  # Red
COLOR_TEXT = (255, 255, 255)  # White
COLOR_FPS = (0, 255, 255)  # Yellow


# Đảm bảo đường dẫn tuyệt đối
IMG_PATH = os.path.abspath(IMG_PATH)
DATA_PATH = os.path.abspath(DATA_PATH)

# Khởi tạo device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tạo thư mục data nếu chưa có
os.makedirs(DATA_PATH, exist_ok=True)

# --- CÁC HÀM HỖ TRỢ ---

def fixed_image_standardization(image_tensor):
    """Chuẩn hóa ảnh theo chuẩn FaceNet: (pixel - 127.5) / 128.0"""
    # QUAN TRỌNG: FaceNet yêu cầu input trong khoảng [-1, 1]
    # Ảnh gốc có pixel [0, 255], công thức này chuyển về [-1, 1]
    # Đây là chuẩn hóa bắt buộc cho FaceNet model
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def trans(img):
    """Transform ảnh cho FaceNet"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

def augment_image(img):
    """Data augmentation - Tăng số lượng ảnh training từ 1 thành 6"""
    augmented = [img]  # Ảnh gốc
    
    # Horizontal Flip (lật ngang)
    # Giúp model nhận diện cả góc trái/phải của khuôn mặt
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Brightness Variations (thay đổi độ sáng)
    # Giúp model hoạt động tốt trong điều kiện ánh sáng khác nhau
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(BRIGHTNESS_RANGE[0]))  # Tối hơn (0.8x)
    augmented.append(enhancer.enhance(BRIGHTNESS_RANGE[1]))  # Sáng hơn (1.2x)
    
    # Contrast Variations (thay đổi độ tương phản) 
    # Giúp model ổn định với camera/màn hình khác nhau
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(CONTRAST_RANGE[0]))  # Giảm contrast (0.9x)
    augmented.append(enhancer.enhance(CONTRAST_RANGE[1]))  # Tăng contrast (1.1x)
    
    return augmented  # Trả về 6 biến thể

def compute_face_quality(face, mtcnn_prob=None):
    """Đánh giá chất lượng khuôn mặt (0-1, cao = tốt)"""
    quality = 0.5
    
    # Sử dụng MTCNN confidence (70% trọng số)
    # MTCNN probability cho biết độ chắc chắn đây là khuôn mặt
    if mtcnn_prob is not None:
        quality = mtcnn_prob * 0.7
    
    # Kiểm tra độ nét bằng Laplacian Variance (30% trọng số)
    # Laplacian variance đo độ sharp của ảnh - càng cao càng nét
    face_np = np.array(face)
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize về [0,1] - giá trị 100-500 được coi là tốt
    # < 100: quá mờ, > 500: rất nét
    blur_score = min(laplacian_var / 300.0, 1.0) * 0.3
    quality += blur_score
    
    return min(quality, 1.0)  # Giới hạn max = 1.0

def load_facelist():
    """Load FaceList (embeddings) và usernames từ file"""
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(USERNAMES_FILE):
        return None, None
    
    try:
        embeds = torch.load(EMBEDDINGS_FILE, map_location=device)
        names = np.load(USERNAMES_FILE)
        return embeds, names
    except Exception as e:
        print(f"Lỗi khi load file dữ liệu: {e}")
        return None, None

def extract_face_from_box(box, img, margin=MARGIN):
    """Trích xuất và chuẩn bị khuôn mặt từ bounding box cho FaceNet"""
    img_size = [img.shape[1], img.shape[0]]  # width, height
    
    # Tính toán margin
    margin_ratio = [
        margin * (box[2] - box[0]) / (FACE_SIZE - margin),
        margin * (box[3] - box[1]) / (FACE_SIZE - margin),
    ]
    
    # Mở rộng bounding box với margin
    box_with_margin = [
        int(max(box[0] - margin_ratio[0] / 2, 0)),
        int(max(box[1] - margin_ratio[1] / 2, 0)),
        int(min(box[2] + margin_ratio[0] / 2, img_size[0])),
        int(min(box[3] + margin_ratio[1] / 2, img_size[1])),
    ]
    
    # Crop
    img_crop = img[box_with_margin[1]:box_with_margin[3], 
                   box_with_margin[0]:box_with_margin[2]]
    
    if img_crop.size == 0:
        return None
    
    # Resize và Convert màu
    face = cv2.resize(img_crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)
    return face

def compute_facenet_embedding(model, face):
    """Tính toán embedding vector từ khuôn mặt sử dụng FaceNet"""
    with torch.no_grad():
        face_tensor = trans(face).unsqueeze(0).to(device)
        embedding = model(face_tensor)
    return embedding

def face_recognition(model, face, facelist_embeds, names_per_embedding, 
                     threshold=RECOGNITION_THRESHOLD, k=K_NEAREST):
    """
    Nhận diện khuôn mặt với k-NN voting và multiple embeddings
    Returns: (name, score) hoặc ("Unknown", score)
    """
    # Tính embedding vector cho khuôn mặt cần nhận dạng
    embed = compute_facenet_embedding(model, face)
    
    # Chuẩn hóa embeddings để tính Cosine Similarity
    # Cosine similarity = dot product của normalized vectors
    # Cho kết quả trong [-1, 1], 1 = giống nhất
    embed_normalized = embed / embed.norm(dim=1, keepdim=True)
    facelist_normalized = facelist_embeds / facelist_embeds.norm(dim=1, keepdim=True)
    
    # Tính similarity với TẤT CẢ embeddings trong database
    # So sánh với mọi embedding đã lưu (có thể có nhiều embedding/người)
    similarities = torch.mm(embed_normalized, facelist_normalized.T).squeeze()
    
    # Áp dụng k-NN - Lấy k embeddings gần nhất
    top_k_scores, top_k_indices = torch.topk(similarities, min(k, len(similarities)))
    
    # k-NN VOTING - Thuật toán quan trọng nhất  
    votes = {}  # Dictionary lưu {tên: score cao nhất}
    total_score = 0
    
    for i in range(len(top_k_scores)):
        # Bỏ qua những match có score thấp hơn threshold
        if top_k_scores[i].item() < threshold:
            continue
        
        # Lấy tên người từ index
        name = names_per_embedding[top_k_indices[i].item()]
        score = top_k_scores[i].item()
        
        # Nếu người này đã có vote, lấy score CAO NHẤT
        # (Một người có nhiều embeddings, chọn cái khớp nhất)
        if name in votes:
            votes[name] = max(votes[name], score)
        else:
            votes[name] = score
        
        total_score += score
    
    # Không ai vượt threshold -> Unknown
    if len(votes) == 0:
        return "Unknown", torch.tensor(0.0)
    
    # Chọn người có score cao nhất làm kết quả cuối cùng
    best_name = max(votes, key=votes.get)
    best_score = votes[best_name]
    
    return best_name, torch.tensor(best_score)

def update_facelist():
    """Cập nhật FaceList từ thư mục ảnh training"""
    print("\n" + "="*60)
    print("ĐANG CẬP NHẬT FACELIST VỚI FACENET")
    print("="*60)
    
    # Tự động tạo thư mục nếu chưa tồn tại
    if not os.path.exists(IMG_PATH):
        print(f"\nThư mục {IMG_PATH} chưa tồn tại!")
        os.makedirs(IMG_PATH, exist_ok=True)
        print("Đã tạo thư mục mẫu. Vui lòng thêm ảnh vào đó.")
        return False
    
    # Load FaceNet model
    print(f"Đang load FaceNet model ({FACENET_MODEL.upper()})...")
    model = InceptionResnetV1(classify=False, pretrained=FACENET_MODEL).to(device)
    model.eval()
    
    users = [d for d in os.listdir(IMG_PATH) if os.path.isdir(os.path.join(IMG_PATH, d))]
    
    if len(users) == 0:
        print("\nKhông tìm thấy thư mục người dùng nào trong assets/images!")
        return False
    
    print(f"\nTìm thấy {len(users)} người: {users}")
    print("-"*60)
    
    all_embeddings = []
    all_names = []  # Mỗi embedding tương ứng với tên người
    
    for i, usr in enumerate(users, 1):
        user_path = os.path.join(IMG_PATH, usr)
        images = glob.glob(os.path.join(user_path, '*.*')) # Lấy mọi loại file
        # Lọc chỉ lấy ảnh
        images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"[{i}/{len(users)}] Đang xử lý: {usr} ({len(images)} ảnh)")
        
        user_embeddings = []
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((FACE_SIZE, FACE_SIZE))
                
                # Đánh giá chất lượng ảnh
                quality = compute_face_quality(img)
                
                # Chỉ lưu ảnh chất lượng cao
                if quality < QUALITY_THRESHOLD:
                    continue
                
                # Data augmentation (tạo thêm biến thể)
                augmented_imgs = augment_image(img)
                
                for aug_img in augmented_imgs:
                    embedding = compute_facenet_embedding(model, aug_img)
                    user_embeddings.append(embedding)
                    
            except Exception as e:
                continue
        
        if len(user_embeddings) > 0:
            # Lưu nhiều embeddings thay vì trung bình (tốt hơn cho variation)
            # Giới hạn số lượng để không quá nhiều
            if len(user_embeddings) > MAX_EMBEDDINGS_PER_PERSON:
                # Chọn ngẫu nhiên MAX_EMBEDDINGS_PER_PERSON embeddings
                indices = np.random.choice(len(user_embeddings), MAX_EMBEDDINGS_PER_PERSON, replace=False)
                user_embeddings = [user_embeddings[idx] for idx in indices]
            
            for emb in user_embeddings:
                all_embeddings.append(emb)
                all_names.append(usr)
            
            print(f"{len(user_embeddings)} embeddings")
        else:
            print(f"Không có ảnh hợp lệ hoặc chất lượng thấp")
    
    if len(all_embeddings) == 0:
        print("\n✗ Không tạo được embedding nào!")
        return False
    
    # Lưu FaceList
    facelist = torch.cat(all_embeddings)
    names_array = np.array(all_names)
    
    torch.save(facelist, EMBEDDINGS_FILE)
    np.save(USERNAMES_FILE, names_array)
    print(f"\n✓ Đã lưu {len(all_names)} người vào FaceList.")
    return True

# --- MAIN FUNCTION ---

def main():
    """Chương trình nhận diện khuôn mặt realtime với FaceNet"""
    print("\n" + "="*60)
    print("HỆ THỐNG NHẬN DIỆN KHUÔN MẶT VỚI FACENET")
    print("="*60)
    
    # 1. Load dữ liệu
    facelist, names = load_facelist()
    
    # Nếu chưa có dữ liệu, yêu cầu cập nhật
    if facelist is None:
        print("\n⚠ Chưa có dữ liệu khuôn mặt. Đang thử cập nhật...")
        if update_facelist():
            facelist, names = load_facelist()
        else:
            print("✗ Vui lòng thêm ảnh vào thư mục assets/images và chạy lại.")
            return

    print(f"\nĐã load {len(names)} người dùng.")

    # 2. Load Model & MTCNN
    print(f"Đang khởi tạo model ({FACENET_MODEL.upper()})...")
    model = InceptionResnetV1(classify=False, pretrained=FACENET_MODEL).to(device)
    model.eval()
    
    # MTCNN với thresholds cao hơn để chất lượng tốt hơn
    mtcnn = MTCNN(keep_all=True, device=device, 
                  min_face_size=MTCNN_MIN_FACE_SIZE, 
                  thresholds=MTCNN_THRESHOLDS)
    
    # Temporal smoothing: lưu kết quả của N frames gần nhất
    face_history = {}  # {face_id: deque of (name, confidence)}
    
    # 3. Mở Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    # Tăng độ phân giải camera (nếu hỗ trợ)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("\nCamera sẵn sàng. Nhấn ESC để thoát.")
    print(f"Cấu hình: Threshold={RECOGNITION_THRESHOLD}, k-NN={K_NEAREST}")
    
    # --- CẤU HÌNH LOGIC REALTIME ---
    frame_count = 0
    tracked_faces = []   # Biến lưu trữ để vẽ (chống nhấp nháy)

    while cap.isOpened():
        start_time = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        display_frame = frame.copy()

        # Chỉ detect mỗi FRAME_SKIP frames để tăng tốc
        # Ví dụ FRAME_SKIP=3 -> detect mỗi 3 frames, FPS tăng gấp 3
        if frame_count % FRAME_SKIP == 0:
            new_tracked_faces = []
            
            # Resize nhỏ để MTCNN detect nhanh hơn
            # Detection trên ảnh nhỏ (0.5x) rồi scale tọa độ lên (x2)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            try:
                boxes, probs = mtcnn.detect(small_frame)
            except:
                boxes = None
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Tăng threshold detection để chất lượng tốt hơn
                    if probs[i] < DETECTION_THRESHOLD: continue
                    
                    # Scale toạ độ về ảnh gốc (nhân 2 vì resize 0.5)
                    bbox = list(map(int, (box * 2).tolist()))
                    
                    # Cắt mặt từ ảnh gốc để nhận diện cho chính xác
                    face = extract_face_from_box(bbox, frame)
                    
                    if face is not None:
                        # Kiểm tra chất lượng khuôn mặt
                        quality = compute_face_quality(face, probs[i])
                        
                        if quality < QUALITY_THRESHOLD:
                            continue
                        
                        # Nhận diện với k-NN
                        name_display, score = face_recognition(model, face, facelist, names)
                        
                        # Tạo face_id từ vị trí (grid 50x50 pixels)
                        # Tracking đơn giản dựa trên vị trí
                        face_id = f"{bbox[0]//50}_{bbox[1]//50}"
                        
                        # Khởi tạo history queue cho face này (max HISTORY_LENGTH frames)
                        if face_id not in face_history:
                            face_history[face_id] = deque(maxlen=HISTORY_LENGTH)
                        
                        # Lưu kết quả nhận diện vào history
                        face_history[face_id].append((name_display, score.item() if isinstance(score, torch.Tensor) else score))
                        
                        # voting qua các farmes Lấy kết quả ổn định nhất   
                        if len(face_history[face_id]) >= 2:
                            name_votes = {}
                            # Duyệt qua HISTORY_LENGTH frames gần nhất
                            for hist_name, hist_conf in face_history[face_id]:
                                if hist_name in name_votes:
                                    # Lấy confidence cao nhất cho mỗi tên
                                    name_votes[hist_name] = max(name_votes[hist_name], hist_conf)
                                else:
                                    name_votes[hist_name] = hist_conf
                            
                            # Tên xuất hiện nhiều nhất + score cao nhất = kết quả cuối
                            name_display = max(name_votes, key=name_votes.get)
                            conf_display = name_votes[name_display]
                        else:
                            # Chưa đủ history, dùng kết quả hiện tại
                            conf_display = score.item() if isinstance(score, torch.Tensor) else score
                        
                        if name_display != "Unknown":
                            color = COLOR_KNOWN
                        else:
                            conf_display = 0.0
                            color = COLOR_UNKNOWN
                        
                        new_tracked_faces.append((bbox, name_display, conf_display, color))
            
            # Cập nhật danh sách faces để vẽ
            tracked_faces = new_tracked_faces
            
            # Xóa history cũ định kỳ (mỗi FRAME_SKIP*10 frames)
            # Tránh memory leak khi faces di chuyển ra khỏi màn hình
            if frame_count % (FRAME_SKIP * 10) == 0:
                face_history.clear()

        # Vẽ chạy mọi frame để smooth, detection chỉ chạy ngắt quãng
        for (bbox, name, conf, color) in tracked_faces:
            # Vẽ Box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Vẽ Tên
            label = f"{name} ({conf*100:.0f}%)" if name != "Unknown" else "Unknown"
            
            # Background chữ
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(display_frame, (bbox[0], bbox[1] - 25), (bbox[0] + w + 10, bbox[1]), color, -1)
            cv2.putText(display_frame, label, (bbox[0] + 5, bbox[1] - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)

        # FPS
        end_time = cv2.getTickCount()
        t = (end_time - start_time) / cv2.getTickFrequency()
        fps = 1.0 / t if t > 0 else 0
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
        cv2.imshow('FaceNet Realtime', display_frame)

        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('r'): # Update
            cap.release()
            cv2.destroyAllWindows()
            if update_facelist():
                facelist, names = load_facelist()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        elif key == ord('s'): # Screenshot
            cv2.imwrite(f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg", display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Lỗi: {e}")