import cv2
import numpy as np
import pywt
from PIL import Image
import os
from pathlib import Path
import pickle

class WaveletImageHash:
    """Lớp tạo hash cho hình ảnh sử dụng Wavelet Transform"""
    
    def __init__(self, hash_size=8, wavelet='haar'):
        """
        Khởi tạo WaveletImageHash
        
        Parameters:
        - hash_size: Kích thước của hash (mặc định 8x8 = 64 bit)
        - wavelet: Loại wavelet sử dụng (haar, db1, sym2, etc.)
        """
        self.hash_size = hash_size
        self.wavelet = wavelet
    
    def compute_hash(self, image_path):
        """
        Tính toán wavelet hash cho một hình ảnh
        
        Parameters:
        - image_path: Đường dẫn đến file hình ảnh
        
        Returns:
        - hash_array: Mảng hash của hình ảnh
        """
        # Đọc ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Resize ảnh về kích thước chuẩn
        img = cv2.resize(img, (self.hash_size * 2, self.hash_size * 2))
        
        # Normalize ảnh về [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Áp dụng Discrete Wavelet Transform 2D
        coeffs = pywt.dwt2(img, self.wavelet)
        cA, (cH, cV, cD) = coeffs
        
        # Sử dụng hệ số xấp xỉ (LL - low-low subband)
        # Đây là phần chứa thông tin quan trọng nhất của ảnh
        cA = cv2.resize(cA, (self.hash_size, self.hash_size))
        
        # Tạo hash binary dựa trên giá trị trung bình
        median = np.median(cA)
        hash_array = (cA > median).astype(np.uint8)
        
        return hash_array
    
    def hash_to_string(self, hash_array):
        """Chuyển hash array thành chuỗi hex"""
        hash_flat = hash_array.flatten()
        hash_int = int(''.join(str(bit) for bit in hash_flat), 2)
        return hex(hash_int)
    
    def hamming_distance(self, hash1, hash2):
        """
        Tính khoảng cách Hamming giữa 2 hash
        
        Parameters:
        - hash1, hash2: Mảng hash để so sánh
        
        Returns:
        - distance: Số bit khác nhau (0-64 với hash_size=8)
        """
        return np.sum(hash1 != hash2)
    
    def similarity(self, hash1, hash2):
        """
        Tính độ tương đồng giữa 2 hash (0-100%)
        
        Returns:
        - similarity: Phần trăm tương đồng
        """
        distance = self.hamming_distance(hash1, hash2)
        max_distance = self.hash_size * self.hash_size
        return (1 - distance / max_distance) * 100


class ImageSearchEngine:
    """Công cụ tìm kiếm hình ảnh sử dụng Wavelet Hash"""
    
    def __init__(self, database_path='image_database.pkl'):
        """
        Khởi tạo công cụ tìm kiếm
        
        Parameters:
        - database_path: Đường dẫn lưu database hash
        """
        self.hasher = WaveletImageHash()
        self.database_path = database_path
        self.database = {}  # {image_path: hash_array}
        self.load_database()
    
    def index_images(self, folder_path):
        """
        Đánh index tất cả ảnh trong một thư mục
        
        Parameters:
        - folder_path: Đường dẫn thư mục chứa ảnh
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Thư mục không tồn tại: {folder_path}")
            return
        
        print(f"Đang đánh index hình ảnh từ: {folder_path}")
        count = 0
        
        for img_path in folder.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    hash_array = self.hasher.compute_hash(str(img_path))
                    self.database[str(img_path)] = hash_array
                    count += 1
                    if count % 10 == 0:
                        print(f"Đã xử lý {count} ảnh...")
                except Exception as e:
                    print(f"Lỗi khi xử lý {img_path}: {e}")
        
        print(f"Hoàn tất! Đã đánh index {count} ảnh.")
        self.save_database()
    
    def search(self, query_image_path, top_k=5, threshold=70):
        """
        Tìm kiếm ảnh tương tự
        
        Parameters:
        - query_image_path: Đường dẫn ảnh truy vấn
        - top_k: Số lượng kết quả trả về
        - threshold: Ngưỡng tương đồng tối thiểu (%)
        
        Returns:
        - results: List các tuple (image_path, similarity_score)
        """
        if not os.path.exists(query_image_path):
            print(f"Ảnh truy vấn không tồn tại: {query_image_path}")
            return []
        
        # Kiểm tra xem đường dẫn có phải là file không
        if not os.path.isfile(query_image_path):
            print(f"Đường dẫn không phải là file ảnh: {query_image_path}")
            print("Vui lòng nhập đường dẫn đến một file ảnh, không phải thư mục.")
            return []
        
        # Tính hash cho ảnh truy vấn
        query_hash = self.hasher.compute_hash(query_image_path)
        
        # So sánh với tất cả ảnh trong database
        results = []
        for img_path, img_hash in self.database.items():
            similarity = self.hasher.similarity(query_hash, img_hash)
            if similarity >= threshold:
                results.append((img_path, similarity))
        
        # Sắp xếp theo độ tương đồng giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def save_database(self):
        """Lưu database hash ra file"""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"Database đã được lưu vào: {self.database_path}")
    
    def load_database(self):
        """Tải database hash từ file"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                self.database = pickle.load(f)
            print(f"Đã tải {len(self.database)} ảnh từ database")
        else:
            print("Chưa có database. Hãy đánh index ảnh trước.")


def main():
    """Hàm main để demo ứng dụng"""
    print("=" * 60)
    print("ỨNG DỤNG TÌM KIẾM HÌNH ẢNH VỚI WAVELET HASH")
    print("=" * 60)
    
    # Khởi tạo search engine
    engine = ImageSearchEngine()
    
    while True:
        print("\n--- MENU ---")
        print("1. Đánh index thư mục ảnh")
        print("2. Tìm kiếm ảnh tương tự")
        print("3. Xem thông tin database")
        print("4. Thoát")
        
        choice = input("\nNhập lựa chọn (1-4): ").strip()
        
        if choice == '1':
            folder = input("Nhập đường dẫn thư mục ảnh: ").strip()
            engine.index_images(folder)
        
        elif choice == '2':
            if len(engine.database) == 0:
                print("\nDatabase trống! Vui lòng đánh index ảnh trước.")
                continue
            
            # Hiển thị một số ảnh mẫu từ database
            print("\nMột số ảnh trong database:")
            for i, img_path in enumerate(list(engine.database.keys())[:5], 1):
                print(f"  {i}. {img_path}")
            if len(engine.database) > 5:
                print(f"  ... và {len(engine.database) - 5} ảnh khác")
            
            query_path = input("\nNhập đường dẫn ảnh truy vấn: ").strip()
            top_k = int(input("Số lượng kết quả (mặc định 5): ").strip() or "5")
            threshold = float(input("Ngưỡng tương đồng % (mặc định 70): ").strip() or "70")
            
            print(f"\nĐang tìm kiếm...")
            results = engine.search(query_path, top_k, threshold)
            
            if results:
                print(f"\nTìm thấy {len(results)} ảnh tương tự:")
                print("-" * 80)
                for i, (path, score) in enumerate(results, 1):
                    print(f"{i}. {path}")
                    print(f"   Độ tương đồng: {score:.2f}%")
                    print("-" * 80)
            else:
                print("\nKhông tìm thấy ảnh tương tự với ngưỡng này.")
        
        elif choice == '3':
            print(f"\nThông tin database:")
            print(f"- Số lượng ảnh: {len(engine.database)}")
            print(f"- Đường dẫn database: {engine.database_path}")
            
            if len(engine.database) > 0:
                view_all = input("\nXem tất cả ảnh trong database? (y/n): ").strip().lower()
                if view_all == 'y':
                    print("\nDanh sách tất cả ảnh:")
                    for i, img_path in enumerate(engine.database.keys(), 1):
                        print(f"  {i}. {img_path}")
        
        elif choice == '4':
            print("\nCảm ơn bạn đã sử dụng!")
            break
        
        else:
            print("\nLựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()