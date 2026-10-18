import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Args:
        data_dir: đường dẫn thư mục gốc chứa các class folders
        output_dir: đường dẫn thư mục output
        train_ratio: tỷ lệ train (0.8 = 80% train, 20% test)
        seed: random seed để có kết quả reproducible
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Tạo thư mục train/test
    for split in ['train', 'test']:
        for class_name in os.listdir(data_path):
            class_dir = data_path / class_name
            if class_dir.is_dir():
                (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Xử lý từng class
    for class_name in os.listdir(data_path):
        class_path = data_path / class_name
        if not class_path.is_dir():
            continue
        
        # Lấy tất cả ảnh
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if len(images) == 0:
            print(f"  {class_name}: không có ảnh nào")
            continue
        
        # Sử dụng train_test_split
        test_ratio = 1 - train_ratio
        train_imgs, test_imgs = train_test_split(
            images,
            test_size=test_ratio,
            random_state=seed
        )
        
        # Copy ảnh vào train folder
        for img in train_imgs:
            src = class_path / img
            dst = output_path / 'train' / class_name / img
            shutil.copy2(src, dst)
        
        # Copy ảnh vào test folder
        for img in test_imgs:
            src = class_path / img
            dst = output_path / 'test' / class_name / img
            shutil.copy2(src, dst)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test")


if __name__ == "__main__":
    # Chạy
    split_dataset(
        data_dir="foody_argument",         
        output_dir="foody",        
        train_ratio=0.8            
    )
    print("\n Hoàn thành!")