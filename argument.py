import os
import cv2
import random

input_root = "foody_clean"
output_root = "foody_argument"

os.makedirs(output_root, exist_ok=True)

def augment(img):
    h, w = img.shape[:2]

    flip = cv2.flip(img, 1)

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotate = cv2.warpAffine(img, M, (w, h))

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return [flip, rotate, bright]


# 🔥 hàm quyết định số lần augment
def get_aug_times(count):
    if count < 400:
        return 4   # rất ít → augment mạnh
    elif count < 800:
        return 2   # trung bình
    else:
        return 0   # nhiều → không augment


for class_name in os.listdir(input_root):
    input_class_path = os.path.join(input_root, class_name)

    if not os.path.isdir(input_class_path):
        continue

    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    files = os.listdir(input_class_path)
    class_count = len(files)

    aug_times = get_aug_times(class_count)

    print(f"{class_name}: {class_count} ảnh → augment {aug_times} lần")

    count = 0

    for filename in files:
        img_path = os.path.join(input_class_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # lưu ảnh gốc
        cv2.imwrite(os.path.join(output_class_path, f"{count}_orig.jpg"), img)

        # augment theo số lần
        if aug_times > 0:
            aug_list = augment(img)

            for i in range(aug_times):
                aug = random.choice(aug_list)
                cv2.imwrite(os.path.join(output_class_path, f"{count}_{i}.jpg"), aug)

        count += 1