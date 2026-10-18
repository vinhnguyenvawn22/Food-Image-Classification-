from argparse import ArgumentParser
import torch
from models import SimpleCNN
import cv2
import numpy as np
import torch.nn as nn
from torchsummary import summary

def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image-path", "-p", type=str, default=None)
    parser.add_argument("--image-size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_cnn.pt")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Hiển thị top-k dự đoán")
    return parser.parse_args()


def draw_results(image, categories, proba_np, top_k=5):
    h, w = image.shape[:2]
    display = image.copy()

    top_indices = np.argsort(proba_np)[::-1][:top_k]
    best_idx = top_indices[0]
    best_label = categories[best_idx]
    best_conf = proba_np[best_idx]

    # --- overlay nền mờ phía trên ---
    overlay = display.copy()
    panel_h = 30  # giảm từ 50 → 30
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # --- tên món + confidence ---
    label_text = f"{best_label}  {best_conf*100:.1f}%"
    cv2.putText(display, label_text, (8, 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # font nhỏ hơn

    # --- bar chart top-k ở góc dưới trái ---
    row_h = 16          # giảm từ 28 → 16
    bar_h = 8           # chiều cao thanh bar
    bar_w_max = 80      # giảm từ 200 → 80
    label_offset = 6    # khoảng cách từ bar đến text

    bar_x0 = 8
    bar_y0 = h - (top_k * row_h + 10)

    bg = display.copy()
    cv2.rectangle(bg, (0, bar_y0 - 4), (bar_x0 + bar_w_max + 90, h), (0, 0, 0), -1)
    cv2.addWeighted(bg, 0.5, display, 0.5, 0, display)

    for rank, idx in enumerate(top_indices):
        y = bar_y0 + rank * row_h
        conf = proba_np[idx]
        bar_len = int(conf * bar_w_max)
        color = (80, 200, 80) if rank == 0 else (160, 160, 160)

        cv2.rectangle(display, (bar_x0, y), (bar_x0 + bar_len, y + bar_h), color, -1)
        cv2.rectangle(display, (bar_x0, y), (bar_x0 + bar_w_max, y + bar_h), (200, 200, 200), 1)

        name = categories[idx]
        cv2.putText(display, f"{name}  {conf*100:.1f}%",
                    (bar_x0 + bar_w_max + label_offset, y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)  # font nhỏ hơn

    return display, best_label, best_conf


if __name__ == "__main__":
    categories = [
        'banh_cuon', 'banh_mi', 'banh_ngot', 'banh_trang_nuong',
        'bun_bo', 'bun_cha', 'bun_rieu', 'com_tam',
        'ga_ran', 'hu_tieu', 'nem_nuong', 'pho',
        'pizza', 'tra_sua', 'vit_quay'
    ]
    args = get_args()
    
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SimpleCNN(numclass=15).to(device)
    summary(model, (3,224,224))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("Không tìm thấy checkpoint.")
        exit(0)

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)

    with torch.no_grad():
        proba = nn.Softmax(dim=1)(model(tensor)).cpu().numpy()[0]

    result_img, best_label, best_conf = draw_results(ori_image, categories, proba, top_k=args.top_k)
 
    print(f"\n  Dự đoán: {best_label}  ({best_conf*100:.2f}%)")
    for i, idx in enumerate(np.argsort(proba)[::-1][:args.top_k]):
        print(f"  {'>' if i==0 else ' '} [{i+1}] {categories[idx]:<20} {proba[idx]*100:.2f}%")
    
    display_img = cv2.resize(result_img, (600, int(600 * ori_image.shape[0] / ori_image.shape[1])))
    cv2.imshow("Food Classifier", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()