import shutil
from pathlib import Path

from ultralytics import YOLO

# =====================================================================
# ⚙️ 1. 設定區塊 (請根據你的路徑進行修改)
# =====================================================================
MODEL_PATH = r"C:\Users\user\Desktop\yolo-main\runs\detect\yolov11s_wtconv3\weights\best.pt"  # 你的模型權重路徑
VAL_IMAGES_DIR = r"C:\Users\user\Downloads\dataset\val\images"  # 驗證集圖片資料夾
VAL_LABELS_DIR = r"C:\Users\user\Downloads\dataset\val\labels"  # 驗證集標籤資料夾 (.txt)
OUTPUT_DIR = r"bad_cases"  # 輸出 Bad Cases 的總資料夾

# 你指定的 6 個目標類別 ID (過濾條件)
TARGET_CLASSES = {7, 8, 10, 11, 12, 13}

# 設定檢測門檻
IOU_THRESHOLD = 0.50  # IoU 判定標準 (50%)
CONF_THRESHOLD = 0.25  # 模型預測的信心度門檻


# =====================================================================
# 📐 2. IoU 計算工具函式
# =====================================================================
def compute_iou(box1, box2):
    """計算兩組 Bounding Box [x1, y1, x2, y2] 的 IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0


def xywh2xyxy(box, img_w, img_h):
    """將 YOLO 的歸一化 [xc, yc, w, h] 轉為絕對座標 [x1, y1, x2, y2]."""
    xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


# =====================================================================
# 🚀 3. 主流程邏輯
# =====================================================================
def main():
    # 建立輸出資料夾
    fp_dir = Path(OUTPUT_DIR) / "False_Positives"
    fn_dir = Path(OUTPUT_DIR) / "False_Negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在載入模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image_paths = list(Path(VAL_IMAGES_DIR).glob("*.[jJ][pP][gG]")) + list(Path(VAL_IMAGES_DIR).glob("*.[pP][nN][gG]"))

    fp_count = 0
    fn_count = 0

    print(f"🔍 開始分析 {len(image_paths)} 張驗證集圖片 (IoU Thresh = {IOU_THRESHOLD})...\n")

    for img_path in image_paths:
        label_path = Path(VAL_LABELS_DIR) / f"{img_path.stem}.txt"

        # 1. 讀取 Ground Truth (GT)
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        gt_boxes.append({"cls": cls_id, "bbox_raw": bbox})

        # 2. 執行模型預測
        results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, iou=0.45, verbose=False)[0]
        img_h, img_w = results.orig_shape

        # 轉化 GT 絕對座標
        for gt in gt_boxes:
            gt["bbox"] = xywh2xyxy(gt["bbox_raw"], img_w, img_h)

        # 讀取 Prediction (Pred)
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].cpu().tolist()
                pred_boxes.append({"cls": cls_id, "bbox": xyxy})

        # 3. 比對 FP / FN 狀態 (加上 TARGET_CLASSES 過濾)
        is_fp = False
        is_fn = False

        # --- 檢查 False Positives (誤判) ---
        # 模型預測出了標籤，但在 GT 找不到同類別且 IoU >= 0.50 的匹配
        for pred in pred_boxes:
            if pred["cls"] in TARGET_CLASSES:
                matched = False
                for gt in gt_boxes:
                    if pred["cls"] == gt["cls"]:
                        if compute_iou(pred["bbox"], gt["bbox"]) >= IOU_THRESHOLD:
                            matched = True
                            break
                if not matched:
                    is_fp = True
                    break

        # --- 檢查 False Negatives (漏檢) ---
        # GT 有真實標籤，但在模型預測中找不到同類別且 IoU >= 0.50 的匹配
        for gt in gt_boxes:
            if gt["cls"] in TARGET_CLASSES:
                matched = False
                for pred in pred_boxes:
                    if gt["cls"] == pred["cls"]:
                        if compute_iou(gt["bbox"], pred["bbox"]) >= IOU_THRESHOLD:
                            matched = True
                            break
                if not matched:
                    is_fn = True
                    break

        # 4. 複製圖片 (安全複製，絕不改動原始資料集)
        if is_fp:
            shutil.copy(img_path, fp_dir / img_path.name)
            fp_count += 1

        if is_fn:
            shutil.copy(img_path, fn_dir / img_path.name)
            fn_count += 1

    print("==================================================")
    print("🎉 分析完成！Bad Cases 統計結果：")
    print(f"❌ 誤判案例 (False Positives) 複製數量: {fp_count} 張")
    print(f"🙈 漏檢案例 (False Negatives) 複製數量: {fn_count} 張")
    print(f"📁 圖片已分別存入: {OUTPUT_DIR}/")
    print("==================================================")


if __name__ == "__main__":
    main()
