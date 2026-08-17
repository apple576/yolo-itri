import cv2
import os
from ultralytics import YOLO

def calculate_iou(boxA, boxB):
    # box 格式為 [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 計算交集面積，若沒有交集則為 0
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # 計算各自的面積
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 計算聯集面積
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def yolo_to_xyxy(yolo_coords, img_width, img_height):
    # (與之前相同，省略註解保持版面簡潔)
    x_c = float(yolo_coords[1]) * img_width
    y_c = float(yolo_coords[2]) * img_height
    w = float(yolo_coords[3]) * img_width
    h = float(yolo_coords[4]) * img_height
    
    x1 = int(x_c - (w / 2))
    y1 = int(y_c - (h / 2))
    x2 = int(x_c + (w / 2))
    y2 = int(y_c + (h / 2))
    return x1, y1, x2, y2

def process_folders(img_dir, gt_dir, model_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = YOLO(model_path)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    for img_path in image_paths:
        # 取得檔名與原始副檔名 (例如: filename='sample', ext='.png')
        filename, ext = os.path.splitext(os.path.basename(img_path))
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w, _ = img.shape
        gt_txt_path = os.path.join(gt_dir, f"{filename}.txt")

        # ==========================================
        # 先用「乾淨的原始圖片」進行模型推論
        # ==========================================
        results = model(img, conf=0.01, iou=0.45, verbose=False) 

        # ==========================================
        # 開始在圖片上作畫 (此時 img 已經不需要再丟給模型了)
        # ==========================================
        
        gt_boxes = [] # <--- 新增：建立一個空的 list 來裝這張圖的所有 GT 座標
        
        # 畫 Ground Truth (綠色)
        if os.path.exists(gt_txt_path):
            with open(gt_txt_path, 'r') as f:
                for line in f.readlines():
                    coords = line.strip().split()
                    if len(coords) >= 5:
                        x1, y1, x2, y2 = yolo_to_xyxy(coords, img_w, img_h)
                        
                        gt_boxes.append([x1, y1, x2, y2]) # <--- 新增：把座標存起來算 IoU 用
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, 'GT', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 畫 Prediction (紅色)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f'Pred: {conf:.2f}'
                cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # ==========================================
                # <--- 新增以下區塊：計算並畫出 IoU
                # ==========================================
                max_iou = 0.0
                for gt_box in gt_boxes:
                    iou = calculate_iou([x1, y1, x2, y2], gt_box)
                    if iou > max_iou:
                        max_iou = iou
                
                # 如果有配對到 GT 框 (IoU > 0)，就畫上黃色的 IoU 數值
                if max_iou > 0:
                    iou_label = f'IoU: {max_iou:.2f}'
                    cv2.putText(img, iou_label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
                # ==========================================

        # ==========================================
        # 使用原始副檔名 (ext) 進行儲存
        # ==========================================
        output_path = os.path.join(output_dir, f"{filename}_result{ext}")
        cv2.imwrite(output_path, img)
        print(f"✅ 完成: {filename}_result{ext}")

def process_single_image(img_path, gt_dir, model_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = YOLO(model_path)
    
    # 取得檔名與原始副檔名 (例如: filename='sample', ext='.png')
    filename, ext = os.path.splitext(os.path.basename(img_path))
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 無法讀取圖片，請檢查路徑: {img_path}")
        return
        
    img_h, img_w, _ = img.shape
    
    # 自動去 GT 資料夾找對應檔名的 txt
    gt_txt_path = os.path.join(gt_dir, f"{filename}.txt")

    # ==========================================
    # 先用「乾淨的原始圖片」進行模型推論
    # ==========================================
    results = model(img,conf=0.25, iou=0.7, verbose=False) 

    # ==========================================
    # 開始在圖片上作畫
    # ==========================================
    gt_boxes = [] 
    
    # 畫 Ground Truth (綠色)
    if os.path.exists(gt_txt_path):
        with open(gt_txt_path, 'r') as f:
            for line in f.readlines():
                coords = line.strip().split()
                if len(coords) >= 5:
                    x1, y1, x2, y2 = yolo_to_xyxy(coords, img_w, img_h)
                    gt_boxes.append([x1, y1, x2, y2]) 
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, 'GT', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print(f"⚠️ 找不到對應的標註檔: {gt_txt_path}")
    
    # 畫 Prediction (紅色)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'Pred: {conf:.2f}'
            cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 計算並畫出 IoU
            max_iou = 0.0
            for gt_box in gt_boxes:
                iou = calculate_iou([x1, y1, x2, y2], gt_box)
                if iou > max_iou:
                    max_iou = iou
            
            '''
            # 如果有配對到 GT 框 (IoU > 0)，畫上黃色的 IoU 數值
            if max_iou > 0:
                iou_label = f'IoU: {max_iou:.2f}'
                cv2.putText(img, iou_label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
            '''
    # ==========================================
    # 儲存結果
    # ==========================================
    output_path = os.path.join(output_dir, f"{filename}_result{ext}")
    cv2.imwrite(output_path, img)
    print(f"✅ 完成: 儲存至 {output_path}")


# 執行區塊
if __name__ == '__main__':

    # ================= 參數設定區 =================
    IMG_FOLDER = r'C:\Users\user\Downloads\dataset\val\images'      # 你的圖片資料夾路徑
    LABEL_FOLDER = r'C:\Users\user\Downloads\dataset\val\labels'    # 你的 GT 標註資料夾路徑
    MODEL_WEIGHTS = r'C:\Users\user\Desktop\best(3).pt' # 模型權重路徑
    OUTPUT_FOLDER = r'C:\Users\user\Desktop\results_WTConv_conf001_thead'   # 想要儲存結果的資料夾名稱
    # ==============================================

    process_folders(IMG_FOLDER, LABEL_FOLDER, MODEL_WEIGHTS, OUTPUT_FOLDER)


    r"""
    # 1. 指定「單張圖片」的完整路徑
    IMG_PATH = r'C:\Users\user\Downloads\dataset\val\images\China_MotorBike_000014.jpg' 
    
    # 2. 指定 GT 標註的「資料夾路徑」(程式會自動用圖片檔名去裡面找 .txt)
    LABEL_FOLDER = r'C:\Users\user\Downloads\dataset\val\labels' 
    
    # 3. 模型權重路徑
    MODEL_WEIGHTS = r'C:\Users\user\Desktop\best(1).pt' 
    
    # 4. 儲存結果的資料夾
    OUTPUT_FOLDER = r'C:\Users\user\Desktop' 
    
    process_single_image(IMG_PATH, LABEL_FOLDER, MODEL_WEIGHTS, OUTPUT_FOLDER)
    """