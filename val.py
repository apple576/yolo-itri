from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 載入你訓練好的模型權重 (請換成你實際的 best.pt 路徑)
    model = YOLO(r'C:\Users\user\Desktop\yolo-main\runs\detect\yolov11s_wtconv3\weights\best.pt')

    # 2. 執行驗證 (Validation)
    print("開始執行 Validation，並套用 NMS 後處理...")
    
    metrics = model.val(
        data="rdd_yolo.yaml",   
        split="val",        
        imgsz=640,          
        batch=8,           
        device=0,
        conf=0.25,                 # 論文建議的 Confidence Threshold
        iou=0.45,                  # 論文建議的 IoU Threshold       
    )

    # 3. 印出結果
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")