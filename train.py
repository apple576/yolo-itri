"""
YOLOv11-P2 訓練腳本
純淨版：僅在原生 YOLO11 基礎上加入 P2 層
用於道路瑕疵檢測（縱向裂縫、橫向裂縫、龜裂、坑洞）
"""

import os
os.environ['NCCL_SHM_DISABLE'] = '1'
# 【關鍵修改 1】強迫 Python 只看得見 GPU 2。這行必須放在 import torch 和 YOLO 之前！
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.set_device(0)

from ultralytics import YOLO


def train():
    """訓練 YOLOv11-P2 模型"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # 從配置檔創建模型（從頭訓練）
    # 或者使用預訓練權重初始化：YOLO("yolo11_p2.yaml").load("yolo11s.pt")
    model = YOLO("yolo11_p2.yaml")
    
    # 開始訓練
    print("🚀 Starting YOLOv11-P2 training...")
    results = model.train(
        data='rdd_yolo.yaml',
        epochs=300,
        imgsz=640,
        batch=16,           # 根據顯卡記憶體調整
        name='yolo11_p2',
        amp=True,           # 混合精度訓練
        workers=1,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        patience=50,        # Early stopping patience
    )
    
    print("✅ Training finished!")
    return results


def train_from_pretrained():
    """使用預訓練權重初始化訓練（推薦）- 原始 0.698 mAP 設定"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    print(f"🖥️ Using device: {device}")
    
    model = YOLO("yolov11s_wtconv.yaml")
    
    print("🚀 Starting training...")
    
                                        

    results = model.train(
        data='rdd_yolo.yaml',
        device=0,
        epochs=1000,
        imgsz=640,
        batch=8,
        name='yolov11s_wtconv',
        amp=True,
        workers=4,
        cache= False,
        patience=50,

        
        # === 原始優化器設定 ===
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        close_mosaic=10,
        
        # === 原始損失函數權重 ===
        box=7.5,
        cls=1.0,
        dfl=1.5,
        degrees=0.0,
        scale=0.5,
        label_smoothing=0.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.7,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr= 0.1,
        multi_scale=True,
        cos_lr=True,
        weight_decay= 0.001,
        momentum= 0.937,
    )

    
    
    print("✅ Training finished!")
    return results


def resume_training():
    print("Resuming training")
    model = YOLO("/home/jovyan/work/shared_data/runs/detect/yolov11s_p2_baseline5/weights/last.pt")
    
    model.train(
        resume=True,          # <--- 開啟重續功能
        device=[0] # 只需要指定乾淨的設備，其他參數 YOLO 會自己讀取
    )




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train yolov11s_wtconv for Road Defect Detection')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    group.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()
    
    if args.resume:
        resume_training()
    elif args.pretrained:
        train_from_pretrained()
    else:
        train()
