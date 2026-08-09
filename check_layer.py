from ultralytics import YOLO

model = YOLO(r'C:\Users\user\Desktop\yolo-main\runs\detect\yolov11s_wtconv3\weights\best.pt')

print(f"\n{'Layer':<8} {'Module Name':<30} {'Params':<12}")
print("=" * 52)

# 遍歷每一層，只抓最外層的模組類別名稱與參數量
for idx, layer in enumerate(model.model.model):
    module_name = layer.__class__.__name__
    params = sum(p.numel() for p in layer.parameters())
    print(f"{idx:<8} {module_name:<30} {params:<12,}")

print("=" * 52)