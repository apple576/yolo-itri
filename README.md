# YOLOv11 專案執行指南

本文件提供完整的專案下載、Conda 環境建立與 VS Code 推論（Inference）執行步驟。

---

## 🛠️ 完整操作步驟

### 1. 下載與解壓縮專案
1. 開啟本 GitHub 頁面，點擊右上角的 **Code** 按鈕 ➔ 選擇 **Download ZIP**。
2. 下載完成後，將檔案解壓縮至你的電腦中。

---

### 2. 建立 Conda 虛擬環境
1. 開啟 **Anaconda Prompt**。

2. 切換路徑至剛解壓縮的專案資料夾：
```bash
   cd 到剛剛解壓縮的資料夾路徑

```
   
3. 輸入指令讀取 `environment.yaml` 建立環境：
```bash
conda env create -f environment.yaml

```

---

### 3. 使用 VS Code 執行推論 (`predict.py`)

1. 開啟 **VS Code**。
2. 點選選單 `File` ➔ `Open Folder...`，開啟剛剛解壓縮的專案資料夾。
3. 在左側檔案樹中點開 **`predict.py`**。
4. 修改 `predict.py` 最下方的參數設定區：
拉到程式碼最底下 `if __name__ == '__main__':` 區塊，
依據你電腦中的資料夾位置修改以下 4 個路徑變數：


# ============ 參數設定區 ============
IMG_FOLDER = r'C:\Users\user\Downloads\dataset\val\images'      # 你的圖片資料夾路徑
LABEL_FOLDER = r'C:\Users\user\Downloads\dataset\val\labels'    # 你的 GT 標註資料夾路徑
MODEL_WEIGHTS = r'C:\Users\user\Desktop\yolo-main\runs\detect\yolov11s_wtconv3\weights\best.pt' # 模型權重路徑
OUTPUT_FOLDER = r'C:\Users\user\Desktop\results_WTConv_conf0001'   # 想要儲存結果的資料夾名稱
# ====================================
📌 參數詳細說明：

IMG_FOLDER：放置待測圖片（例如驗證集 images）的絕對或相對路徑。

LABEL_FOLDER：放置對應 Ground Truth 標註檔（.txt）的資料夾路徑。

MODEL_WEIGHTS：訓練好的 .pt 模型權重路徑（預設放在專案內 runs/detect/yolov11s_wtconv3/weights/best.pt）。

OUTPUT_FOLDER：執行完成後預計儲存結果圖表與數據的輸出資料夾名稱。

5. **修改推論參數**：
* 找到 `process_folders` 函式內的模型推論程式碼：
```python
results = model(img, conf=0.001, iou=0.45, verbose=False)

```


* 可依需求自行修改信心度門檻（`conf`）與交併比門檻（`iou`）。


6. **切換 Python 執行環境**：
* 點擊 VS Code 右下角的 Python 環境選擇器（或按 `Ctrl + Shift + P` 輸入 `Python: Select Interpreter`）。
* 切換環境為剛才 2-2 建立的 **`yolov11_test`**。


7. **執行程式**：
* 按下右上角的 Play 按鈕（或在 Terminal 中執行 `python predict.py`）開始跑推論。



```

```