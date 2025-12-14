# 🐦 AIOT HW4 – 八哥影像辨識系統（Myna Classification）

## 📌 專案簡介

本專案為 **AIoT 課程作業 HW4**，目標是利用**深度學習影像分類技術**，建立一個可透過網頁操作的八哥（Myna）辨識系統。  
使用者只需上傳一張八哥照片，系統即可自動判斷該影像屬於哪一種八哥，並以**機率數值與視覺化圖表**顯示辨識結果。

本系統採用 **Transfer Learning** 架構，以 **ResNet50V2** 作為特徵擷取模型，並結合 **Data Augmentation** 技術，在有限資料下提升模型泛化能力。最終透過 **Streamlit** 建立互動式 Web 介面，並部署至 **Streamlit Cloud**，使所有使用者皆可直接在線上體驗。

---

## 🚀 Demo（線上展示）

👉 **Streamlit Cloud Demo**  
🔗 https://aiothw4-xsd8wb6ecw6clxpp5jz8ki.streamlit.app/

### 使用方式
1. 開啟上述連結  
2. 上傳一張八哥圖片（`jpg / png`）  
3. 系統即顯示三種類型八哥的預測機率與最終辨識結果  

---

## 🐤 辨識類別

本專案辨識三種台灣常見的八哥：

| 類別資料夾 | 中文名稱 |
|-----------|---------|
| crested_myna | 土八哥 |
| javan_myna | 白尾八哥 |
| common_myna | 家八哥 |

---

## 🧠 系統方法說明

### 1️⃣ 模型架構
- Backbone：**ResNet50V2（ImageNet 預訓練權重）**
- 凍結 ResNet 權重（Frozen Backbone）
- 僅訓練最後一層 Dense（Softmax）

### 2️⃣ 資料處理
- 輸入影像統一 resize 為 `224 × 224`
- 使用 `preprocess_input` 進行正規化

### 3️⃣ Data Augmentation
在訓練階段加入以下影像增強方式：
- 隨機旋轉
- 平移
- 縮放
- 水平翻轉  

以提升模型對不同拍攝角度與條件的適應能力。

### 4️⃣ 資料切分
- 訓練集（Training）：80%
- 驗證集（Validation）：20%

---

## 📊 辨識結果呈現方式

系統會同時顯示：
- 各類別的**預測機率（百分比）**
- **長條圖（Bar Chart）**視覺化機率分佈
- 最可能的八哥種類（Highest Probability）

---

## 📁 專案目錄結構

```text
AIOT_HW4/
│
├─ main.py                  # 模型訓練程式（含 Data Augmentation）
├─ app.py                   # Streamlit Web App（模型推論）
├─ requirements.txt         # 套件需求清單
├─ README.md                # 專案說明文件
│
├─ myna/                    # 八哥影像資料集
│   ├─ crested_myna/
│   ├─ javan_myna/
│   └─ common_myna/
│
└─ myna_model.h5            # 訓練完成後的模型檔


## 🔧 執行方式（本地端）
1️⃣ 安裝套件
pip install -r requirements.txt

2️⃣ 訓練模型（只需一次）
python main.py

3️⃣ 啟動 Web App
streamlit run app.py

## ✨ 與參考資料相比的改進與新增內容

本專案以以下教學作為基礎進行延伸與改進：

### 📚 參考資料
https://github.com/yenlung/AI-Demo

在原始範例的基礎上，本專案新增與改進了以下功能：
專案模組化
將訓練 (main.py) 與推論介面 (app.py) 分離
符合實務專案結構
資料切分（Training / Validation）
原始範例使用所有資料進行訓練
本專案加入 Validation Split 以評估模型泛化能力
Data Augmentation
加入影像隨機變換，提高模型對資料不足情況的適應性
模型保存與重複使用
訓練完成後將模型儲存為 .h5
推論階段直接載入模型，避免重複訓練
Streamlit Web 介面
取代原始 Gradio 範例
提供更直覺的 UI 與圖表視覺化
雲端部署
將專案部署至 Streamlit Cloud
使所有使用者可直接線上體驗


## 📌 使用技術

Python
TensorFlow / Keras
ResNet50V2
Streamlit
NumPy / Pillow / SciPy
