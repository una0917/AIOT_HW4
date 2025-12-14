import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ===============================
# 基本設定
# ===============================
MODEL_PATH = "myna_model.h5"
LABELS_ZH = ["土八哥", "白尾八哥", "家八哥"]

st.set_page_config(page_title="八哥辨識器", layout="centered")

st.title("🐦 八哥辨識器")
st.write("請上傳一張八哥照片，我會幫你判斷是哪一種八哥")

# ===============================
# 載入模型
# ===============================
@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# ===============================
# 上傳圖片
# ===============================
uploaded_file = st.file_uploader(
    "請上傳圖片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="上傳的圖片", width=400)

    # 預處理
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = img_array.reshape((1, 224, 224, 3))
    img_array = preprocess_input(img_array)

    # 預測
    preds = model.predict(img_array)[0]

    st.subheader("🔍 辨識結果（機率）")

    # ===== 文字顯示 =====
    for i, label in enumerate(LABELS_ZH):
        st.write(f"{label}: {preds[i]*100:.2f}%")

    # ===== 圖表顯示 =====
    st.subheader("📊 機率分佈圖")

    chart_data = {
        LABELS_ZH[i]: preds[i] * 100
        for i in range(len(LABELS_ZH))
    }

    st.bar_chart(chart_data)

    # ===== 最終判斷 =====
    st.success(f"最可能是：**{LABELS_ZH[np.argmax(preds)]}**")

