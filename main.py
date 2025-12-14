import os
import numpy as np

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 基本設定
# ===============================
CATEGORIES = ["crested_myna", "javan_myna", "common_myna"]
LABELS_ZH = ["土八哥", "白尾八哥", "家八哥"]

DATA_DIR = "./myna"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "myna_model.h5"

N = len(CATEGORIES)

# ===============================
# Data Augmentation
# ===============================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # ⭐ 80% train / 20% validation
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# 建立模型（Transfer Learning）
# ===============================
resnet = ResNet50V2(
    include_top=False,
    pooling="avg",
    weights="imagenet"
)
resnet.trainable = False

model = Sequential([
    resnet,
    Dense(N, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# 訓練
# ===============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===============================
# 儲存模型
# ===============================
model.save(MODEL_PATH)
print(f"模型已儲存為 {MODEL_PATH}")
