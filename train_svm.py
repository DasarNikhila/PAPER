import os
import numpy as np
import pickle

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ===============================
# CONFIG
# ===============================
DATASET_DIR = "dataset/train"
IMAGE_SIZE = (224, 224)
CLASSES = ["Black Rot", "Healthy", "Insect Hole"]

# augmentation settings (reduced for speed)
AUG_PER_IMAGE = 1

# SVM grid (balanced for speed and accuracy)
GRID_PARAMS = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

print("🔹 Loading CNN feature extractor...")

feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

X, y = [], []

print("🔹 Extracting features...")

# simple ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, cls)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset folder missing: {folder}")

    for img_name in os.listdir(folder):
        if img_name.lower().endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(folder, img_name)

            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            img = image.img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            feat = feature_extractor.predict(img, verbose=0)
            X.append(feat[0])
            y.append(label)

            # generate a few augmented variants per image
            aug_iter = datagen.flow(img, batch_size=1)
            for k in range(AUG_PER_IMAGE):
                aug_img = next(aug_iter)
                aug_feat = feature_extractor.predict(aug_img, verbose=0)
                X.append(aug_feat[0])
                y.append(label)

X = np.array(X)
y = np.array(y)

print("✅ Feature extraction done")
print("Samples after augmentation:", X.shape[0])

# ===============================
# SCALE FEATURES
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# HYPERPARAMETER TUNING (GridSearchCV)
# ===============================
print("🔹 Starting GridSearchCV for SVM (this may take a while)...")
svm = SVC(probability=True, class_weight='balanced')
grid = GridSearchCV(svm, GRID_PARAMS, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_scaled, y)
svm = grid.best_estimator_
print("✅ Grid search done. Best params:", grid.best_params_)

# ===============================
# SAVE MODELS (CRITICAL PART)
# ===============================
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete")
print("✅ svm_model.pkl and scaler.pkl saved successfully")
