import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths & Parameters
DATA_PATH = r"E:\Simu Drive -AI\Data sets\GTSRB\Final_Training\Images"
IMG_SIZE = 48
NUM_CLASSES = 43

images = []
labels = []

# Read each class folder
for class_id in range(NUM_CLASSES):
    folder_path = os.path.join(DATA_PATH, f"{class_id:05d}")
    csv_path = os.path.join(folder_path, f"GT-{class_id:05d}.csv")
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(';')
            img_path = os.path.join(folder_path, parts[0])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(int(parts[-1]))

# Normalize and encode
X = np.array(images, dtype="float32") / 255.0
y = to_categorical(labels, NUM_CLASSES)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save model
model.save("models/traffic_sign_model.h5")
print("✅ Model saved to models/traffic_sign_model.h5")
