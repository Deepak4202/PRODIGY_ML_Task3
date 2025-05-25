import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ----------------------------
# Load training data
# ----------------------------
def load_data(data_dir, image_size=(64, 64)):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        if file.startswith('cat'):
            label = 0
        elif file.startswith('dog'):
            label = 1
        else:
            continue
        path = os.path.join(data_dir, file)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, image_size)
            image = image / 255.0
            images.append(image.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

# ----------------------------
# Load test data
# ----------------------------
def load_test_images(test_dir, image_size=(64, 64)):
    test_images = []
    image_names = []
    for image_file in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, image_size)
            image = image / 255.0
            test_images.append(image.flatten())
            image_names.append(image_file)
    return np.array(test_images), image_names

# ----------------------------
# Paths
# ----------------------------
train_dir = "train1"
test_dir = "test1"

# ----------------------------
# Load and preprocess training data
# ----------------------------
print("📦 Loading training data...")
X, y = load_data(train_dir, image_size=(64, 64))

# Optional: use subset to speed up
subset_size = 5000
X = X[:subset_size]
y = y[:subset_size]

print(f"✅ Loaded {len(X)} images. 🐱 Cats: {np.sum(y==0)}, 🐶 Dogs: {np.sum(y==1)}")

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Train SVM
# ----------------------------
print("⚙️ Training SVM model...")
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
print("✅ SVM model trained.")

# ----------------------------
# Evaluate
# ----------------------------
accuracy = svm_classifier.score(X_val, y_val)
print(f"📊 Validation Accuracy: {accuracy * 100:.2f}%")

# ----------------------------
# Load and predict on test data
# ----------------------------
print("📦 Loading test data...")
X_test, test_image_names = load_test_images(test_dir, image_size=(64, 64))
print(f"✅ Loaded {len(X_test)} test images.")

print("🔍 Predicting test images...")
y_pred = svm_classifier.predict(X_test)

# ----------------------------
# Show first few predictions
# ----------------------------
print("\n🖼️ Sample Predictions:")
for name, pred in zip(test_image_names[:10], y_pred[:10]):
    label = "Dog 🐶" if pred == 1 else "Cat 🐱"
    print(f"{name}: {label}")

# ----------------------------
# Save to CSV
# ----------------------------
df = pd.DataFrame({
    'id': [int(name.split('.')[0]) for name in test_image_names],
    'label': y_pred
})
df = df.sort_values(by='id')
df.to_csv("submission.csv", index=False)
print("\n📝 Saved predictions to submission.csv")
