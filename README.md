# 🐶🐱 Dogs vs Cats Image Classifier using SVM

This project implements a **Support Vector Machine (SVM)** classifier to distinguish between images of **dogs** and **cats**. It uses `OpenCV` for image processing and `scikit-learn` for model training.

---

## 📁 Project Structure

.
├── svm_dog_cat_classifier.py # Main script to load data and train the SVM model

├── train/ # Folder containing cat and dog images

│ ├── cat.0.jpg

│ ├── dog.0.jpg

│ └── ...

├── test/ # (Optional) Folder for testing new images

└── README.md

yaml
Copy
Edit

---

## ✅ Requirements

Make sure the following Python libraries are installed:

```bash
pip install opencv-python numpy scikit-learn
🧠 What the Script Does
Loads image data from the train/ folder.

Preprocesses each image (resizing to 100x100).

Labels images:

Images starting with cat → label 0

Images starting with dog → label 1

Flattens the image data.

Trains a linear SVM classifier.

Prints class distribution and training status.

🚀 How to Run
Place your training images in the train/ folder with filenames like:

cat.0.jpg, cat.1.jpg, ...

dog.0.jpg, dog.1.jpg, ...

Ensure your Python file is named svm_dog_cat_classifier.py.

Run the script:

bash
Copy
Edit
python svm_dog_cat_classifier.py
🧪 Example Output
javascript
Copy
Edit
Loading training data...
Training data loaded.
Number of images for class 0 (cats): 12500
Number of images for class 1 (dogs): 12500
Training SVM classifier...
SVM classifier trained successfully.
📝 Notes
The model uses raw pixel values for classification, which is simple but not ideal for accuracy.

For better results, consider using feature extraction techniques like:

Histogram of Oriented Gradients (HOG)

Pre-trained CNN features (e.g., from VGG16, ResNet)

📌 To-Do / Future Improvements
Add testing and prediction code.

Save and load the trained model (joblib or pickle).

Integrate GUI or web interface to upload and classify new images.

Improve accuracy with feature engineering or deep learning.

