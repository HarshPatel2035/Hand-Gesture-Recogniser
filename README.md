
# Hand Gesture Recognition using Deep Learning

## 📌 Overview
This project focuses on building a **hand gesture recognition system** using **deep learning** techniques. The system classifies hand gestures captured using the **Leap Motion Controller** into different categories.  

Hand gesture recognition has applications in:
- Human-computer interaction
- Virtual reality (VR)
- Gaming
- Robotics control
- Sign language interpretation

This project uses the **LeapGestRecog dataset** from Kaggle, which contains 10 different hand gestures performed by multiple users.

---

## 🚀 Project Objectives
- Preprocess the raw images for training and testing.
- Build a **Convolutional Neural Network (CNN)** model to classify gestures.
- Train and evaluate the model on the Kaggle dataset.
- Develop a scalable pipeline for real-world applications.

---

## 📂 Dataset
The dataset used in this project is **[LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)**.

- **Number of classes:** 10 different hand gestures  
- **Total images:** 20,000  
- **Image format:** `.png`  
- **Image size:** 120 × 320 pixels (preprocessed to a smaller size for model training)

The dataset is organized as:
```
leapGestRecog/
│
├── 00/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
│
├── 01/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
│
└── ...
```

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Other Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn

---

## 🧠 Model Architecture
The model is a **Convolutional Neural Network (CNN)** designed to extract spatial features from the images.  

**Layers used:**
1. Convolutional layers with ReLU activation  
2. MaxPooling layers to reduce dimensionality  
3. Dropout layers to prevent overfitting  
4. Fully connected dense layers  
5. Softmax output layer for classification  

---

## 📊 Results
The CNN achieved the following performance metrics:

| Metric        | Value |
|---------------|-------|
| **Training Accuracy** | ~95% |
| **Validation Accuracy** | ~93% |
| **Test Accuracy** | ~92% |

Visualizations of training performance are included in the notebook.

---

## 🔧 Installation & Usage
Follow the steps below to set up and run the project:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YourUsername/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset
- Download the **LeapGestRecog** dataset from Kaggle:  
  [LeapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- Extract it into the `data/` folder.

```
data/
└── leapGestRecog/
```

### 4️⃣ Run the notebook
Open the Jupyter notebook and run all cells:
```bash
jupyter notebook Hand\ Gestures.ipynb
```

---

## 📈 Future Improvements
- Deploy the model as a **real-time hand gesture recognition app** using OpenCV.
- Implement transfer learning for better accuracy.
- Add support for dynamic gestures (video-based recognition).

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the project, please fork the repository and create a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
