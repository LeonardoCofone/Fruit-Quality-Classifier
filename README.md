# Fruits and Vegetables Quality Classifier  
## MobileNetV2 + Transfer Learning on a 5 GB Custom Image Dataset  

---

## 1 · Introduction

This repository contains the complete code and documentation for an automatic classification system that determines the **fresh / rotten** status of 28 different types of fruits and vegetables.  
The project was entirely developed by Leonardo Cofone: from data collection and cleaning to training a deep learning model and creating a ready-to-use inference script.

---

## 2 · Dataset

| Feature       | Value                                       |
|---------------|---------------------------------------------|
| Origin        | Proprietary data collection and annotation  |
| Size          | ≈ 5 GB total (images in `.jpg` format)      |
| Total classes | 28 (14 products × 2 conditions: `fresh`, `rotten`) |
| Resolution    | Variable; resized to **128 × 128** during preprocessing |

The dataset is organized in folders as follows:  
Unified_Dataset/  
└── <fruit_name>/  
 ├── fresh/  
 │ └── *.jpg  
 └── rotten/  
  └── *.jpg

> **Note** The dataset is not uploaded on GitHub due to size constraints; it is available as `kaggle/input/fruitquality1/Unified_Dataset` or upon request.
> **Watch my kaggle profile to watch better the dataset**

---

## 3 · Main Dependencies

| Package                | Tested Version |
|------------------------|----------------|
| Python                 | 3.11           |
| TensorFlow / Keras     | 2.18.0         |
| scikit-learn           | latest         |
| imbalanced-learn       | latest         |
| Pillow                 | latest         |
| seaborn / matplotlib   | latest         |  
| numpy                  | 1.26.4         |
| kivy                   | 2.3.1          |
| joblib                 | 1.4.2          |
| pywin32                | latest         |

Quick install:

```bash
pip install -r requirements.txt
```

## 4 · Project Structure and Usage
**main.py**  
Main script that launches a Graphical User Interface (GUI) built with Kivy.
The GUI allows users to perform real-time classification of fruit and vegetable images using the trained TensorFlow Lite model (FruitQuality.tflite).

**fruits-and-vegetables-quality.ipynb**  
Jupyter notebook used for exploratory data analysis, model training, and evaluation.

**FruitQuality.tflite**  
Optimized TensorFlow Lite model for lightweight, fast inference.

**label_encoder.pkl**  
Label encoder for mapping predicted class indices to human-readable class names.

**thresholds.pkl**  
Contains classification thresholds used to improve decision accuracy.

**requirements.txt**  
List of required Python packages and dependencies.

**README.md and LICENSE.txt**  
Project documentation and license information.

