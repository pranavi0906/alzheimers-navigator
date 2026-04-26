# 🧠 Alzheimer's Navigator (ML Project)

## 📌 Inspiration

This project was developed to address the challenge of early detection of Alzheimer's disease. Early diagnosis is difficult but crucial for effective treatment and care.

By leveraging **ML and MRI brain images**, this system aims to assist in identifying different stages of Alzheimer's in a simple and accessible way.

---

## 📌 Overview

A Machine Learning and Deep Learning based system that analyzes **MRI brain images** to predict the stage of Alzheimer's disease.

The system uses pre-trained deep learning models for feature extraction and applies multiple machine learning algorithms with ensemble techniques to improve prediction accuracy.

---

## 🚀 Key Features

### 🧠 AI-Based Detection
- Predicts Alzheimer's stage using MRI brain images  
- Supports multi-class classification  

### 🤖 Model Architecture
- Feature extraction using **MobileNet** and **ResNet**  
- Classification using:
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Gradient Boosting  

### 🔗 Ensemble Learning
- Combines multiple models to improve prediction performance  
- Enhances accuracy and robustness  

### 🖥️ User Interface
- Interactive UI built using **Streamlit**  
- Users can upload MRI images and get predictions  

---

## 📊 Dataset

- Uses **MRI brain images** for Alzheimer's detection  
- Includes multiple classes:
  - Non-Demented  
  - Very Mild Demented  
  - Mild Demented  
  - Moderate Demented  
- Data preprocessing and feature extraction applied before model training  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- NumPy, Pandas  
- Streamlit  

---

## 📂 Project Structure

- `app.py` → Main application  
- `models/` → Contains class_names.json  
- `notebooks/` → Feature extraction and experiments  
- `requirements.txt` → Dependencies  

---

## ⚙️ Installation

1. Clone repository:
  git clone https://github.com/pranavi0906/alzheimers-navigator.git
  cd alzheimers-navigator

2. Install dependencies:
   pip install -r requirements.txt
 
3. Run the application:
   streamlit run app.py

---

## ⚠️ Note

Model files (`.h5`, `.pkl`) are not included due to size constraints.  
They can be provided upon request.

---
