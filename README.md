# 🩸 HematoVision: Advanced Blood Cell Classification Using Transfer Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-11557c?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white">
</p>

---

## 📌 Category
Artificial Intelligence | Deep Learning | Healthcare AI  

---

## 📖 Project Overview

HematoVision is an AI-powered web application that classifies microscopic blood cell images using **Transfer Learning**.

The system is trained on **12,000 annotated blood cell images** categorized into:

- 🔹 Eosinophils  
- 🔹 Lymphocytes  
- 🔹 Monocytes  
- 🔹 Neutrophils  

Using a pretrained CNN (MobileNetV2), the model achieves high accuracy while reducing training time and computational cost.

This tool provides a reliable and scalable solution for automated blood cell classification in healthcare environments.

---

## 🎯 Objectives

- Develop an automated blood cell classification system  
- Improve diagnostic speed and accuracy  
- Reduce manual workload in pathology labs  
- Provide scalable AI-based healthcare support  

---

## 🧠 What is Transfer Learning?

Transfer learning is a deep learning technique where a pretrained model is reused for a related task.

Instead of training from scratch, we fine-tune **MobileNetV2** to classify blood cells.

### ✅ Benefits
- Faster Training  
- Improved Accuracy  
- Reduced Data Requirement  
- Better Generalization  

---

## 🔬 Methodology

### 1️⃣ Data Collection
Dataset sourced from Kaggle:  
https://www.kaggle.com/datasets/paultimothymooney/blood-cells  

### 2️⃣ Data Preprocessing
- Image resizing (224x224)
- Pixel normalization (0–1 scaling)
- Train-validation split
- Data augmentation

### 3️⃣ Model Architecture
- Pretrained CNN (MobileNetV2)
- Frozen base layers
- Custom dense layers
- Softmax output layer (4 classes)

### 4️⃣ Training Configuration
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Early Stopping implemented  
- Validation Accuracy: **~94.5%**

### 5️⃣ Deployment
The trained model is deployed using **Flask Web Framework**.

Users can:
- Upload a blood cell image
- Get predicted class
- View prediction results instantly

---

## 🌐 Web Application Features

- Image upload (JPG, PNG, JPEG)
- Real-time prediction
- Clean glass-morphism UI
- TensorFlow model integration
- Responsive design

---

## 📊 Results

- Validation Accuracy: ~94.5%
- Strong classification performance
- Reliable real-time predictions
- Minor confusion between visually similar cell types

---

## 📂 Project Structure

```
HematoVision/
│
├── static/
│   ├── style.css
│   ├── result.css
│   └── uploads/
│
├── templates/
│   ├── home.html
│   └── result.html
│
├── Blood Cell.h5
├── app.py
├── requirements.txt
└── README.md
```

### 📁 Folder Description

- **static/** → CSS files & uploaded images  
- **templates/** → HTML templates for Flask  
- **Blood Cell.h5** → Trained deep learning model  
- **app.py** → Flask backend application  
- **requirements.txt** → Python dependencies  
- **README.md** → Project documentation  

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/HematoVision.git
cd HematoVision
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 🏥 Real-World Applications

### 🔹 Automated Diagnostic Systems
- Real-time blood sample analysis  
- Faster workflow  
- Reduced human error  

### 🔹 Remote Medical Consultations
- Telemedicine integration  
- Remote blood image classification  
- Improved rural healthcare access  

### 🔹 Medical Education
- AI-powered learning tools  
- Instant classification feedback  
- Practical diagnostic skill development  

---

## 🚀 Future Enhancements

- Fine-tuning deeper CNN layers  
- Ensemble learning  
- Docker containerization  
- Cloud deployment (AWS / Azure)  
- Mobile application version  
- Hospital database integration  

---

## 👥 Team Details

**Team ID:** LTVIP2026TMIDS46367  

- Team Leader: Papadesu Gayatri  
- Member: Gavini Hari Vikas  
- Member: Pothu Kishore  
- Member: K Anikshema  

---

## 📜 License

This project is developed for academic and research purposes only.

---

## ⭐ Acknowledgements

- Kaggle Blood Cell Dataset  
- TensorFlow & Keras Documentation  
- Flask Framework  

---

⭐ If you found this project helpful, consider giving it a star!