# HematoVision: Advanced Blood Cell Classification Using Transfer Learning

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-11557c.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-55d0ff.svg?logo=TensorFlow&logoColor=white"></a>
<a href="#"><img alt="Keras" src="https://img.shields.io/badge/Keras-7ce8ff.svg?logo=Keras&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-00acdf.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-0080ff.svg?logo=opencv&logoColor=white"></a>
<a href="#"><img alt="Flask" src="https://img.shields.io/badge/Flask-000000.svg?logo=flask&logoColor=white"></a>

---

## 📌 Category
Artificial Intelligence  

## 🛠 Skills Required
Python, Deep Learning, Transfer Learning

---

## 📖 Project Description

HematoVision aims to develop an accurate and efficient model for classifying blood cells using Transfer Learning techniques. The project utilizes a dataset of 12,000 annotated blood cell images categorized into four classes:

- Eosinophils  
- Lymphocytes  
- Monocytes  
- Neutrophils  

The system leverages a pre-trained Convolutional Neural Network (CNN) to accelerate training and improve classification accuracy. Transfer learning allows the model to reuse learned image features, reducing computational cost and improving performance.

This AI-based solution provides a reliable and scalable tool for pathologists and healthcare professionals, ensuring precise and efficient blood cell classification.

---

## 🎯 Project Goals

- Develop an automated blood cell classification system.
- Improve diagnostic speed and accuracy.
- Reduce manual workload in pathology labs.
- Provide scalable AI-based healthcare support.

---

## 🧠 What is Transfer Learning?

Transfer learning is a deep learning technique where a model trained on a large dataset is reused for a related task. Instead of training a model from scratch, we use pre-trained models (like MobileNetV2) and fine-tune them for blood cell classification.

### ✅ Benefits
- Faster Training
- Improved Accuracy
- Reduced Data Requirement
- Better Generalization

---

## 🔬 Methodology

### 1️⃣ Data Acquisition
Dataset downloaded from Kaggle:
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

### 4️⃣ Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Early Stopping used
- Achieved ~94.5% validation accuracy

### 5️⃣ Model Deployment
The trained model is deployed using Flask Web Framework.

Users can:
- Upload blood cell image
- Get predicted class
- View prediction confidence percentage

---

## 🌐 Web Application Features

- Image upload (JPG, PNG, JPEG)
- Real-time prediction
- Confidence percentage display
- Clean web interface
- TensorFlow model integration

---

## 📊 Results

- Validation Accuracy: ~94.5%
- Strong classification performance
- Minor confusion between visually similar cell types
- Reliable real-time predictions

---

## 🏥 Real-World Applications

### 🔹 Scenario 1: Automated Diagnostic Systems
- Real-time blood sample analysis
- Faster diagnostic workflow
- Reduced human error

### 🔹 Scenario 2: Remote Medical Consultations
- Telemedicine integration
- Remote blood image classification
- Improved rural healthcare access

### 🔹 Scenario 3: Educational Medical Training
- AI-powered learning tools
- Instant classification feedback
- Practical skill development for students

---

## 🚀 Future Improvements

- Fine-tuning deeper CNN layers
- Ensemble learning techniques
- Docker-based deployment
- Mobile application development
- Integration with hospital databases

---

## 👥 Team Details

**Team ID:** LTVIP2026TMIDS46367  

- Team Leader: Papadesu Gayatri  
- Member: Gavini Hari Vikas  
- Member: Pothu Kishore  
- Member: K Anikshema  

---

## 📂 Project Structure
