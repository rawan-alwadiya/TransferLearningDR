# **TransferLearningDR: EfficientNet-Based Diabetic Retinopathy Detector**

TransferLearningDR is a deep learning project that uses **transfer learning with EfficientNetB3** to classify retinal fundus images as **Diabetic Retinopathy (DR)** or **No DR**.  
It demonstrates a complete **end-to-end computer vision workflow** including **dataset preparation, augmentation, EfficientNet fine-tuning, evaluation, and deployment with Streamlit & Hugging Face**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-transferlearning-efficientnet-activity-7373417012628344832-e5pZ?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://transferlearningdr-httrdrrzqndfzqbyak8wkc.streamlit.app/)  
- 🤗 [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/TransferLearningDR)

![App Demo](https://github.com/rawan-alwadiya/TransferLearningDR/blob/main/TransferLearningDR_App.png)  
![No DR Example](https://github.com/rawan-alwadiya/TransferLearningDR/blob/main/No_DR.png)  
![DR Example](https://github.com/rawan-alwadiya/TransferLearningDR/blob/main/DR.png)

---

## **Project Overview**

The workflow includes:  
- **Dataset Preparation**: balanced re-splitting into train/validation/test sets  
- **Exploration & Visualization**: class distribution checks and image inspection  
- **Data Augmentation**: rotation, zoom, brightness adjustment, horizontal flipping  
- **Modeling (EfficientNetB3)**: pre-trained on ImageNet, fine-tuned for DR classification  
- **Evaluation**: accuracy, precision, recall, and F1-score metrics  
- **Deployment**: interactive **Streamlit web app** and **Hugging Face Spaces** for real-time predictions

---

## **Objective**

Develop and deploy a robust **transfer learning model** to detect Diabetic Retinopathy early, providing an accessible AI-powered screening tool to help prevent vision loss.

---

## **Dataset**

- **Source**: [Diagnosis of Diabetic Retinopathy (Kaggle)](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data)  
- **Classes**: Diabetic Retinopathy (DR), No DR  
- **Preprocessing**: images resized to `(224×224)` RGB, normalized, and balanced across splits

---

## **Project Workflow**

- **EDA & Visualization**: inspected sample images and verified balanced class distribution  
- **Preprocessing**:  
  - Resize to `(224×224)`  
  - RGB normalization `(0–1)`  
  - Balanced train/validation/test splits  
- **Augmentation**: random rotation, zoom, brightness adjustment, horizontal flip  
- **Modeling (Transfer Learning)**:  
  - Base model: **EfficientNetB3** with ImageNet weights  
  - Custom layers: BatchNormalization, Dense, Dropout, Sigmoid output  
- **Training Setup**:  
  - Optimizer: **Adamax**  
  - Loss: Binary Crossentropy  
  - Callbacks: EarlyStopping & ModelCheckpoint  
  - Epochs: 30 (with early stopping)

---

## **Performance Results**

**EfficientNetB3 Transfer Learning Classifier:**  
- **Accuracy**: `98.2%`  
- **Precision**: `98.6%`  
- **Recall**: `97.9%`  
- **F1-score**: `98.2%`

The model achieved **high precision and recall**, ensuring reliable DR detection while minimizing false positives.

---

## **Project Links**

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/transferlearningdr-diabetic-retinopathy-detection)  
- **Streamlit App**: [Try it Now](https://transferlearningdr-httrdrrzqndfzqbyak8wkc.streamlit.app/)  
- **Hugging Face Repo**: [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/TransferLearningDR)  

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- TensorFlow / Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit (Deployment)

**Techniques**:  
- Transfer Learning (EfficientNetB3 fine-tuning)  
- Data Augmentation (rotation, zoom, brightness, flipping)  
- EarlyStopping & ModelCheckpoint  
- Real-time deployment with **Streamlit** & **Hugging Face**
