#  Spam Email Detection using Machine Learning

An AI/ML project that detects spam emails using **TF-IDF text vectorization**
and **Multinomial Naive Bayes**, with a **Streamlit web interface** for real-time predictions.

---

##  Project Overview

Spam emails are a major cybersecurity and productivity issue.  
This project builds an **end-to-end spam detection system** that classifies
emails/messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP)
and Machine Learning.

The project covers:
- Data preprocessing
- Feature extraction using TF-IDF
- Model training with Naive Bayes
- Model evaluation
- Deployment using Streamlit

---

##  Features

-  Spam / Not Spam classification
-  Machine Learning pipeline
-  TF-IDF text vectorization
-  Fast and lightweight Naive Bayes model
-  Streamlit-based web application
-  Confidence score for predictions

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Pickle** (for model saving)

---



---

##  Dataset

- **Dataset Name:** SMS Spam Collection Dataset  
- **Type:** Text (SMS / Email-like messages)  
- **Total Records:** ~5,500 messages  
- **Language:** English  

###  Labels
- `ham` → Legitimate message (Not Spam)
- `spam` → Unwanted or fraudulent message (Spam)

###  Source
- Publicly available SMS spam dataset widely used for
  **Natural Language Processing (NLP)** and **Text Classification** tasks.
- The dataset contains real-world examples of both spam and non-spam messages.

###  Data Preprocessing Steps
To ensure clean and reliable input for the machine learning model, the following preprocessing steps are applied:

1. **Column Selection**
   - Removed unnecessary and empty columns.
   - Retained only:
     - `label` (spam / ham)
     - `text` (message content)

2. **Text Normalization**
   - Converted all text to lowercase to avoid case sensitivity issues.
   - Ensures words like *Free* and *free* are treated the same.

3. **Label Encoding**
   - Converted categorical labels into numerical form:
     - `ham` → `0`
     - `spam` → `1`
   - Required for machine learning algorithms.

4. **Missing Value Handling**
   - Rows with missing or null values were removed.

These preprocessing steps help improve model accuracy and consistency.

---

## 🧠 Machine Learning Approach

This project follows a **standard NLP text classification pipeline**.

### 1️ Text Preprocessing
- Lowercasing all text
- Basic text cleaning
- Removal of irrelevant symbols during vectorization

### 2️ Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert text into numerical vectors.
- TF-IDF highlights important words while reducing the impact of commonly used words.
- Helps the model focus on words that are meaningful for spam detection.

### 3️ Model Selection
- **Multinomial Naive Bayes**
  - Lightweight and fast
  - Well-suited for text classification problems
  - Performs well on word-frequency–based features like TF-IDF

### 4️ Model Evaluation
The model is evaluated using standard classification metrics:

- **Accuracy** – Overall correctness of predictions
- **Precision** – How many predicted spam messages are actually spam
- **Recall** – How many actual spam messages are correctly detected
- **F1-score** – Balance between precision and recall

A confusion matrix is also used to analyze misclassifications.

### 5️ Deployment
- The trained model is saved as a **pipeline** using `pickle`.
- The pipeline is deployed using **Streamlit**, enabling:
  - Real-time spam prediction
  - Single message input
  - Bulk prediction via CSV file upload

---

## ▶️ How to Run the Project

Follow the steps below to run the project locally.

### 1️ Clone the repository
```bash
git clone https://github.com/Skarangayake/SPAM-EMAIL-DETECTION/tree/main
cd SPAM-EMAIL-DETECTION

