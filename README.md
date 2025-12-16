# Customer Churn Prediction Using Machine Learning & Deep Learning

## 📌 Project Overview

This repository implements a **customer churn prediction system** for a banking dataset using a **comparative machine learning approach**. Multiple models were trained and evaluated, and **XGBoost was selected as the final production model**, while an **Artificial Neural Network (ANN)** was implemented as a benchmark to demonstrate deep learning understanding.

The project emphasizes **model comparison, data-driven model selection, interpretability, and deployment readiness**, rather than focusing on accuracy alone.

---

A **Streamlit web app** is provided for interactive use — allowing users to input customer details and receive a real-time churn risk assessment.

---

## 🧠 Problem Statement

Customer churn significantly impacts business revenue. The objective of this project is to **predict whether a customer is likely to leave the bank**, enabling proactive retention strategies.

### Key Challenges

* Imbalanced dataset (churn vs non-churn)
* Need to correctly identify churned customers (Recall-focused)
* Requirement for interpretable models in business settings

---

## 📂 Project Structure

├── Model_Selection.ipynb # Model Selection notebook

├── Churn_Modelling_Training.ipynb # Model training notebook

├── train.csv # Dataset used for training

├── pipeline.pkl # Preprocessing and Model Pipeline

├── app.py # Streamlit web application

├── requirements.txt # Project dependencies

└── README.md # Project documentation


---

## 🧩 Model Development Workflow

### 1️⃣ Data Preprocessing

* Removed non-informative features (Customer ID, Surname)
* Encoded categorical variables (Gender, Geography)
* Scaled numerical features
* Handled class imbalance using **SMOTE**
* Saved preprocessing pipeline (`preprocessed.pkl`) for consistent inference

---

### 2️⃣ Model Training & Selection

All experiments and comparisons are documented in **`Model_Selection.ipynb`**.

#### Models Evaluated

* **XGBoost (Final Model)**
* Random Forest
* Artificial Neural Network (ANN – TensorFlow/Keras)

All models achieved similar accuracy (~86%), indicating a **performance ceiling driven by feature separability**. Therefore, model selection was based on **robustness, interpretability, and suitability for tabular data**, not accuracy alone.

---

### ✅ Final Model Choice: XGBoost

XGBoost was selected because it:

* Performs exceptionally well on structured/tabular data
* Captures non-linear feature interactions efficiently
* Requires less data and tuning than ANN
* Provides strong **feature importance and explainability**

The ANN model is retained as a **benchmark**, implemented in `Churn_Modelling_Training.ipynb`, to compare deep learning against classical ML approaches.

---


## 📊 Model Evaluation Metrics

Given the imbalanced nature of churn prediction, models were evaluated using:

* Accuracy 
* Recall 
* F1-score

Accuracy was **not used as the sole decision metric**.

---

## 🔍 Model Explainability (SHAP)

To improve transparency and business trust, **SHAP (SHapley Additive exPlanations)** was used with the XGBoost model to:

* Identify the most influential features driving churn
* Explain individual predictions
* Support data-driven business decisions

This step highlights the importance of **interpretability in real-world ML systems**.

---

## 🌐 Deployment

* Developed an interactive **Streamlit web application** (`app.py`)
* Allows users to input customer details and receive real-time churn probability
* Integrated trained XGBoost model and saved preprocessing pipeline for inference

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mridul0010/Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning.git
cd Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning
```


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. **Project Link**
```bash
https://customer-churn-prediction-using-ann-6uasmkhzxz4ht3xu372ytx.streamlit.app/](https://customer-churn-prediction-using-machine-learning-deep-learning.streamlit.app/
```

---

## 🧠 Example Usage

1. Input customer details in the sidebar (e.g., Age, Geography, Tenure, etc.)

2. Click “🚀 Predict Churn Risk”

3. The model returns:
  - Churn probability (in %)
  - Risk level (Low / Moderate / High)
  - Data-driven recommendations for action

---

## 🛠 Tech Stack

* **Programming:** Python
* **Machine Learning:** XGBoost, Random Forest, Scikit-learn
* **Deep Learning:** TensorFlow, Keras
* **Explainability:** SHAP
* **Data Processing:** Pandas, NumPy
* **Deployment:** Streamlit

---

## 📊 Screenshots

<img width="1848" height="998" alt="image" src="https://github.com/user-attachments/assets/add4d33f-75fb-4cef-a4fc-8680218178aa" />

<img width="1908" height="1075" alt="image" src="https://github.com/user-attachments/assets/14ff7043-e82d-4c5d-ae70-cc33f2d7a08d" />

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/400de439-eec3-4f78-a84f-e50f7ad884fe" />

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/519a4be4-8101-41f2-b2dd-1124600d9eb1" />


--- 

## 🚀 Key Takeaways

* Demonstrates **comparative model evaluation and informed model selection**
* Highlights handling of **imbalanced classification problems**
* Balances **ML performance, interpretability, and engineering best practices**
* Shows when deep learning is *not* the optimal solution for tabular data

---

## 🧩 Future Improvements

- Implement model explainability dashboard.

- Extend dataset with behavioral transaction data.

- Deploy app on Streamlit Cloud or AWS EC2.

---

## 👩‍💻 Author

Mridul Lata

📍 Jaipur, India

💼 Aspiring Data Scientist / ML Engineer

🔗 www.linkedin.com/in/mridullata

🔗 https://github.com/mridul0010/Customer-Churn-Prediction-Using-ANN

---

📌 *This project focuses on real-world ML decision-making, not blind accuracy optimization.*

---

 ⭐ If you found this helpful, please give the repository a star and share your feedback!

---
