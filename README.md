# Customer-Churn-Prediction-Using-ANN
Customer Churn Prediction using Deep Learning (ANN). Full workflow from EDA and feature analysis to model training and prediction.

# 🧠 Customer Churn Prediction using Deep Learning (ANN)

### 🚀 Project Overview
This project predicts **customer churn probability** for a bank using a **Deep Learning Artificial Neural Network (ANN)**.  
The model analyzes customer demographics, account details, and financial behavior to assess the risk of churn, empowering businesses to proactively engage and retain at-risk clients.

A **Streamlit web app** is provided for interactive use — allowing users to input customer details and receive a real-time churn risk assessment.

---

## 📂 Project Structure
├── 1_Analytics.ipynb # Data Analysis notebook

├── 2_Churn_Modelling_Training.ipynb # Model training notebook

├── 3_Churn_Modelling_Predict.ipynb # Model prediction notebook

├── Churn_Modelling.csv # Dataset used for training

├── model.keras # Trained ANN model (TensorFlow/Keras)

├── preprocessed.pkl # Preprocessing pipeline (ColumnTransformer)

├── app.py # Streamlit web application

├── requirements.txt # Project dependencies

└── README.md # Project documentation


---

## 🧩 Model Development Workflow

### 1. Data Preprocessing
- **Dataset:** `Churn_Modelling.csv`  
- Removed irrelevant columns (e.g., `RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical variables:
  - `Gender` → Label Encoding
  - `Geography` → One-Hot Encoding
- Normalized numerical features using **StandardScaler**
- Combined all transformations into a unified **ColumnTransformer**, saved as `preprocessed.pkl`

### 2. Model Architecture (ANN)
| Layer Type | Units | Activation | Description |
|-------------|--------|-------------|--------------|
| Input Layer | 12 | - | Preprocessed numerical + encoded categorical features |
| Dense | 16 | ReLU | First hidden layer |
| Dense | 8 | ReLU | Second hidden layer |
| Output | 1 | Sigmoid | Outputs churn probability (0–1) |

**Loss:** Binary Crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy

Achieved:
- ✅ **Validation Accuracy:** ~86%
- 📉 **Validation Loss:** ~0.33

---

## 💡 Key Insights
- **Older**, **inactive**, or **low-credit-score** customers have a higher churn risk.
- **Active members** with multiple products and good credit scores are less likely to leave.
- **Balance** and **Tenure** also influence customer loyalty patterns.

---

## 🧮 Streamlit App (`app.py`)

### Features:
- Interactive input fields for:
  - Demographics: `Age`, `Gender`, `Geography`
  - Account details: `CreditScore`, `Tenure`, `NumOfProducts`, etc.
  - Financials: `Balance`, `EstimatedSalary`
- Predicts churn probability in real-time using the trained ANN model.
- Displays:
  - **Churn Probability (%):** with confidence bar
  - **Risk Category:** Low / Moderate / High
  - **Strategic Recommendations**
  - **Mock Feature Importance**
  - **Model performance metrics & training history**

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mridul0010/customer-churn-prediction.git
cd customer-churn-prediction
```


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Open in browser
```bash
https://customer-churn-prediction-using-ann-6uasmkhzxz4ht3xu372ytx.streamlit.app/
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

## 🧾 Requirements
- tensorflow==2.20.0
- pandas
- numpy
- scikit-learn
- tensorboard
- matplotlib
- streamlit

---

## 📊 Screenshots

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/44185aec-d5d1-4011-86f6-8b30bacfc0f5" />

<img width="1895" height="1021" alt="image" src="https://github.com/user-attachments/assets/a61716de-2d75-43da-b942-d6ca6aa5af14" />

<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/9b1ab661-33e0-44f5-a98d-79831561cb68" />


--- 

## 🧩 Future Improvements

- Add SHAP-based feature importance visualization.

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

 ⭐ If you found this helpful, please give the repository a star and share your feedback!

---
