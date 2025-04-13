# python_project
# 🧠 Kidney Disease Detection using Machine Learning

This project involves analyzing clinical data and implementing supervised machine learning algorithms to accurately predict the presence of **Chronic Kidney Disease (CKD)**. It includes data preprocessing, exploratory data analysis (EDA), and building predictive models like Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

---

## 📁 Project Structure
---

## 🎯 Objective

- Clean and preprocess clinical kidney disease data
- Perform exploratory data analysis and identify key indicators of CKD
- Build predictive models for early diagnosis of CKD
- Compare performance of Logistic Regression, SVM, and KNN classifiers

---

## 📊 Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
- Format: CSV
- Features: 24 clinical and physiological parameters
- Target: `classification` column (`ckd` or `notckd`)

---

## 🔧 Tools & Libraries Used

- **Python**
- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – Machine Learning models & preprocessing

---

## 📈 Models & Results

| Model               | Accuracy   |
|--------------------|------------|
| Logistic Regression | 98.86%     |
| Support Vector Machine (SVM) | 98.86%     |
| K-Nearest Neighbors (KNN - k=22) | 98.86%     |

> All models performed remarkably well, highlighting the strength of clinical indicators in CKD prediction.

---

## 🔍 Exploratory Data Analysis (EDA)

- **Visualizations** include histograms, boxplots, correlation heatmaps, and count plots.
- Key medical features analyzed:
  - **Hemoglobin**, **Serum Creatinine**, **Blood Urea**
  - **Packed Cell Volume**, **Specific Gravity**
  - **Appetite**, **Pus Cell Clumps**, **Bacteria**

---

