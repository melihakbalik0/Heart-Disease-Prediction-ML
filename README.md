# Heart Disease Prediction & Clinical Analysis 🫀

### Author: Melih Talha Akbalık
### Objective
This project aims to predict the risk of heart disease using the **CDC 2015 BRFSS** dataset. By leveraging Machine Learning, we compare a baseline **Logistic Regression** model against a more robust **Random Forest** classifier to maximize recall and clinical reliability.

---

## 📊 Project Overview
Heart disease is a leading cause of mortality worldwide. Early detection is critical for survival. This project implements an end-to-end Machine Learning pipeline to identify high-risk individuals based on 21 health indicators.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
* **Environment:** Google Colab / PyCharm

---

## 🔍 Data Quality & Integrity

Dataset[https://ieee-dataport.org/documents/heart-disease-dataset]
Before modeling, a rigorous data audit was performed. The dataset consists of **253,680 records**. As shown in our analysis, the dataset maintains a **100% Fill Rate** across all features, ensuring high reliability for statistical modeling.

| Metric | Value |
| :--- | :--- |
| **Total Records** | 253,680 |
| **Features** | 21 |
| **Missing Values** | 0 (Clean) |

---

## 📈 Model Performance Benchmark
We prioritized **Recall (Sensitivity)** and **ROC-AUC** to ensure that potential heart disease cases are not missed (minimizing False Negatives).

| Performance Metric | Logistic Regression | Random Forest (Winner) |
| :--- | :--- | :--- |
| **Accuracy** | 0.7490 | **0.7638** |
| **Recall (Sensitivity)** | 0.7638 | **0.7932** |
| **ROC-AUC** | 0.8251 | **0.8385** |
| **F1-Score** | 0.3855 | **0.4022** |

> **Note:** Although Precision is lower (~0.25), this is a deliberate trade-off to achieve high Recall, which is vital in clinical screening to avoid missing at-risk patients.

---

## 💡 Key Decision Factors (XAI)
Using **Random Forest Feature Importance**, we identified the top drivers for heart disease prediction:
1. **GenHlth** (General Health perception)
2. **Age** 3. **HighBP** (High Blood Pressure)
4. **BMI** (Body Mass Index)

---

## 📂 Repository Structure
* `notebooks/`: Contains the full exploratory data analysis (EDA) and model experiments.
* `data/`: Placeholder for the CDC dataset.
* `src/`: (Coming Soon) Modularized Python scripts for production environments.
