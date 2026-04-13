# Heart Disease Prediction and Clinical Analysis

**Author:** Melih Talha Akbalık  
**Field:** Software Engineering / Machine Learning Research

---

## Project Overview
This study focuses on predictive modeling of heart disease risks using the CDC 2015 BRFSS dataset. The core objective is to develop a robust machine learning pipeline capable of identifying high-risk individuals by evaluating 21 distinct health indicators. 

The project prioritizes clinical utility, specifically focusing on Recall and ROC-AUC metrics to ensure that the model acts as an effective screening tool that minimizes the risk of missing potential cases (False Negatives).

## Technical Environment
* **Language:** Python 3.x
* **Core Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Development Workflow:** Experimental analysis was conducted in Jupyter Notebooks, while the final modularized production code was developed using PyCharm.

---

## Dataset Analysis
The analysis is based on 253,680 individual records provided by the CDC. 
* **Data Integrity:** A comprehensive data audit confirmed a 100% fill rate across all features. 
* **Feature Engineering:** The dataset includes 21 features ranging from binary lifestyle indicators (Smoking, Physical Activity) to categorical clinical scales (Age Groups, General Health Status).

---

## Model Evaluation and Benchmarking
A comparative analysis was performed between Logistic Regression and Random Forest classifiers. The Random Forest model was selected for the final pipeline due to its superior handling of non-linear relationships and higher sensitivity.

| Metric | Logistic Regression | Random Forest (Selected) |
| :--- | :--- | :--- |
| **Accuracy** | 0.7490 | **0.7638** |
| **Recall (Sensitivity)** | 0.7638 | **0.7932** |
| **ROC-AUC Score** | 0.8251 | **0.8385** |

**Note on Clinical Trade-offs:**
The model is specifically tuned for high Recall (~0.79). In a medical context, maximizing the detection of at-risk patients is prioritized over Precision, as the cost of a missed diagnosis significantly outweighs the cost of further clinical validation for a false positive.

---

## Key Predictors (Feature Importance)
Based on Random Forest Gini Importance, the top four features driving the predictions are:
1. **General Health (GenHlth):** Self-perceived health status.
2. **Age:** Biological age group.
3. **High Blood Pressure (HighBP):** Clinical history of hypertension.
4. **BMI:** Body Mass Index.

---

## Repository Structure
* `data/`: Local storage for the CDC dataset.
* `Heart_Disease_Project.ipynb`: Exploratory Data Analysis (EDA) and model prototyping.
* `Machine_Learning_Project_Pycharm/`: Modularized implementation including the main execution script and source modules for data loading and visualization.
* `LICENSE`: Project licensing.

---

## Future Work
* Development of a Streamlit-based web interface for real-time risk estimation and user data input.
