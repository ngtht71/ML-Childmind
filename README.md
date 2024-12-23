# Machine Learning - ChildMind Kaggle Competition  

**Class:** INT3405E 55  
**Group:** KTT Team  

**Members:**  
- Nguyễn Đàm Kiên - 22028226  
- Lê Sĩ Toàn - 22028318  
- Nguyễn Thanh Trà - 22028252  

---

# README: Machine Learning Solution Using Ensemble Models  

## Overview  
This project addresses a machine learning problem focused on predicting the **SII index** (a measure of Internet usability issues) using a rich dataset comprising time-series data and features related to personal information, health, and behavior.  
The primary goal is to optimize predictions by leveraging an ensemble model to achieve the highest **Quadratic Weighted Kappa (QWK)** score.  

---
## Model Improvements
**Model improvements are shown in the notebook and annotated with [IMPROVEMENT] blocks.**

---

## Problem Description  
The task involves predicting the SII index based on time-series data and other related features.  
The model’s effectiveness is evaluated using the **QWK metric**, which measures agreement between predicted and actual categories.  

---

## Solution Workflow  

### 1. Data Preprocessing  

#### 1.1 Data Loading  
Data is loaded from the following sources:  
- `train.csv` and `test.csv` (primary datasets).  
- Time-series data from `series_train.parquet` and `series_test.parquet` using multi-threading for efficiency.  

Important statistics like mean, variance, maximum, and minimum are extracted from time-series data and transformed into features for modeling.  

#### 1.2 Data Transformation  

**Categorical Features:**  
- Identified and encoded as integers.  
- Mapped consistently between training and testing datasets to ensure uniformity.  

**Numerical Features:**  
- Processed to ensure no missing values in both train and test datasets.  

#### 1.3 Data Cleaning  

**Handling Missing Data:**  
- Numerical features: Missing values are filled with column means.  
- Categorical features: Missing values are replaced with `"Missing"` and encoded as integers.  

**Feature Filtering:**  
- Irrelevant features (e.g., `id`) are dropped.  
- Rows with excessive missing values or invalid data are removed.  

This rigorous preprocessing ensures the dataset is clean, complete, and ready for modeling.  

---

### 2. Modeling  

#### 2.1 Model Selection  
An ensemble of three advanced regression models was used:  
- **LightGBM**: Known for speed and efficiency with large datasets.  
- **XGBoost**: Robust for structured data and large-scale problems.  
- **CatBoost**: Excels in handling categorical features without extensive preprocessing.  

#### 2.2 Ensemble Methodology  
Predictions from the three models were combined using **Voting Regressor** with a weighted average to enhance accuracy.  

#### 2.3 Cross-Validation  
**Stratified K-Fold Cross-Validation** with 5 folds was applied:  
- Ensures balanced class distribution across training and validation sets.  
- Prevents overfitting and provides comprehensive performance evaluation.  
- Each model’s performance was measured using the QWK metric.  

#### 2.4 Threshold Optimization  
**Nelder-Mead optimization** was employed to determine the best thresholds for mapping continuous predictions to discrete categories (e.g., None, Mild, Moderate, Severe).  

---

### 3. Evaluation  

#### 3.1 Performance Metrics  
Out-of-Fold (OOF) predictions were used to assess model performance.  

**Evaluation tools included:**  
- **Confusion Matrix**: Highlights model accuracy and common misclassification patterns.  
- **Classification Report**: Provides precision, recall, and F1-score for each class.  

**Results:**  
- Total misclassifications: 1,186 out of 2,736 samples.  
- Model achieved approximately **56.65% accuracy** with notable performance improvements on class 0 and class 1.  

---

### 4. Advantages of Approach  
- **Accuracy Improvements:** Combining multiple models compensates for individual weaknesses.  
- **Robustness:** Ensemble reduces the risk of overfitting by averaging predictions.  
- **Flexibility:** Handles heterogeneous data types (e.g., numerical and categorical).  
- **Scalability:** The ensemble design allows for the addition or replacement of models.  

---

### 5. Limitations and Future Work  

**Limitations:**  
- **Resource Intensive:** Training multiple models requires significant computational power.  
- **Complex Maintenance:** Managing and updating three models increases complexity.  
- **Common Errors:** Shared weaknesses among base models can impact ensemble performance.  

**Future Directions:**  
- **Optimization:** Implement model pruning or leverage cloud-based systems for efficient training.  
- **Deep Learning Integration:** Explore neural networks for capturing complex patterns.  
- **AutoML:** Use automated hyperparameter tuning for better performance.  
- **Simplification:** Consider reducing ensemble complexity with stacking or boosting techniques.  
- **Explainability:** Develop tools for interpreting model decisions.  

---

### 6. Conclusion  
This project demonstrates the power of ensemble modeling in solving complex machine learning problems.  
The use of **LightGBM**, **XGBoost**, and **CatBoost**, combined with **Voting Regressor** and advanced optimization techniques, ensures high prediction accuracy and robust performance.  
Future enhancements will focus on optimizing computational resources and incorporating deep learning models for further improvements.  

---

## Key Technologies and Libraries  
- **LightGBM**, **XGBoost**, **CatBoost**  
- **scikit-learn:** Cross-validation, Voting Regressor  
- **scipy.optimize:** Threshold optimization  
- **pandas**, **numpy:** Data preprocessing  
- **matplotlib**, **seaborn:** Visualization  
