# Loan Default Prediction and Risk Analysis

This project uses **logistic regression** to assess loan default risk using borrower demographics, financial attributes, and loan characteristics. 

Beyond model development, the analysis emphasizes **business-aligned evaluation techniques** used in credit risk modeling.

The goal is to showcase a realistic, **end-to-end approach** to binary classification on an imbalanced financial dataset.


## 🎯 Project Objectives

The project aims to:

- Understand key risk drivers through **exploratory data analysis (EDA)**
- Address **class imbalance** in loan default outcomes
- Evaluate the model using appropriate metrics for imbalanced classification
- Interpret model coefficients to extract business-relevant insights
- Incorporate **cost-sensitive decision analysis** to align predictions with business risk


## 📂 Dataset
- **Source:** https://www.kaggle.com/datasets/nikhil1e9/loan-default/data
- **Target Variable:** `Default`  
  - `1` → Loan defaults  
  - `0` → Loan does not default
- **Observations:** 255,347 loans
  - The dataset exhibits **significant class imbalance**, with defaults representing a small fraction of observations.

### Features:
<img width="893" height="533" alt="Screenshot 2026-01-16 at 7 09 42 PM" src="https://github.com/user-attachments/assets/85c84248-ab7b-4621-92f0-0d47dc0744fd" />


## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyze distributions of numerical features to identify patterns
- Examine default rates across categorical variables to uncover risk factors
- Assess feature correlations to detect and address multicollinearity
- Identify borrower segments associated with higher default risk

### 2. Feature Engineering & Preprocessing
- Remove non-informative identifiers to focus on predictive features
- Engineer risk-relevant features (e.g., loan-to-income ratios)
- Apply one-hot encoding to transform categorical variables
- Implement feature scaling via `StandardScaler` within a **scikit-learn pipeline** to prevent data leakage

### 3. Model Training
- Build logistic regression model using a **pipeline-based workflow** for reproducibility
- Address class imbalance with `class_weight='balanced'`
- Evaluate model generalization through train-test split validation


## Model Evaluation

Given the imbalanced nature of the dataset, model performance is assessed using:

- **Recall** and **F1-score** for default detection
- **ROC-AUC** to measure ranking performance
- **Confusion Matrix** for detailed error analysis
- **Cross-validation** to ensure robustness

### Decision Threshold Tuning
Rather than relying on the default 0.5 probability threshold, multiple thresholds are evaluated to analyze trade-offs between **false positives** and **false negatives**.

### Cost-Sensitive Evaluation
A hypothetical cost framework is introduced to reflect real-world lending considerations:

- **False Negatives (missed defaults)** are assigned a higher cost than  
- **False Positives (incorrectly flagged borrowers)**

Total estimated cost is computed across different thresholds to identify a **balanced** risk–cost trade-off.


## 🔍 Model Interpretability
One advantage of logistic regression is interpretability.

- Model coefficients are analyzed to understand feature influence on default risk
- Positive coefficients → higher default likelihood  
- Negative coefficients → lower default likelihood

Feature importance is visualized to highlight the most impactful predictors.


## 📌 Key Findings
- The model achieves a ROC-AUC of **~0.76**, indicating moderate discriminatory power
- Lower thresholds reduce missed defaults but significantly increase false positives
- An intermediate threshold provides the best balance between financial risk and error volume
- Cost-sensitive evaluation demonstrates why business objectives matter beyond accuracy


## ⚠️ Model Limitations
- As a linear classifier, the model may underperform when non-linear feature interactions are present
- Cost assumptions are illustrative and may vary across institutions
- While cost-sensitive evaluation is performed, the model is **not explicitly trained** to minimize financial loss
- Results depend on the quality and representativeness of the dataset


## 🧰 Technologies Used
- **Python**
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn**


## ✅ Conclusion

This project follows a **practical** and **interpretable** approach to loan default prediction. It demonstrates **professional** credit risk modeling through comprehensive EDA, 
appropriate evaluation metrics, decision threshold optimization, and cost-sensitive analysis. The notebook provides a strong foundation for further model development.

