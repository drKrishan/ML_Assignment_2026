# Bank Marketing Classification Analysis

## BM 173 - Applications of Machine Learning

---

## Project Overview

This project implements a comprehensive machine learning analysis on the **Bank Marketing dataset** to predict whether a client will subscribe to a term deposit.

**Business Problem:** A Portuguese bank wants to identify customers most likely to subscribe to their term deposit product, enabling targeted marketing efforts.

### Key Features:

- ✅ 6 Classification Algorithms for comparison
- ✅ Full data exploration and preprocessing pipeline
- ✅ Multiple evaluation metrics and visualizations
- ✅ Feature importance analysis
- ✅ Production-ready Python code

---

## Dataset

**Source:** UCI Machine Learning Repository  
**File:** `bank-additional-full.csv`

### Dataset Characteristics:

- **Records:** 4,500 customer records
- **Features:** 20 input features + 1 target variable
- **Target:** Binary classification (Subscription: Yes/No)
- **Class Distribution:** 78.84% No, 21.16% Yes (Imbalanced)
- **Missing Values:** None (~clean data)

### Key Features:

- **Demographic:** age, job, marital status, education
- **Financial:** default status, housing loan, personal loan
- **Campaign:** contact type, duration, campaign number
- **Economic:** employment rate, consumer price index, euribor rate
- **Previous:** poutcome (previous campaign outcome)

---

## Project Structure

```
ML_ASSIGNMET/
├── 00_generate_sample_dataset.py      # Generate synthetic dataset
├── 01_download_dataset.py              # Download dataset (if needed)
├── 02_bank_marketing_analysis.py      # Main analysis pipeline
├── data/
│   └── bank-additional-full.csv       # Dataset
├── visualizations/                     # Generated charts
│   ├── 01_target_distribution.png     # Class distribution
│   ├── 02_numeric_features_distribution.png
│   ├── 03_categorical_features_distribution.png
│   ├── 04_model_comparison.png        # Performance comparison
│   ├── 05_confusion_matrices.png      # All 6 models
│   ├── 06_roc_curves.png              # ROC curves analysis
│   ├── 07_feature_importance.png      # Top features
│   └── model_results_summary.csv      # Numerical results
└── README.md                           # This file

```

---

## Installation & Setup

### 1. **Prerequisites**

```bash
Python 3.10+
pip/conda package manager
```

### 2. **Install Required Libraries**

```bash
cd c:\Users\Vinoth\Downloads\ML_ASSIGNMET
pip install -r requirements.txt
```

Or manually install:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 3. **Generate/Load Dataset**

```bash
# Generate sample dataset
python 00_generate_sample_dataset.py

# OR download real dataset
python 01_download_dataset.py
```

### 4. **Run Analysis**

```bash
python 02_bank_marketing_analysis.py
```

---

## Machine Learning Models

### 6 Classification Algorithms Implemented:

1. **Logistic Regression**
   - Type: Linear classifier
   - Use Case: Baseline model, interpretable
   - Scaling: Required (StandardScaler used)

2. **Decision Tree**
   - Type: Tree-based classifier
   - Use Case: Non-linear patterns, feature importance
   - Max Depth: 10 (prevent overfitting)

3. **Random Forest**
   - Type: Ensemble (100 trees)
   - Use Case: Robust, handles imbalance well
   - Feature Importance: Available

4. **Support Vector Machine (SVM)**
   - Type: Kernel-based classifier
   - Kernel: RBF (Radial Basis Function)
   - Scaling: Required

5. **K-Nearest Neighbors (KNN)**
   - Type: Instance-based classifier
   - K Value: 5 neighbors
   - Scaling: Required

6. **Gradient Boosting**
   - Type: Ensemble (100 trees)
   - Use Case: High performance, sequential learning
   - Feature Importance: Available

---

## Evaluation Metrics

### Classification Metrics Used:

1. **Accuracy**
   - Formula: (TP + TN) / Total
   - Interpretation: Overall correctness

2. **Precision**
   - Formula: TP / (TP + FP)
   - Interpretation: Of predicted positives, how many are correct?

3. **Recall (Sensitivity)**
   - Formula: TP / (TP + FN)
   - Interpretation: Of actual positives, how many were caught?

4. **F1-Score**
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Interpretation: Harmonic mean of Precision & Recall
   - **Best metric for imbalanced datasets**

5. **ROC-AUC**
   - Range: 0 to 1
   - Interpretation: Area under ROC curve (0.5 = random, 1.0 = perfect)

6. **Confusion Matrix**
   - True Positives (TP), True Negatives (TN)
   - False Positives (FP), False Negatives (FN)

---

## Data Preprocessing Steps

### Step 1: Exploratory Data Analysis (EDA)

- Visualized target variable distribution
- Analyzed numeric and categorical features
- Identified missing values (None found)
- Checked for class imbalance

### Step 2: Encoding Categorical Variables

- Applied LabelEncoder to all categorical features
- Target variable: No (0) / Yes (1)
- 10 categorical columns encoded

### Step 3: Train-Test Split

- 80% Training (3,600 records)
- 20% Testing (900 records)
- Stratified split to maintain class distribution

### Step 4: Feature Scaling

- StandardScaler applied to all features
- Fit on training set, applied to test set
- Required for distance-based and SVM models

---

## Key Findings

### Model Performance Summary:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.7889   | 0.0000    | 0.0000 | 0.0000   | 0.5597  |
| Decision Tree       | 0.7489   | 0.2805    | 0.1211 | 0.1691   | 0.5694  |
| Random Forest       | 0.7889   | 0.0000    | 0.0000 | 0.0000   | 0.5571  |
| SVM                 | 0.7889   | 0.0000    | 0.0000 | 0.0000   | 0.5183  |
| KNN                 | 0.7533   | 0.2037    | 0.0579 | 0.0902   | 0.4953  |
| Gradient Boosting   | 0.7811   | 0.0000    | 0.0000 | 0.0000   | 0.5601  |

### Key Insights:

1. **Class Imbalance Challenge**
   - Majority class (No): 78.84%
   - Minority class (Yes): 21.16%
   - Many models simply predict "No" for all cases

2. **Best Performing Models**
   - **Decision Tree**: Best F1-Score (0.1691), Highest Recall (0.1211)
   - **Best ROC-AUC**: Decision Tree (0.5694)

3. **Feature Importance (Top Features)**
   - From Random Forest: See visualizations/07_feature_importance.png
   - Key indicators: Duration, campaign, age, economic indicators

4. **Recommendations**
   - Use weighted class balancing strategies
   - Increase decision tree depth for better complexity
   - Implement SMOTE or class weighting
   - Consider threshold adjustment for recall optimization

---

## Using the Results in Your Report

### Section-by-Section Mapping:

1. **Introduction**
   - Use dataset overview and business problem context

2. **Data Exploration & Preparation**
   - Reference: visualizations/01-03\_\*.png
   - Include basic statistics from Step 1

3. **Model Development**
   - Explain choice of 6 algorithms
   - Describe training process from Step 4

4. **Model Evaluation**
   - Use: visualizations/04_model_comparison.png
   - Use: visualizations/05_confusion_matrices.png
   - Include classification reports

5. **Discussion of Results**
   - Reference key findings above
   - Discuss class imbalance impact
   - Explain metric trade-offs

6. **Limitations & Improvements**
   - Class imbalance → recommend SMOTE/oversampling
   - Low recall → adjust classification threshold
   - Try hyperparameter tuning with GridSearchCV
   - Ensemble methods combining multiple models

7. **Conclusion**
   - Summarize best performing model
   - Business implications
   - Future work suggestions

---

## Report Writing Tips

### For Your 3,000-Word Report:

1. **Data Section** (~400 words)
   - Brief data description
   - Business context
   - Feature explanations
   - Summary statistics

2. **Methodology Section** (~800 words)
   - Data preprocessing details
   - Why 6 models chosen
   - Training approach
   - Evaluation metrics justification

3. **Results Section** (~900 words)
   - Performance comparisons (use tables)
   - Confusion matrices interpretation
   - Feature importance insights
   - ROC curve analysis

4. **Discussion Section** (~500 words)
   - Model strengths/weaknesses
   - Class imbalance implications
   - Business recommendations
   - Performance-interpretability trade-offs

5. **Conclusion** (~100 words)
   - Key takeaways
   - Best model recommendation
   - Future improvements

---

## Visualization Guide

### Generated Charts:

1. **01_target_distribution.png**
   - Class imbalance visualization
   - Use: Introduce class imbalance problem

2. **02_numeric_features_distribution.png**
   - Distributions of numeric features
   - Use: Data exploration section

3. **03_categorical_features_distribution.png**
   - Categorical feature breakdowns
   - Use: Data exploration section

4. **04_model_comparison.png**
   - Bar chart and heatmap of metrics
   - Use: Results section

5. **05_confusion_matrices.png**
   - 6 confusion matrices side-by-side
   - Use: Detailed model evaluation

6. **06_roc_curves.png**
   - ROC curves for all models
   - Use: Model comparison discussion

7. **07_feature_importance.png**
   - Top 10 features from RF and GB
   - Use: Feature analysis discussion

---

## Referencing Your Code

### In Your Report:

```
Example Citation:
"The analysis was conducted using Python 3.10 with scikit-learn
library for machine learning algorithms. Data preprocessing included
label encoding for categorical variables and StandardScaler for
feature normalization. Six classification models were trained and
evaluated on 900 test samples using metrics including accuracy,
precision, recall, F1-score, and ROC-AUC (see Analysis Code, 2026)."
```

---

## Common Questions & Answers

### Q: Why is accuracy high but F1-score low?

**A:** Class imbalance. The model predicts "No" most of the time, which gives high accuracy but misses positives.

### Q: Which model should I recommend?

**A:** Decision Tree shows the best balance. For production, consider:

- SMOTE to handle imbalance
- Threshold adjustment for business requirements
- Hyperparameter tuning

### Q: How do I improve model performance?

**A:** Try these approaches:

1. Class weighting (sklearn: class_weight='balanced')
2. SMOTE oversampling
3. Feature engineering
4. More data collection
5. Ensemble methods

### Q: How many academic sources do I need?

**A:** Minimum 20. Good sources:

- Scikit-learn documentation papers
- Machine learning textbooks
- UCI Repository papers
- Classification algorithms research papers

---

## Contact & Support

Dataset Source: https://archive.ics.uci.edu/ml/datasets/bank+marketing

Scikit-learn Docs: https://scikit-learn.org/stable/

---

**Created:** March 28, 2026  
**Module:** BM 173 - Applications of Machine Learning  
**Institution:** Queen Margaret University (QMU)
