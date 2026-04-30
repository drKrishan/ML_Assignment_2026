"""
Bank Marketing Classification Analysis
BM 173 - Applications of Machine Learning
Student: [Your Name]
Date: March 2026

This script performs a comprehensive machine learning analysis on the Bank Marketing dataset
to predict whether a client will subscribe to a term deposit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import warnings

warnings.filterwarnings("ignore")

# Set style for visualizations
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ==============================================================================
# 1. LOAD AND EXPLORE THE DATASET
# ==============================================================================

print("=" * 80)
print("STEP 1: LOADING AND EXPLORING THE DATASET")
print("=" * 80)

# Download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
df = pd.read_csv(url, sep=";")
print("\nLoading dataset from local file...")

dataset_path = "data/bank-additional-full.csv"

print("\n1.1 Dataset Overview")
print(f"Dataset shape: {df.shape}")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

print("\n1.2 First Few Records")
df = pd.read_csv(dataset_path, sep=";")
print(f"Dataset loaded from: {dataset_path}")
print(df.head())

print("\n1.3 Data Types")
print(df.dtypes)

print("\n1.4 Missing Values")
print(df.isnull().sum())

print("\n1.5 Basic Statistics")
print(df.describe())

print("\n1.6 Target Variable Distribution")
print(df["y"].value_counts())
print(f"Class Distribution:")
print(
    f"  - No Subscription: {(df['y']=='no').sum()} ({(df['y']=='no').sum()/len(df)*100:.2f}%)"
)
print(
    f"  - Subscription: {(df['y']=='yes').sum()} ({(df['y']=='yes').sum()/len(df)*100:.2f}%)"
)

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Create visualizations directory
import os

os.makedirs("visualizations", exist_ok=True)

# 2.1 Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df["y"].value_counts().plot(kind="bar", ax=axes[0], color=["lightcoral", "lightgreen"])
axes[0].set_title(
    "Target Variable Distribution: Bank Subscription", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Subscription")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["No", "Yes"], rotation=0)

df["y"].value_counts().plot(
    kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["lightcoral", "lightgreen"]
)
axes[1].set_title("Target Variable Proportion", fontsize=12, fontweight="bold")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("visualizations/01_target_distribution.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: visualizations/01_target_distribution.png")
plt.close()

# 2.2 Numeric features analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols[:6]):
    axes[idx].hist(df[col], bins=30, color="skyblue", edgecolor="black")
    axes[idx].set_title(f"Distribution of {col}", fontweight="bold")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    "visualizations/02_numeric_features_distribution.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: visualizations/02_numeric_features_distribution.png")
plt.close()

# 2.3 Categorical features analysis
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("y")  # Remove target variable

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols[:8]):
    df[col].value_counts().plot(
        kind="bar", ax=axes[idx], color="lightblue", edgecolor="black"
    )
    axes[idx].set_title(f"{col} Distribution", fontweight="bold", fontsize=10)
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Count")
    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(
    "visualizations/03_categorical_features_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
print("✓ Saved: visualizations/03_categorical_features_distribution.png")
plt.close()

# ==============================================================================
# 3. DATA PREPARATION AND PREPROCESSING
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 3: DATA PREPARATION AND PREPROCESSING")
print("=" * 80)

df_processed = df.copy()

print("\n3.1 Encoding Categorical Variables")

# Identify categorical columns
categorical_features = df_processed.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns: {categorical_features}")

# Label encode the target variable
le_target = LabelEncoder()
df_processed["y"] = le_target.fit_transform(df_processed["y"])
print(f"Target variable encoded: No -> 0, Yes -> 1")

# Label encode other categorical variables
label_encoders = {}
for col in categorical_features:
    if col != "y":
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")

print("\n3.2 Handling Missing Values")
missing_count = df_processed.isnull().sum().sum()
print(f"Total missing values: {missing_count}")

print("\n3.3 Feature Scaling")
X = df_processed.drop("y", axis=1)
y = df_processed["y"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training set class distribution:")
print(f"  - Class 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"  - Class 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ==============================================================================
# 4. MODEL DEVELOPMENT AND TRAINING
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 4: MODEL DEVELOPMENT AND TRAINING")
print("=" * 80)

# Dictionary to store models and their results
models = {}
results = {}

print("\n4.1 Training Multiple Classification Models\n")

# Model 1: Logistic Regression
print("Training Logistic Regression...")
models["Logistic Regression"] = LogisticRegression(random_state=42, max_iter=1000)
models["Logistic Regression"].fit(X_train_scaled, y_train)
print("✓ Logistic Regression trained")

# Model 2: Decision Tree
print("Training Decision Tree Classifier...")
models["Decision Tree"] = DecisionTreeClassifier(random_state=42, max_depth=10)
models["Decision Tree"].fit(X_train, y_train)
print("✓ Decision Tree trained")

# Model 3: Random Forest
print("Training Random Forest Classifier...")
models["Random Forest"] = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
models["Random Forest"].fit(X_train, y_train)
print("✓ Random Forest trained")

# Model 4: Support Vector Machine
print("Training Support Vector Machine (SVM)...")
models["SVM"] = SVC(kernel="rbf", random_state=42, probability=True)
models["SVM"].fit(X_train_scaled, y_train)
print("✓ SVM trained")

# Model 5: K-Nearest Neighbors
print("Training K-Nearest Neighbors...")
models["KNN"] = KNeighborsClassifier(n_neighbors=5)
models["KNN"].fit(X_train_scaled, y_train)
print("✓ KNN trained")

# Model 6: Gradient Boosting
print("Training Gradient Boosting Classifier...")
models["Gradient Boosting"] = GradientBoostingClassifier(
    n_estimators=100, random_state=42
)
models["Gradient Boosting"].fit(X_train, y_train)
print("✓ Gradient Boosting trained")

# ==============================================================================
# 5. MODEL EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL EVALUATION AND PERFORMANCE METRICS")
print("=" * 80)

# Prepare data for evaluation based on whether model uses scaled or unscaled data
X_test_eval = {
    "Logistic Regression": X_test_scaled,
    "Decision Tree": X_test,
    "Random Forest": X_test,
    "SVM": X_test_scaled,
    "KNN": X_test_scaled,
    "Gradient Boosting": X_test,
}

print("\n5.1 Model Performance on Test Set\n")

for model_name, model in models.items():
    X_eval = X_test_eval[model_name]

    # Predictions
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    print(f"{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}\n")

# ==============================================================================
# 6. COMPARISON AND VISUALIZATION OF RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 6: MODEL COMPARISON AND VISUALIZATION")
print("=" * 80)

# Create comparison dataframe
results_df = pd.DataFrame(results).T
results_df = results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]

print("\n6.1 Model Comparison Table")
print(results_df)

# 6.2 Comparison Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Model Performance Comparison
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
results_df[metrics_to_plot].plot(kind="bar", ax=axes[0], width=0.8)
axes[0].set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Model", fontweight="bold")
axes[0].set_ylabel("Score", fontweight="bold")
axes[0].set_ylim([0, 1.05])
axes[0].legend(loc="lower right")
axes[0].grid(axis="y", alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Plot 2: Heatmap of metrics
sns.heatmap(
    results_df[metrics_to_plot],
    annot=True,
    fmt=".4f",
    cmap="RdYlGn",
    ax=axes[1],
    cbar_kws={"label": "Score"},
    vmin=0,
    vmax=1,
)
axes[1].set_title("Model Performance Heatmap", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("visualizations/04_model_comparison.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: visualizations/04_model_comparison.png")
plt.close()

# 6.3 Best Model Based on Different Criteria
print("\n6.2 Best Models by Metric:")
for metric in metrics_to_plot:
    best_model = results_df[metric].idxmax()
    best_score = results_df[metric].max()
    print(f"  Best {metric}: {best_model} ({best_score:.4f})")

# ==============================================================================
# 7. DETAILED CONFUSION MATRICES AND CLASSIFICATION REPORTS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 7: DETAILED EVALUATION - CONFUSION MATRICES & CLASSIFICATION REPORTS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, (model_name, model) in enumerate(models.items()):
    X_eval = X_test_eval[model_name]
    y_pred = results[model_name]["y_pred"]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[idx],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes[idx].set_title(
        f'{model_name}\nAccuracy: {results[model_name]["Accuracy"]:.4f}',
        fontweight="bold",
    )
    axes[idx].set_ylabel("True Label")
    axes[idx].set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("visualizations/05_confusion_matrices.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: visualizations/05_confusion_matrices.png")
plt.close()

# Print detailed classification reports
print("\n7.1 Detailed Classification Reports:\n")
for model_name, model in models.items():
    y_pred = results[model_name]["y_pred"]
    print(f"\n{model_name}")
    print("-" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=["No Subscription", "Subscription"]
        )
    )

# ==============================================================================
# 8. ROC CURVES
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 8: ROC CURVES ANALYSIS")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for idx, (model_name, model) in enumerate(models.items()):
    y_pred_proba = results[model_name]["y_pred_proba"]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = results[model_name]["ROC-AUC"]

    ax.plot(
        fpr, tpr, color=colors[idx], lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})"
    )

# Plot random classifier baseline
ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier (AUC = 0.5000)")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontweight="bold", fontsize=12)
ax.set_ylabel("True Positive Rate", fontweight="bold", fontsize=12)
ax.set_title("ROC Curves - Model Comparison", fontweight="bold", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("visualizations/06_roc_curves.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: visualizations/06_roc_curves.png")
plt.close()

# ==============================================================================
# 9. FEATURE IMPORTANCE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\n9.1 Feature Importance for Tree-based Models\n")

# Random Forest Feature Importance
rf_importance = models["Random Forest"].feature_importances_
rf_features = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_importance}
).sort_values("Importance", ascending=False)

print("Random Forest Top 10 Important Features:")
print(rf_features.head(10))

# Gradient Boosting Feature Importance
gb_importance = models["Gradient Boosting"].feature_importances_
gb_features = pd.DataFrame(
    {"Feature": X.columns, "Importance": gb_importance}
).sort_values("Importance", ascending=False)

print("\nGradient Boosting Top 10 Important Features:")
print(gb_features.head(10))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
rf_features.head(10).plot(
    x="Feature", y="Importance", kind="barh", ax=axes[0], color="forestgreen"
)
axes[0].set_title(
    "Random Forest - Top 10 Feature Importance", fontweight="bold", fontsize=12
)
axes[0].set_xlabel("Importance Score")
axes[0].invert_yaxis()

# Gradient Boosting
gb_features.head(10).plot(
    x="Feature", y="Importance", kind="barh", ax=axes[1], color="coral"
)
axes[1].set_title(
    "Gradient Boosting - Top 10 Feature Importance", fontweight="bold", fontsize=12
)
axes[1].set_xlabel("Importance Score")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("visualizations/07_feature_importance.png", dpi=300, bbox_inches="tight")
print("\n✓ Saved: visualizations/07_feature_importance.png")
plt.close()

# ==============================================================================
# 10. SUMMARY AND CONCLUSIONS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 10: SUMMARY AND KEY FINDINGS")
print("=" * 80)

best_model_name = results_df["F1-Score"].idxmax()
best_f1 = results_df["F1-Score"].max()

print(f"\n✓ Analysis Complete!")
print(f"\nBest Performing Model: {best_model_name}")
print(f"Best F1-Score: {best_f1:.4f}")

print(f"\nTop 3 Models by F1-Score:")
top_3 = results_df["F1-Score"].nlargest(3)
for rank, (model, score) in enumerate(top_3.items(), 1):
    print(f"  {rank}. {model}: {score:.4f}")

print(f"\nKey Insights:")
print(f"  - Dataset has {len(df)} records with {df.shape[1]} features")
print(
    f"  - Target variable imbalance: {(df['y']=='no').sum()/len(df)*100:.1f}% negative, {(df['y']=='yes').sum()/len(df)*100:.1f}% positive"
)
print(f"  - {len(models)} classification models trained and evaluated")
print(f"  - Visualizations saved to 'visualizations/' directory")

print("\n✓ All analysis complete! Ready for report writing.")
print("=" * 80)
