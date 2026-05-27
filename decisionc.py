# filename: decision_tree_classification_auto_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto Decision Tree Classification", layout="centered")
st.title("🌳 Automated Decision Tree Classification (AutoML Lite)")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # 🔹 AUTO TARGET DETECTION
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(categorical_cols) > 0:
        target_column = categorical_cols[-1]
    else:
        target_column = numeric_cols[-1]

    feature_columns = [col for col in df.columns if col != target_column]

    st.success(f"Auto-selected Target: {target_column}")
    st.write(f"Auto-selected Features: {feature_columns}")

    X = df[feature_columns]
    y = df[target_column]

    # 🔹 HANDLE CATEGORICAL FEATURES
    X = pd.get_dummies(X, drop_first=True)

    # 🔹 ENCODE TARGET
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Run Automated Classification"):

        # 🔹 HYPERPARAMETER GRID
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"]
        }

        dt = DecisionTreeClassifier(random_state=42)

        grid = GridSearchCV(
            dt,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        st.subheader("✅ Best Parameters Found")
        st.write(grid.best_params_)

        # Predictions
        y_pred = best_model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"Accuracy: {acc:.4f}")

        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # 🔹 CONFUSION MATRIX
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # 🔹 Sample predictions
        results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        st.subheader("🔍 Sample Predictions")
        st.write(results.head())
