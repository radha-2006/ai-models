# filename: knn_classification_auto_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto KNN Classification", layout="centered")
st.title("⚡ Automated KNN Classification (AutoML Lite)")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # 🔹 AUTO DETECT TARGET (categorical preferred)
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

    # 🔹 ENCODE TARGET IF NEEDED
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔹 SCALING (CRITICAL for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if st.button("🚀 Run Automated Classification"):

        # 🔹 HYPERPARAMETER GRID
        param_grid = {
            "n_neighbors": list(range(1, 21)),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }

        knn = KNeighborsClassifier()

        grid = GridSearchCV(
            knn,
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

        # 🔹 SAMPLE OUTPUT
        results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        st.subheader("🔍 Sample Predictions")
        st.write(results.head())