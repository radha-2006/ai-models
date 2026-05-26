# filename: svr_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

st.title("Support Vector Regression (SVR) App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features and target
    columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Variable", columns)
    feature_columns = st.multiselect("Select Feature Variables", columns)

    if target_column and feature_columns:

        X = df[feature_columns]
        y = df[target_column]

        # Train-test split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # SVR parameters
        st.sidebar.header("SVR Parameters")

        kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.sidebar.slider("Regularization (C)", 0.1, 100.0, 1.0)
        epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)

        if st.button("Train Model"):

            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
            y_test_actual = scaler_y.inverse_transform(y_test.values.reshape(-1, 1))

            # Metrics
            mse = mean_squared_error(y_test_actual, y_pred)
            r2 = r2_score(y_test_actual, y_pred)

            st.subheader("Model Performance")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R2 Score: {r2:.4f}")

            # Plot
            st.subheader("Actual vs Predicted")

            fig, ax = plt.subplots()
            ax.scatter(y_test_actual, y_pred)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")

            st.pyplot(fig)