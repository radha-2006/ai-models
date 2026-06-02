import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stacking Regressor App", layout="wide")

st.title("📊 All-in-One Stacking Regressor App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df)
    st.write("Shape:", df.shape)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Missing values detected! Filling with mean...")
        imputer = SimpleImputer(strategy='mean')
        df[df.columns] = imputer.fit_transform(df)

    # Select columns
    columns = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Variable (Numerical)", columns)
    features = st.multiselect("📥 Select Feature Variables", columns)

    if target and features:
        X = df[features]
        y = df[target]

        # Scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Base models
        st.subheader("⚙️ Model Configuration")

        estimators = [
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor(n_estimators=100)),
            ("svr", SVR())
        ]

        final_estimator = LinearRegression()

        # Stacking model
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator
        )

        model.fit(X_train, y_train)

        st.success("✅ Stacking model trained successfully!")

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📊 Model Performance")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"R² Score: {r2:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"MSE: {mse:.4f}")

        # Visualization (only 1 feature)
        if len(features) == 1:
            st.subheader("📉 Regression Plot")
            plt.figure()
            plt.scatter(X_test, y_test)
            plt.plot(X_test, y_pred)
            st.pyplot(plt)

        # Prediction UI
        st.subheader("🔮 Make a Prediction")
        input_data = []
        for feature in features:
            val = st.number_input(f"Enter value for {feature}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            st.success(f"Predicted Value: {prediction[0]:.4f}")

        # Download model
        st.subheader("💾 Download Model")
        model_package = {
            "model": model,
            "scaler": scaler,
            "features": features
        }

        model_bytes = pickle.dumps(model_package)

        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="stacking_regressor.pkl"
        )

else:
    st.info("📂 Please upload a CSV file to begin.")