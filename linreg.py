import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Linear Regression App", layout="wide")

st.title("📊 All-in-One Linear Regression App")

# Upload CSV
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

    # Column selection
    columns = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Variable", columns)
    features = st.multiselect("📥 Select Feature Variables", columns)

    if target and features:
        X = df[features]
        y = df[target]

        # Train-test split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # Coefficients
        st.subheader("📈 Model Details")
        coeff_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.dataframe(coeff_df)
        st.write("Intercept:", model.intercept_)

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

        # Visualization (only for 1 feature)
        if len(features) == 1:
            st.subheader("📉 Regression Plot")
            plt.figure()
            plt.scatter(X_test, y_test)
            plt.plot(X_test, y_pred)
            plt.xlabel(features[0])
            plt.ylabel(target)
            st.pyplot(plt)

        # Custom prediction
        st.subheader("🔮 Make a Prediction")
        input_data = []
        for feature in features:
            val = st.number_input(f"Enter value for {feature}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            prediction = model.predict([input_data])
            st.success(f"Predicted Value: {prediction[0]:.4f}")

        # Download model
        st.subheader("💾 Download Trained Model")
        model_bytes = pickle.dumps(model)
        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="linear_regression_model.pkl"
        )

else:
    st.info("📂 Please upload a CSV file to start.")