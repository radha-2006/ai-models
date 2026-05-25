import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Logistic Regression App", layout="wide")

st.title("📊 All-in-One Logistic Regression App")

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
    target = st.selectbox("🎯 Select Target Variable (Categorical)", columns)
    features = st.multiselect("📥 Select Feature Variables", columns)

    if target and features:
        X = df[features]
        y = df[target]

        # Encode target if not numeric
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        # Scaling (important for logistic regression)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # Model coefficients
        st.subheader("📈 Model Details")
        coeff_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_[0]
        })
        st.dataframe(coeff_df)
        st.write("Intercept:", model.intercept_[0])

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📊 Model Performance")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Visualization (only for 1 feature)
        if len(features) == 1:
            st.subheader("📉 Decision Boundary Plot")

            X_vis = X_test[:, 0]
            plt.figure()
            plt.scatter(X_vis, y_test)
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
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            st.success(f"Predicted Class: {prediction[0]}")

        # Download model + scaler
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
            file_name="logistic_regression_model.pkl"
        )

else:
    st.info("📂 Please upload a CSV file to start.")