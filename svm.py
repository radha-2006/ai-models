import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SVM Classifier App", layout="wide")

st.title("📊 All-in-One SVM Classifier App")

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

    # Column selection
    columns = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Variable (Categorical)", columns)
    features = st.multiselect("📥 Select Feature Variables", columns)

    if target and features:
        X = df[features]
        y = df[target]

        # Encode target if needed
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        # Scaling (very important for SVM)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Kernel selection
        st.subheader("⚙️ Model Configuration")
        kernel = st.selectbox("Select Kernel", ["linear", "rbf", "poly", "sigmoid"])
        C = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
        gamma = st.selectbox("Gamma", ["scale", "auto"])

        # Train model
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

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

        # Visualization (only if 2 features)
        if len(features) == 2:
            st.subheader("📉 Decision Boundary")

            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure()
            plt.contourf(xx, yy, Z, alpha=0.3)
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            st.pyplot(plt)

        # Prediction
        st.subheader("🔮 Make a Prediction")
        input_data = []
        for feature in features:
            val = st.number_input(f"Enter value for {feature}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            st.success(f"Predicted Class: {prediction[0]}")

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
            file_name="svm_model.pkl"
        )

else:
    st.info("📂 Please upload a CSV file to begin.")