# filename: random_forest_reg_auto_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto Random Forest Regression", layout="centered")
st.title("🌲 Automated Random Forest Regression (AutoML Lite)")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # 🔹 AUTO DETECT NUMERIC COLUMNS
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
    else:
        # 🔹 AUTO TARGET (last numeric column)
        target_column = numeric_cols[-1]
        feature_columns = numeric_cols[:-1]

        st.success(f"Auto-selected Target: {target_column}")
        st.write(f"Auto-selected Features: {feature_columns}")

        X = df[feature_columns]
        y = df[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 🔹 Scaling (not required, but kept consistent)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.button("🚀 Run Automated Training"):

            # 🔹 HYPERPARAMETER GRID
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None]
            }

            rf = RandomForestRegressor(random_state=42)

            grid = GridSearchCV(
                rf,
                param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            st.subheader("✅ Best Parameters Found")
            st.write(grid.best_params_)

            # Predictions
            y_pred = best_model.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📊 Model Performance")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R2 Score: {r2:.4f}")

            # Plot
            st.subheader("📈 Actual vs Predicted")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")

            st.pyplot(fig)

            # 🔹 Feature Importance
            st.subheader("🌟 Feature Importance")

            importances = best_model.feature_importances_
            feature_names = feature_columns

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.write(importance_df)

            # 🔹 Sample predictions
            results = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": y_pred
            })

            st.subheader("🔍 Sample Predictions")
            st.write(results.head())