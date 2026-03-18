import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# =========================
# DATA CLEANING
# =========================

def clean_data(df):

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove columns with too many missing values
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    return df


# =========================
# HANDLE MISSING VALUES
# =========================

def handle_missing(df):

    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(df[col].mode()[0], inplace=True)

        else:

            df[col].fillna(df[col].median(), inplace=True)

    return df


# =========================
# ENCODING
# =========================

def encode_features(df):

    label_encoders = {}

    for col in df.columns:

        if df[col].dtype == "object":

            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])

            label_encoders[col] = le

    return df, label_encoders


# =========================
# VISUALIZATION INSIGHTS
# =========================

import matplotlib.pyplot as plt
import seaborn as sns

def generate_insights(df):

    os.makedirs("static", exist_ok=True)

    plots = []

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    path1 = "static/correlation.png"
    plt.savefig(path1)
    plt.close()

    plots.append(path1)

    # Distribution plots
    df.hist(figsize=(10,8))
    path2 = "static/distribution.png"
    plt.savefig(path2)
    plt.close()

    plots.append(path2)

    # Boxplot for outliers
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df)
    path3 = "static/outliers.png"
    plt.savefig(path3)
    plt.close()

    plots.append(path3)

    return plots


# =========================
# TRAIN MODELS
# =========================

def train_models(df, target):

    df = clean_data(df)
    df = handle_missing(df)

    df, encoders = encode_features(df)

    X = df.drop(target, axis=1)
    y = df[target]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Detect problem type
    if y.nunique() < 20:

        problem_type = "classification"

        models = {

            "Random Forest": RandomForestClassifier(),

            "Logistic Regression": LogisticRegression(max_iter=500),

            "Decision Tree": DecisionTreeClassifier(),

            "SVM": SVC(),

            "KNN": KNeighborsClassifier()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            results[name] = round(score,3)

    else:

        problem_type = "regression"

        models = {

            "Random Forest": RandomForestRegressor(),

            "Linear Regression": LinearRegression(),

            "Decision Tree": DecisionTreeRegressor(),

            "SVR": SVR(),

            "KNN": KNeighborsRegressor()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = round(score,3)

    # Select best model
    best_model_name = max(results, key=results.get)

    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")

    # Feature importance (if available)
    importance_plot = None

    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_

        plt.figure(figsize=(8,6))

        sns.barplot(x=importance, y=df.drop(target,axis=1).columns)

        importance_plot = "static/feature_importance.png"

        plt.title("Feature Importance")

        plt.savefig(importance_plot)

        plt.close()

    insights = generate_insights(df)

    return {

        "problem_type": problem_type,

        "results": results,

        "best_model": best_model_name,

        "feature_importance": importance_plot,

        "insight_plots": insights

    }