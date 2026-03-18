from flask import Flask, render_template, request
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from flask import send_file

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_FOLDER = "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

df = None

# ===============================
# DATA CLEANING
# ===============================

def clean_data(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].astype(str)

            df[col] = df[col].str.replace(",", "", regex=True)
            df[col] = df[col].str.replace("kms", "", regex=True)
            df[col] = df[col].str.replace("km", "", regex=True)
            df[col] = df[col].str.replace("₹", "", regex=True)
            df[col] = df[col].str.strip()

            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    # Fill missing values
    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(df[col].mode()[0], inplace=True)

        else:

            df[col].fillna(df[col].median(), inplace=True)

    return df


# ===============================
# EDA GRAPHS
# ===============================

def generate_graphs(df):

    graphs = []
    df = df.sample(min(len(df), 5000))
    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    heatmap_path = "static/heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    graphs.append(heatmap_path)

    # Plotly histogram (interactive)
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 0:

        fig = px.histogram(numeric_df)

        plot_path = "static/histogram.html"

        fig.write_html(plot_path)

        graphs.append(plot_path)

    # Missing value plot
    plt.figure(figsize=(8,5))
    df.isnull().sum().plot(kind='bar')
    plt.title("Missing Values")
    missing_path = "static/missing.png"
    plt.savefig(missing_path)
    plt.close()

    graphs.append(missing_path)

    return graphs

# ===============================
# DATA SUMMARY
# ===============================

def generate_eda(df):

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    return summary


# ===============================
# HOME PAGE
# ===============================

@app.route("/")
def home():

    return render_template("index.html")


# ===============================
# DATASET UPLOAD
# ===============================

@app.route("/upload", methods=["POST"])
def upload():

    global df

    file = request.files["dataset"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(path)

    df = pd.read_csv(path)

    summary = generate_eda(df)

    graphs = generate_graphs(df)

    return render_template(
        "eda.html",
        tables=[df.head().to_html(classes="table-auto w-full border border-gray-300 text-sm", border=0)],
        summary=summary,
        graphs=graphs
    )


# ===============================
# MODEL TRAINING
# ===============================

@app.route("/train", methods=["POST"])
def train():

    global df

    if df is None:
        return "No dataset uploaded"

    target = request.form["target"]

    if target not in df.columns:
        return "Invalid target column"

    df_clean = clean_data(df.copy())

    X = df_clean.drop(target, axis=1)
    y = df_clean[target]

    X = pd.get_dummies(X, drop_first=True)

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # TRAIN MODELS
    # ===============================

    results = {}
    models = {}

    if y.nunique() < 20:

        task = "Classification"

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50),
            "LogisticRegression": LogisticRegression(max_iter=500),
            "DecisionTree": DecisionTreeClassifier()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results[name] = round(acc, 3)

    else:

        task = "Regression"

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=50),
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = round(score, 3)

    # ===============================
    # BEST MODEL
    # ===============================

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    preds = best_model.predict(X_test)

    # ===============================
    # EVALUATION
    # ===============================

    cm_path = None
    roc_path = None
    residual_path = None

    if task == "Classification":

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")

        cm_path = "static/confusion.png"
        plt.savefig(cm_path)
        plt.close()

        # ROC CURVE (ONLY BINARY)
        if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):

            probs = best_model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], '--')

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()

            roc_path = "static/roc.png"
            plt.savefig(roc_path)
            plt.close()

    else:

        # RESIDUAL PLOT
        residuals = y_test - preds

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=preds, y=residuals)

        plt.axhline(0, color='red', linestyle='--')

        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")

        residual_path = "static/residual.png"
        plt.savefig(residual_path)
        plt.close()

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================

    importance_path = None

    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_
        features = X.columns

        plt.figure(figsize=(8, 4))
        sns.barplot(x=importance, y=features, palette="viridis")

        plt.title("Feature Importance")

        importance_path = "static/importance.png"
        plt.savefig(importance_path)
        plt.close()

    elif hasattr(best_model, "coef_"):

        importance = abs(best_model.coef_[0])
        features = X.columns

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance, y=features)

        plt.title("Feature Importance")

        importance_path = "static/importance.png"
        plt.savefig(importance_path)
        plt.close()

    # ===============================
    # MODEL COMPARISON CHART
    # ===============================

    model_names = list(results.keys())
    scores = list(results.values())

    fig = px.bar(
        x=model_names,
        y=scores,
        text=scores,
        color=scores,
        color_continuous_scale="viridis",
        title="Model Performance Comparison"
    )

    fig.update_traces(textposition='outside')

    chart_path = "static/model_comparison.html"
    fig.write_html(chart_path)

    # ===============================
    # RETURN RESULTS
    # ===============================

    return render_template(
        "results.html",
        results=results,
        best_model=best_model_name,
        importance=importance_path,
        chart=chart_path,
        cm=cm_path,
        roc=roc_path,
        residual=residual_path,
        task=task
    )

@app.route("/download_model")
def download_model():

    return send_file(
        "models/best_model.pkl",
        as_attachment=True
    )


@app.route("/predict")
def predict_page():

    features = joblib.load("models/features.pkl")

    return render_template("predict.html", features=features)


@app.route("/predict_result", methods=["POST"])
def predict_result():

    model = joblib.load("models/best_model.pkl")
    features = joblib.load("models/features.pkl")

    input_data = []

    for feature in features:
        val = request.form.get(feature)

        if val is None or val == "":
            val = 0

        input_data.append(float(val))

    prediction = model.predict([input_data])[0]

    return render_template("results.html", prediction=prediction)

@app.route("/explore")
def explore():

    global df

    if df is None:
        return "Upload dataset first"

    query = request.args.get("q")

    filtered_df = df.copy()

    if query:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: row.astype(str).str.contains(query, case=False).any(),
                axis=1
            )
        ]

    return render_template(
        "explore.html",
        tables=[filtered_df.head(50).to_html(classes="table-auto w-full overflow-x-auto border rounded-lg", border=0)],
        query=query
    )


# ===============================
# RUN APP
# ===============================

if __name__ == "__main__":

    app.run(debug=True)