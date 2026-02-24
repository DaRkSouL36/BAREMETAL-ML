import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# =========================
# CONSTANTS TO MATCH OUR C/C++ IMPLEMENTATION
# =========================
TEST_SIZE = 0.3
RANDOM_SEED = 42   
EPOCHS = 1000

# =========================
# FUNCTION: LOG_METRIC
# PURPOSE: APPENDS MODEL PERFORMANCE TO A SHARED CSV FILE IN THE 'RESULT' DIRECTORY.
# =========================
def log_metric(model_name, metric_name, value):
    # CREATE DIRECTORY IF IT DOES NOT EXIST
    os.makedirs("RESULT", exist_ok=True)
    
    file_path = "RESULT/PYTHON_SKLEARN_RESULTS.csv"
    file_exists = os.path.isfile(file_path)
    
    # APPEND METRICS TO CSV
    with open(file_path, "a") as file:
        if not file_exists:
            file.write("LANGUAGE,MODEL,METRIC,VALUE\n")
        file.write(f"PYTHON,{model_name},{metric_name},{value}\n")

# =========================
# REGRESSION PIPELINE
# =========================
def run_regression_validation():
    print("\n" + "="*50)
    print("      SCIKIT-LEARN: LINEAR REGRESSION")
    print("="*50)
    
    try:
        X = pd.read_csv("DATA/RegressionX.csv", header=None).values
        y = pd.read_csv("DATA/RegressionY.csv", header=None).values.ravel()
    except FileNotFoundError:
        print("ERROR: REGRESSION DATA NOT FOUND.")
        return

    # SPLIT AND SCALE (Z-SCORE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. NORMAL EQUATION
    lr_exact = LinearRegression()
    lr_exact.fit(X_train_scaled, y_train)
    preds_exact = lr_exact.predict(X_test_scaled)
    mse_exact = mean_squared_error(y_test, preds_exact)

    # 2. GRADIENT DESCENT (SGD)
    lr_gd = SGDRegressor(
        max_iter=EPOCHS,
        learning_rate='constant',
        eta0=0.01,
        random_state=RANDOM_SEED
    )
    lr_gd.fit(X_train_scaled, y_train)
    preds_gd = lr_gd.predict(X_test_scaled)
    mse_gd = mean_squared_error(y_test, preds_gd)

    print(f"GRADIENT DESCENT MSE : {mse_gd:.6f}")
    print(f"NORMAL EQUATION MSE  : {mse_exact:.6f}")

    # ------------------------------------------------------------------
    # LOG REGRESSION METRICS TO CSV
    # ------------------------------------------------------------------
    log_metric("LINEAR REGRESSION (GRADIENT DESCENT)", "MSE", mse_gd)
    log_metric("LINEAR REGRESSION (NORMAL EQUATION)", "MSE", mse_exact)


# =========================
# CLASSIFICATION PIPELINE
# =========================
def run_classification_validation():
    print("\n" + "="*50)
    print("      SCIKIT-LEARN: CLASSIFICATION")
    print("="*50)
    
    try:
        X = pd.read_csv("DATA/ClassifierX.csv", header=None).values
        y = pd.read_csv("DATA/ClassifierY.csv", header=None).values.ravel()
    except FileNotFoundError:
        print("ERROR: CLASSIFICATION DATA NOT FOUND.")
        return

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # ---------------------------------
    # SCALING STRATEGY (MULTI-SCALER)
    # ---------------------------------

    # 1. Z-SCORE (StandardScaler)
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    # 2. MIN-MAX (MinMaxScaler)
    mm_scaler = MinMaxScaler()
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)

    # ---------------------------------
    # MODEL 1: LOGISTIC REGRESSION (Z-SCORE)
    # ---------------------------------
    log_reg = LogisticRegression(
        penalty=None,
        max_iter=EPOCHS,
        random_state=RANDOM_SEED
    )
    log_reg.fit(X_train_std, y_train)
    log_preds = log_reg.predict(X_test_std)
    log_acc = accuracy_score(y_test, log_preds)

    print("\n--- LOGISTIC REGRESSION (Z-SCORE) ---")
    print(f"ACCURACY: {log_acc:.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, log_preds)}")
    
    log_metric("LOGISTIC REGRESSION", "ACCURACY", log_acc)

    # ---------------------------------
    # MODEL 2: KNN (MIN-MAX)
    # ---------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_mm, y_train)
    knn_preds = knn.predict(X_test_mm)
    knn_acc = accuracy_score(y_test, knn_preds)

    print("\n--- K-NEAREST NEIGHBORS K=5 (MIN-MAX) ---")
    print(f"ACCURACY: {knn_acc:.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, knn_preds)}")
    
    log_metric("K-NEAREST NEIGHBORS", "ACCURACY", knn_acc)

    # ---------------------------------
    # MODEL 3: GAUSSIAN NB (Z-SCORE)
    # ---------------------------------
    gnb = GaussianNB()
    gnb.fit(X_train_std, y_train)
    gnb_preds = gnb.predict(X_test_std)
    gnb_acc = accuracy_score(y_test, gnb_preds)

    print("\n--- GAUSSIAN NAIVE BAYES (Z-SCORE) ---")
    print(f"ACCURACY: {gnb_acc:.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, gnb_preds)}")
    
    log_metric("GAUSSIAN NAIVE BAYES", "ACCURACY", gnb_acc)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("INITIATING SCIKIT-LEARN VALIDATION SUITE\n")
    
    # CREATE DIRECTORY UPFRONT JUST IN CASE
    os.makedirs("RESULT", exist_ok=True)
    
    run_regression_validation()
    run_classification_validation()
    
    print("\nALL VALIDATION PIPELINES EXECUTED SUCCESSFULLY.")
    print("[SYSTEM] RESULTS LOGGED TO: RESULT/PYTHON_SKLEARN_RESULTS.csv")