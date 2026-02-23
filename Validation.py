import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# CONSTANTS TO MATCH OUR C IMPLEMENTATION
TEST_SIZE = 0.3
RANDOM_SEED = 42   # HARDCODE THIS IN C data_split.c FOR EXACT MATCH
EPOCHS = 1000


# =========================
# REGRESSION (UNCHANGED)
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


# =========================
# CLASSIFICATION (UPDATED)
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

    print("\n--- LOGISTIC REGRESSION (Z-SCORE) ---")
    print(f"ACCURACY: {accuracy_score(y_test, log_preds):.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, log_preds)}")

    # ---------------------------------
    # MODEL 2: KNN (MIN-MAX)
    # ---------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_mm, y_train)
    knn_preds = knn.predict(X_test_mm)

    print("\n--- K-NEAREST NEIGHBORS K=5 (MIN-MAX) ---")
    print(f"ACCURACY: {accuracy_score(y_test, knn_preds):.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, knn_preds)}")

    # ---------------------------------
    # MODEL 3: GAUSSIAN NB (Z-SCORE)
    # ---------------------------------
    gnb = GaussianNB()
    gnb.fit(X_train_std, y_train)
    gnb_preds = gnb.predict(X_test_std)

    print("\n--- GAUSSIAN NAIVE BAYES (Z-SCORE) ---")
    print(f"ACCURACY: {accuracy_score(y_test, gnb_preds):.6f}")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, gnb_preds)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_regression_validation()
    run_classification_validation()