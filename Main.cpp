#include <iostream>
#include <iomanip>

/* INCLUDE C UTILS */
#include "UTILS/data_loader.h"
#include "UTILS/data_split.h"
#include "UTILS/normalization.h"
#include "UTILS/metrics.h"

/* INCLUDE MODELS */
#include "MODELS/LINEAR REGRESSION/linear_regression.h"
#include "MODELS/LOGISTIC REGRESSION/logistic_regression.h"
#include "MODELS/K-NEAREST NEIGHBOR/knn.h"
#include "MODELS/NAIVE BAYES/naive_bayes.h"

using namespace std;

/*
 * FUNCTION: RUN_REGRESSION_PIPELINE
 * PURPOSE: LOADS DATA, SPLITS, NORMALIZES, AND TRAINS LINEAR REGRESSION.
 */
void run_regression_pipeline() 
{
    cout << "\n==================================================\n";
    cout << "          PIPELINE 1: LINEAR REGRESSION             \n";
    cout << "==================================================\n";

    Dataset* X = load_csv("DATA/RegressionX.csv");
    Dataset* y = load_csv("DATA/RegressionY.csv");
    
    if (!X || !y) 
    {
        cout << "ERROR: REGRESSION DATA NOT FOUND. SKIPPING.\n";
        return;
    }

    DataSplit split = split_dataset(X, y, 0.3);

    /* GRADIENT DESCENT PREFERS Z-SCORE STANDARDIZATION */
    StandardScaler* scaler = create_standard_scaler(X->cols);
    Dataset* X_train_scaled = standard_scaler_fit_transform(scaler, split.X_train);
    Dataset* X_test_scaled = standard_scaler_transform(scaler, split.X_test);

    Matrix mat_X_train = LinearRegression::dataset_to_matrix(X_train_scaled);
    Matrix mat_y_train = LinearRegression::dataset_to_matrix(split.y_train);
    Matrix mat_X_test = LinearRegression::dataset_to_matrix(X_test_scaled);
    Matrix mat_y_test = LinearRegression::dataset_to_matrix(split.y_test);

    LinearRegression lr_gd;
    LinearRegression lr_ne;

    cout << "\n--- TRAINING: GRADIENT DESCENT ---\n";
    lr_gd.fit_gradient_descent(mat_X_train, mat_y_train, 0.01, 1000);
    Matrix preds_gd = lr_gd.predict(mat_X_test);

    cout << "\n--- TRAINING: NORMAL EQUATION ---\n";
    lr_ne.fit_normal_equation(mat_X_train, mat_y_train);
    Matrix preds_ne = lr_ne.predict(mat_X_test);

    double mse_gd = 0.0, mse_ne = 0.0;
    int m = mat_y_test.get_rows();

    for (int i = 0; i < m; ++i) 
    {
        double diff_gd = preds_gd(i, 0) - mat_y_test(i, 0);
        double diff_ne = preds_ne(i, 0) - mat_y_test(i, 0);
        mse_gd += (diff_gd * diff_gd);
        mse_ne += (diff_ne * diff_ne);
    }

    mse_gd /= m; mse_ne /= m;

    cout << "\n--- EVALUATION RESULTS ---\n";
    cout << "GRADIENT DESCENT MSE : " << mse_gd << "\n";
    cout << "NORMAL EQUATION MSE  : " << mse_ne << "\n";

    /* MEMORY CLEANUP */
    free_standard_scaler(scaler);
    free_dataset(X); free_dataset(y);
    free_dataset(split.X_train); free_dataset(split.y_train);
    free_dataset(split.X_test); free_dataset(split.y_test);
    free_dataset(X_train_scaled); free_dataset(X_test_scaled);
}

/*
 * FUNCTION: RUN_CLASSIFICATION_PIPELINE
 * PURPOSE: ORCHESTRATES MODELS WITH APPROPRIATE SCALING TECHNIQUES.
 */
void run_classification_pipeline() 
{
    cout << "\n==================================================\n";
    cout << "        PIPELINE 2: CLASSIFICATION MODELS           \n";
    cout << "==================================================\n";

    Dataset* X = load_csv("DATA/ClassifierX.csv");
    Dataset* y = load_csv("DATA/ClassifierY.csv");
    
    if (!X || !y) 
    {
        cout << "ERROR: CLASSIFICATION DATA NOT FOUND. SKIPPING.\n";
        return;
    }

    DataSplit split = split_dataset(X, y, 0.3);

    /* ------------------------------------------------------------------ */
    /* SCALING ARCHITECTURE */
    /* ------------------------------------------------------------------ */
    
    /* 1. Z-SCORE SCALING (FOR LOGISTIC REGRESSION AND NAIVE BAYES) */
    StandardScaler* std_scaler = create_standard_scaler(X->cols);
    Dataset* X_train_std = standard_scaler_fit_transform(std_scaler, split.X_train);
    Dataset* X_test_std = standard_scaler_transform(std_scaler, split.X_test);

    /* 2. MIN-MAX SCALING (FOR KNN) */
    MinMaxScaler* mm_scaler = create_min_max_scaler(X->cols);
    Dataset* X_train_mm = min_max_scaler_fit_transform(mm_scaler, split.X_train);
    Dataset* X_test_mm = min_max_scaler_transform(mm_scaler, split.X_test);

    /* ------------------------------------------------------------------ */
    /* MODEL 2: LOGISTIC REGRESSION (USES Z-SCORE) */
    /* ------------------------------------------------------------------ */
    cout << "\n--- TRAINING: LOGISTIC REGRESSION (Z-SCORE SCALED) ---\n";
    LogisticRegression* log_reg = create_logistic_regression(X->cols);
    logistic_regression_fit(log_reg, X_train_std, split.y_train, 0.01, 1000);
    
    Dataset* log_preds = logistic_regression_predict(log_reg, X_test_std, 0.5);
    cout << "LOGISTIC REGRESSION ACCURACY: " << accuracy_score(split.y_test, log_preds) << "\n";
    print_confusion_matrix(compute_confusion_matrix(split.y_test, log_preds));

    /* ------------------------------------------------------------------ */
    /* MODEL 3: K-NEAREST NEIGHBORS (USES MIN-MAX) */
    /* ------------------------------------------------------------------ */
    cout << "\n--- TRAINING: K-NEAREST NEIGHBORS (K=5) (MIN-MAX SCALED) ---\n";
    KNNClassifier* knn = create_knn_classifier(5);
    knn_fit(knn, X_train_mm, split.y_train);
    
    Dataset* knn_preds = knn_predict(knn, X_test_mm);
    cout << "KNN ACCURACY: " << accuracy_score(split.y_test, knn_preds) << "\n";
    print_confusion_matrix(compute_confusion_matrix(split.y_test, knn_preds));

    /* ------------------------------------------------------------------ */
    /* MODEL 4: GAUSSIAN NAIVE BAYES (USES Z-SCORE) */
    /* ------------------------------------------------------------------ */
    cout << "\n--- TRAINING: GAUSSIAN NAIVE BAYES (Z-SCORE SCALED) ---\n";
    GaussianNB* gnb = create_gaussian_nb(X->cols);
    gaussian_nb_fit(gnb, X_train_std, split.y_train);
    
    Dataset* gnb_preds = gaussian_nb_predict(gnb, X_test_std);
    cout << "NAIVE BAYES ACCURACY: " << accuracy_score(split.y_test, gnb_preds) << "\n";
    print_confusion_matrix(compute_confusion_matrix(split.y_test, gnb_preds));

    /* MEMORY CLEANUP */
    free_logistic_regression(log_reg);
    free_knn_classifier(knn);
    free_gaussian_nb(gnb);
    free_dataset(log_preds); free_dataset(knn_preds); free_dataset(gnb_preds);
    
    free_standard_scaler(std_scaler);
    free_min_max_scaler(mm_scaler);
    
    free_dataset(X_train_std); free_dataset(X_test_std);
    free_dataset(X_train_mm); free_dataset(X_test_mm);
    
    free_dataset(X); free_dataset(y);
    free_dataset(split.X_train); free_dataset(split.y_train);
    free_dataset(split.X_test); free_dataset(split.y_test);
}

int main() 
{
    cout << "INITIATING ML FROM SCRATCH EVALUATION SUITE\n";
    
    run_regression_pipeline();
    run_classification_pipeline();
    
    cout << "\nALL PIPELINES EXECUTED SUCCESSFULLY.\n";
    return 0;
}