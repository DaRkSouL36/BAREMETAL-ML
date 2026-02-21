#include "linear_regression.h"
#include <stdexcept>
#include <iostream>

using namespace std;

/* CONSTRUCTOR: INITIALIZE UN-FITTED STATE */
LinearRegression::LinearRegression() : weights(1, 1, 0.0), is_fitted(false) {}

/*
 * FUNCTION: DATASET_TO_MATRIX
 * PURPOSE: BRIDGES OUR C-BASED DATA LOADER WITH OUR C++ MATH MODULE.
 */
Matrix LinearRegression::dataset_to_matrix(Dataset* dataset) 
{
    if (!dataset) 
        throw invalid_argument("NULL DATASET PROVIDED");
    
    Matrix mat(dataset->rows, dataset->cols);
    for (int i = 0; i < dataset->rows; ++i) 
    {
        for (int j = 0; j < dataset->cols; ++j) 
            mat(i, j) = dataset->data[i][j];
    }

    return mat;
}

/*
 * FUNCTION: ADD_BIAS_TERM
 * MATH: X_AUG = [1 | X]
 * ADDS A COLUMN OF ONES TO ALLOW: H_THETA(X) = X_AUG * THETA
 */
Matrix LinearRegression::add_bias_term(const Matrix& X) const 
{
    int rows = X.get_rows();
    int cols = X.get_cols();
    
    Matrix X_aug(rows, cols + 1);

    for (int i = 0; i < rows; ++i) 
    {
        X_aug(i, 0) = 1.0; /* BIAS TERM */
        
        for (int j = 0; j < cols; ++j)
            X_aug(i, j + 1) = X(i, j);
        
    }

    return X_aug;
}

/*
 * FUNCTION: FIT_GRADIENT_DESCENT
 * PURPOSE: OPTIMIZE WEIGHTS USING BATCH GRADIENT DESCENT.
 * * MATHEMATICAL DERIVATION:
 * 1. HYPOTHESIS: PRED = X * THETA
 * 2. ERROR: ERR = PRED - Y
 * 3. GRADIENT OF MSE: GRAD = (1/M) * X^T * ERR
 * 4. UPDATE RULE: THETA = THETA - (ALPHA * GRAD)
 */
void LinearRegression::fit_gradient_descent(const Matrix& X, const Matrix& y, double learning_rate, int epochs) 
{
    if (X.get_rows() != y.get_rows()) 
        throw invalid_argument("ROW COUNT MISMATCH BETWEEN X AND Y");

    /* STEP 1: PREPARE DATA */
    Matrix X_aug = add_bias_term(X);
    int m = X_aug.get_rows(); /* NUMBER OF SAMPLES */
    int n = X_aug.get_cols(); /* NUMBER OF FEATURES (INCLUDING BIAS) */

    /* STEP 2: INITIALIZE WEIGHTS TO ZERO (COLUMN VECTOR N x 1) */
    weights = Matrix(n, 1, 0.0);

    /* STEP 3: OPTIMIZATION LOOP */
    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        /* COMPUTE PREDICTIONS: (M x N) * (N x 1) = (M x 1) */
        Matrix predictions = X_aug.dot(weights);

        /* COMPUTE ERROR: (M x 1) */
        Matrix errors = predictions - y;

        /* COMPUTE GRADIENT: (N x M) * (M x 1) = (N x 1) */
        Matrix gradient = X_aug.transpose().dot(errors) * (1.0 / m);

        /* UPDATE WEIGHTS: THETA = THETA - ALPHA * GRADIENT */
        weights = weights - (gradient * learning_rate);

        if (epoch % 100 == 0) 
        {
            Matrix squared_errors = errors.transpose().dot(errors);
            double mse = squared_errors(0, 0) / m;
            cout << "EPOCH " << epoch << " | MSE: " << mse << "\n";
        }
    }

    is_fitted = true;
    cout << "LINEAR REGRESSION FITTED (GRADIENT DESCENT).\n";
}

/*
 * FUNCTION: FIT_NORMAL_EQUATION
 * PURPOSE: SOLVE FOR OPTIMAL WEIGHTS ANALYTICALLY.
 * * MATHEMATICAL DERIVATION (ORDINARY LEAST SQUARES):
 * SETTING THE DERIVATIVE OF THE MSE COST FUNCTION TO ZERO YIELDS:
 * THETA = (X^T * X)^-1 * X^T * Y
 * * ADVANTAGE: NO LEARNING RATE NEEDED, EXACT SOLUTION.
 * DISADVANTAGE: O(N^3) COMPUTATION TIME DUE TO MATRIX INVERSION.
 */
void LinearRegression::fit_normal_equation(const Matrix& X, const Matrix& y) 
{
    if (X.get_rows() != y.get_rows()) 
        throw invalid_argument("ROW COUNT MISMATCH BETWEEN X AND Y");

    /* STEP 1: PREPARE DATA */
    Matrix X_aug = add_bias_term(X);
    Matrix X_T = X_aug.transpose();

    /* STEP 2: COMPUTE (X^T * X) */
    Matrix X_T_X = X_T.dot(X_aug);

    /* STEP 3: COMPUTE INVERSE (X^T * X)^-1 */
    Matrix inverse_X_T_X = X_T_X.inverse();

    /* STEP 4: COMPUTE THETA = INVERSE * X^T * Y */
    weights = inverse_X_T_X.dot(X_T).dot(y);

    is_fitted = true;
    cout << "LINEAR REGRESSION FITTED (NORMAL EQUATION).\n";
}

/*
 * FUNCTION: PREDICT
 * MATH: Y_PRED = X_AUG * THETA
 */
Matrix LinearRegression::predict(const Matrix& X) const 
{
    if (!is_fitted) 
        throw logic_error("MODEL IS NOT FITTED. CALL FIT() BEFORE PREDICT().");
    
    Matrix X_aug = add_bias_term(X);
    return X_aug.dot(weights);
}

/*
 * FUNCTION: GET_WEIGHTS
 */
Matrix LinearRegression::get_weights() const 
{
    if (!is_fitted) 
        throw logic_error("MODEL IS NOT FITTED. NO WEIGHTS AVAILABLE.");

    return weights;
}