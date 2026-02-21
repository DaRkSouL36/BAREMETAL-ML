#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "../../MATH/matrix.h"
#include "../../UTILS/data_loader.h" /* FOR C-COMPATIBILITY IN MAIN */

using namespace std;

/*
 * CLASS: LINEAR_REGRESSION
 * ------------------------
 * IMPLEMENTS ORDINARY LEAST SQUARES (OLS) LINEAR REGRESSION.
 * DESIGNED WITH SCIKIT-LEARN PARITY (FIT, PREDICT PARADIGM).
 */
class LinearRegression 
{
    private:
        /* * THE WEIGHT VECTOR (THETA). 
        * STORED AS A COLUMN MATRIX (N X 1) FOR COMPATIBILITY WITH OUR MATRIX CLASS.
        * INDEX 0 HOLDS THE INTERCEPT (BIAS), INDICES 1 TO N HOLD FEATURE WEIGHTS.
        */
        Matrix weights;
        bool is_fitted;

        /*
        * FUNCTION: ADD_BIAS_TERM
        * PURPOSE: APPENDS A COLUMN OF 1.0s TO THE LEFT OF THE FEATURE MATRIX.
        * THIS ALLOWS US TO COMPUTE THE INTERCEPT VIA MATRIX MULTIPLICATION.
        */
        Matrix add_bias_term(const Matrix& X) const;

    public:
        /* CONSTRUCTOR */
        LinearRegression();

        /*
        * METHOD 1: FIT_GRADIENT_DESCENT
        * ------------------------------
        * ITERATIVELY OPTIMIZES WEIGHTS TO MINIMIZE MEAN SQUARED ERROR (MSE).
        * REQUIRES LEARNING RATE (ALPHA) AND NUMBER OF EPOCHS.
        */
        void fit_gradient_descent(const Matrix& X, const Matrix& y, double learning_rate, int epochs);

        /*
        * METHOD 2: FIT_NORMAL_EQUATION
        * -----------------------------
        * COMPUTES THE EXACT MATHEMATICAL SOLUTION IN A SINGLE STEP.
        * O(N^3) COMPLEXITY DUE TO MATRIX INVERSION.
        */
        void fit_normal_equation(const Matrix& X, const Matrix& y);

        /*
        * FUNCTION: PREDICT
        * -----------------
        * OUTPUTS CONTINUOUS TARGET VALUES BASED ON LEARNED WEIGHTS.
        */
        Matrix predict(const Matrix& X) const;

        /*
        * FUNCTION: GET_WEIGHTS
        * ---------------------
        * ALLOWS INSPECTION OF THE LEARNED PARAMETERS (THETA).
        */
        Matrix get_weights() const;
        
        /* UTILITY TO CONVERT C-STRUCT DATASET TO C++ MATRIX */
        static Matrix dataset_to_matrix(Dataset* dataset);
};

#endif