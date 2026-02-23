#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "logistic_regression.h"

/*
 * FUNCTION: SIGMOID
 * MATH: S(Z) = 1 / (1 + E^(-Z))
 * INCLUDES BOUNDS CHECKING TO PREVENT FLOATING POINT OVERFLOW ON E^(-Z).
 */
static double sigmoid(double z) 
{
    if (z < -40.0) 
        return 0.0; /* PREVENT EXP() OVERFLOW */
    if (z > 40.0) 
        return 1.0;
    
    return 1.0 / (1.0 + exp(-z));
}

LogisticRegression* create_logistic_regression(int num_features) 
{
    LogisticRegression* model = (LogisticRegression*)malloc(sizeof(LogisticRegression));
    
    model->num_features = num_features;
    model->weights = (double*)calloc(num_features, sizeof(double)); /* INIT TO 0 */
    model->bias = 0.0;
    model->is_fitted = 0;
    
    return model;
}

void free_logistic_regression(LogisticRegression* model) 
{
    if (model) 
    {
        free(model->weights);
        free(model);
    }
}

/*
 * FUNCTION: LOGISTIC_REGRESSION_FIT
 * PURPOSE: EXECUTES BATCH GRADIENT DESCENT TO MINIMIZE LOG LOSS.
 */
void logistic_regression_fit(LogisticRegression* model, Dataset* X, Dataset* y, double learning_rate, int epochs) 
{
    if (X->cols != model->num_features || X->rows != y->rows) 
    {
        printf("ERROR: DIMENSION MISMATCH IN LOGISTIC REGRESSION FIT.\n");
        return;
    }

    int m = X->rows;
    int n = X->cols;
    
    /* TEMPORARY ARRAYS TO HOLD GRADIENTS FOR THE CURRENT EPOCH */
    double* dw = (double*)malloc(n * sizeof(double));
    double db = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        /* RESET GRADIENTS TO ZERO FOR THIS EPOCH */
        for (int j = 0; j < n; ++j) 
            dw[j] = 0.0;
        db = 0.0;

        /* ITERATE OVER ALL SAMPLES TO COMPUTE TOTAL GRADIENT */
        for (int i = 0; i < m; ++i) 
        {
            /* 1. COMPUTE LINEAR COMBINATION: Z = W*X + B */
            double z = model->bias;
            
            for (int j = 0; j < n; ++j) 
                z += model->weights[j] * X->data[i][j];
            
            /* 2. APPLY SIGMOID TO GET PREDICTION */
            double y_pred = sigmoid(z);
            double y_true = y->data[i][0];

            /* 3. CALCULATE ERROR (DIFFERENCE) */
            double dz = y_pred - y_true;

            /* 4. ACCUMULATE GRADIENTS */
            db += dz;

            for (int j = 0; j < n; ++j) 
                dw[j] += dz * X->data[i][j];
        }

        /* UPDATE WEIGHTS AND BIAS USING AVERAGED GRADIENTS */
        model->bias -= learning_rate * (db / m);
        
        for (int j = 0; j < n; ++j) 
            model->weights[j] -= learning_rate * (dw[j] / m);
    }

    free(dw);
    model->is_fitted = 1;
    printf("LOGISTIC REGRESSION FITTED SUCCESSFULLY.\n");
}

/*
 * FUNCTION: LOGISTIC_REGRESSION_PREDICT_PROBA
 * PURPOSE: RETURNS THE RAW PROBABILITY THAT A SAMPLE BELONGS TO CLASS 1.
 */
Dataset* logistic_regression_predict_proba(LogisticRegression* model, Dataset* X) 
{
    if (!model->is_fitted) 
    {
        printf("ERROR: MODEL NOT FITTED. CANNOT PREDICT.\n");
        return NULL;
    }

    Dataset* probas = (Dataset*)malloc(sizeof(Dataset));
    probas->rows = X->rows;
    probas->cols = 1;
    probas->data = (double**)malloc(X->rows * sizeof(double*));

    for (int i = 0; i < X->rows; ++i) 
    {
        probas->data[i] = (double*)malloc(sizeof(double));
        
        double z = model->bias;
        
        for (int j = 0; j < X->cols; ++j) 
            z += model->weights[j] * X->data[i][j];
        
        probas->data[i][0] = sigmoid(z);
    }

    return probas;
}

/*
 * FUNCTION: LOGISTIC_REGRESSION_PREDICT
 * PURPOSE: APPLIES DECISION BOUNDARY TO YIELD DISCRETE LABELS (0 OR 1).
 */
Dataset* logistic_regression_predict(LogisticRegression* model, Dataset* X, double threshold) 
{
    Dataset* probas = logistic_regression_predict_proba(model, X);
    if (!probas) return NULL;

    Dataset* labels = (Dataset*)malloc(sizeof(Dataset));
    labels->rows = X->rows;
    labels->cols = 1;
    labels->data = (double**)malloc(X->rows * sizeof(double*));

    for (int i = 0; i < X->rows; ++i) 
    {
        labels->data[i] = (double*)malloc(sizeof(double));
        labels->data[i][0] = (probas->data[i][0] >= threshold) ? 1.0 : 0.0;
    }

    free_dataset(probas); /* FREE THE TEMPORARY PROBABILITIES */
    return labels;
}