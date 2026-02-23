#include <math.h>
#include <stdio.h>
#include "metrics.h"

/*
 * FUNCTION: MEAN_SQUARED_ERROR (MSE)
 * FORMULA: MSE = (1/N) * SIGMA (y_true - y_pred)^2
 */
double mean_squared_error(Dataset* y_true, Dataset* y_pred) 
{
    double sum_error = 0.0;
    int n = y_true->rows;

    for (int i = 0; i < n; ++i) 
    {
        double diff = y_true->data[i][0] - y_pred->data[i][0];
        sum_error += (diff * diff);
    }
    return sum_error / n;
}

/*
 * FUNCTION: ROOT_MEAN_SQUARED_ERROR (RMSE)
 * FORMULA: SQRT(MSE)
 */
double root_mean_squared_error(Dataset* y_true, Dataset* y_pred) 
{
    return sqrt(mean_squared_error(y_true, y_pred));
}

/*
 * FUNCTION: R2_SCORE
 * FORMULA: 1 - (SS_RES / SS_TOT)
 * SS_RES = SUM SQUARED RESIDUALS (ERROR)
 * SS_TOT = SUM SQUARED DIFFERENCE FROM MEAN
 */
double r2_score(Dataset* y_true, Dataset* y_pred) 
{
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double mean_y = 0.0;
    
    int n = y_true->rows;

    /* CALCULATE MEAN OF OBSERVED DATA */
    for (int i = 0; i < n; ++i) 
        mean_y += y_true->data[i][0];
    
    mean_y /= n;

    for (int i = 0; i < n; ++i) 
    {
        double diff_pred = y_true->data[i][0] - y_pred->data[i][0];
        double diff_mean = y_true->data[i][0] - mean_y;
        
        ss_res += (diff_pred * diff_pred);
        ss_tot += (diff_mean * diff_mean);
    }

    if (ss_tot == 0) return 0.0; // AVOID DIVISION BY ZERO
    return 1.0 - (ss_res / ss_tot);
}

/*
 * FUNCTION: ACCURACY_SCORE
 * FORMULA: CORRECT_PREDICTIONS / TOTAL_PREDICTIONS
 */
double accuracy_score(Dataset* y_true, Dataset* y_pred) 
{
    int correct = 0;
    int n = y_true->rows;

    for (int i = 0; i < n; ++i) 
    {
        /* ASSUMING DISCRETE CLASSES AS INTEGERS (STORED AS DOUBLES) */
        /* USING A SMALL EPSILON FOR FLOATING POINT COMPARISON IF NEEDED */
        if ((int)y_true->data[i][0] == (int)y_pred->data[i][0]) 
            ++correct;
        
    }
    return (double)correct / n;
}

/*
 * FUNCTION: COMPUTE_CONFUSION_MATRIX
 * PURPOSE: TALLIES PREDICTIONS AGAINST GROUND TRUTH FOR BINARY CLASSIFICATION.
 * ASSUMES POSITIVE CLASS IS 1.0 AND NEGATIVE CLASS IS 0.0.
 */
ConfusionMatrix compute_confusion_matrix(Dataset* y_true, Dataset* y_pred) 
{
    ConfusionMatrix cm = {0, 0, 0, 0};
    int n = y_true->rows;

    for (int i = 0; i < n; ++i) 
    {
        int actual = (int)y_true->data[i][0];
        int predicted = (int)y_pred->data[i][0];

        if (actual == 1 && predicted == 1) 
            ++cm.tp;
        else if (actual == 0 && predicted == 0) 
            ++cm.tn;
        else if (actual == 0 && predicted == 1) 
            ++cm.fp;
        else if (actual == 1 && predicted == 0) 
            ++cm.fn;
    }

    return cm;
}

/*
 * FUNCTION: PRECISION_SCORE
 * MATH: TP / (TP + FP)
 * PURPOSE: OF ALL POSITIVE PREDICTIONS, HOW MANY WERE ACTUALLY POSITIVE?
 */
double precision_score(Dataset* y_true, Dataset* y_pred) 
{
    ConfusionMatrix cm = compute_confusion_matrix(y_true, y_pred);
    
    if (cm.tp + cm.fp == 0) 
        return 0.0; /* AVOID DIVISION BY ZERO */
    
    return (double)cm.tp / (cm.tp + cm.fp);
}

/*
 * FUNCTION: RECALL_SCORE
 * MATH: TP / (TP + FN)
 * PURPOSE: OF ALL ACTUAL POSITIVES, HOW MANY DID WE FIND?
 */
double recall_score(Dataset* y_true, Dataset* y_pred) 
{
    ConfusionMatrix cm = compute_confusion_matrix(y_true, y_pred);
    if (cm.tp + cm.fn == 0) 
        return 0.0; /* AVOID DIVISION BY ZERO */
    
    return (double)cm.tp / (cm.tp + cm.fn);
}

/*
 * FUNCTION: PRINT_CONFUSION_MATRIX
 * PURPOSE: VISUAL DEBUGGING OF CLASSIFICATION PERFORMANCE.
 */
void print_confusion_matrix(ConfusionMatrix cm) 
{
    printf("\n=== CONFUSION MATRIX ===\n");
    printf("                 PREDICTED 0 | PREDICTED 1\n");
    printf("ACTUAL 0 (NEG) | %11d | %11d\n", cm.tn, cm.fp);
    printf("ACTUAL 1 (POS) | %11d | %11d\n", cm.fn, cm.tp);
    printf("========================\n\n");
}