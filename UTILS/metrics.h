#ifndef METRICS_H
#define METRICS_H

#include "data_loader.h"

#ifdef __cplusplus
extern "C" 
{
    #endif

    /* REGRESSION METRICS */
    double mean_squared_error(Dataset* y_true, Dataset* y_pred);
    double root_mean_squared_error(Dataset* y_true, Dataset* y_pred);
    double r2_score(Dataset* y_true, Dataset* y_pred);

    /* CLASSIFICATION METRICS */
    double accuracy_score(Dataset* y_true, Dataset* y_pred);

    /*
    * STRUCT: CONFUSION_MATRIX
    * ------------------------
    * HOLDS TRUE POSITIVES, TRUE NEGATIVES, FALSE POSITIVES, FALSE NEGATIVES.
    */
    typedef struct 
    {
        int tp;
        int tn;
        int fp;
        int fn;
    } ConfusionMatrix;

    /* EXTENDED CLASSIFICATION METRICS */
    ConfusionMatrix compute_confusion_matrix(Dataset* y_true, Dataset* y_pred);
    double precision_score(Dataset* y_true, Dataset* y_pred);
    double recall_score(Dataset* y_true, Dataset* y_pred);
    void print_confusion_matrix(ConfusionMatrix cm);

    #ifdef __cplusplus
}
#endif

#endif