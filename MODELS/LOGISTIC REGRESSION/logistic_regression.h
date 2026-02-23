#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "../../UTILS/data_loader.h"

#ifdef __cplusplus
extern "C" 
{
    #endif

    /*
    * STRUCT: LOGISTIC_REGRESSION
    * ---------------------------
    * HOLDS THE STATE OF THE TRAINED MODEL.
    * USES AN EXPLICIT BIAS TERM RATHER THAN AUGMENTING THE FEATURE MATRIX,
    * AS MATRIX AUGMENTATION IS HEAVIER IN PURE C WITHOUT A MATRIX LIBRARY.
    */
    typedef struct 
    {
        int num_features;
        double* weights;
        double bias;
        int is_fitted;
    } LogisticRegression;

    /* CONSTRUCTOR & DESTRUCTOR */
    LogisticRegression* create_logistic_regression(int num_features);
    void free_logistic_regression(LogisticRegression* model);

    /*
    * FUNCTION: LOGISTIC_REGRESSION_FIT
    * ---------------------------------
    * TRAINS THE MODEL USING BATCH GRADIENT DESCENT YIELDING OPTIMAL WEIGHTS.
    */
    void logistic_regression_fit(LogisticRegression* model, Dataset* X, Dataset* y, double learning_rate, int epochs);

    /*
    * FUNCTION: LOGISTIC_REGRESSION_PREDICT_PROBA
    * -------------------------------------------
    * RETURNS THE RAW SIGMOID OUTPUTS (PROBABILITIES BETWEEN 0 AND 1).
    */
    Dataset* logistic_regression_predict_proba(LogisticRegression* model, Dataset* X);

    /*
    * FUNCTION: LOGISTIC_REGRESSION_PREDICT
    * -------------------------------------
    * APPLIES A DECISION THRESHOLD (USUALLY 0.5) TO YIELD STRICT 0 OR 1 LABELS.
    */
    Dataset* logistic_regression_predict(LogisticRegression* model, Dataset* X, double threshold);

    #ifdef __cplusplus
}
#endif

#endif