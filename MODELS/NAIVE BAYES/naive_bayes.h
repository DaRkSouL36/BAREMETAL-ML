#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "../../UTILS/data_loader.h"

#ifdef __cplusplus
extern "C" 
{
    #endif

    /*
    * STRUCT: GAUSSIAN_NB
    * -------------------
    * STORES THE LEARNED PARAMETERS FOR A BINARY CLASSIFIER (CLASSES 0 AND 1).
    * PRIORS: PROBABILITY OF OCCURRENCE OF EACH CLASS.
    * MEANS & VARIANCES: ARRAYS OF SIZE (NUM_FEATURES) FOR EACH CLASS.
    */
    typedef struct 
    {
        int num_features;
        
        /* CLASS 0 PARAMETERS */
        double prior_0;
        double* mean_0;
        double* var_0;
        
        /* CLASS 1 PARAMETERS */
        double prior_1;
        double* mean_1;
        double* var_1;
        
        int is_fitted;
    } GaussianNB;

    /* CONSTRUCTOR & DESTRUCTOR */
    GaussianNB* create_gaussian_nb(int num_features);
    void free_gaussian_nb(GaussianNB* model);

    /*
    * FUNCTION: GAUSSIAN_NB_FIT
    * -------------------------
    * COMPUTES THE PRIORS, MEANS, AND VARIANCES FOR EACH FEATURE, 
    * SEPARATED BY CLASS.
    */
    void gaussian_nb_fit(GaussianNB* model, Dataset* X, Dataset* y);

    /*
    * FUNCTION: GAUSSIAN_NB_PREDICT
    * -----------------------------
    * CALCULATES THE LOG-POSTERIOR FOR EACH CLASS AND RETURNS THE 
    * CLASS WITH THE MAXIMUM VALUE.
    */
    Dataset* gaussian_nb_predict(GaussianNB* model, Dataset* X);

    #ifdef __cplusplus
}
#endif

#endif