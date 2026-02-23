#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "naive_bayes.h"

/* CONSTANT FOR LOG(2 * PI) USED IN GAUSSIAN LIKELIHOOD */
#define LOG_2_PI 1.83787706640934548356

/* EPSILON TO PREVENT DIVISION BY ZERO IN VARIANCE (VARIANCE SMOOTHING) */
#define EPSILON 1e-9

GaussianNB* create_gaussian_nb(int num_features) 
{
    GaussianNB* model = (GaussianNB*)malloc(sizeof(GaussianNB));
    model->num_features = num_features;
    
    model->prior_0 = 0.0;
    model->mean_0 = (double*)calloc(num_features, sizeof(double));
    model->var_0 = (double*)calloc(num_features, sizeof(double));
    
    model->prior_1 = 0.0;
    model->mean_1 = (double*)calloc(num_features, sizeof(double));
    model->var_1 = (double*)calloc(num_features, sizeof(double));
    
    model->is_fitted = 0;
    return model;
}

void free_gaussian_nb(GaussianNB* model) 
{
    if (model) 
    {
        free(model->mean_0);
        free(model->var_0);
        free(model->mean_1);
        free(model->var_1);
        free(model);
    }
}

/*
 * FUNCTION: GAUSSIAN_NB_FIT
 * PURPOSE: CALCULATE PRIORS, MEANS, AND VARIANCES FROM TRAINING DATA.
 * LOGIC:
 * 1. COUNT FREQUENCY OF CLASSES FOR PRIORS.
 * 2. COMPUTE MEAN OF EACH FEATURE PER CLASS.
 * 3. COMPUTE VARIANCE OF EACH FEATURE PER CLASS.
 */
void gaussian_nb_fit(GaussianNB* model, Dataset* X, Dataset* y) 
{
    if (X->cols != model->num_features || X->rows != y->rows) 
    {
        printf("ERROR: DIMENSION MISMATCH IN NAIVE BAYES FIT.\n");
        return;
    }

    int m = X->rows;
    int n = X->cols;
    int count_0 = 0;
    int count_1 = 0;

    /* STEP 1: COMPUTE MEANS AND CLASS COUNTS */
    for (int i = 0; i < m; ++i) 
    {
        int label = (int)y->data[i][0];
        if (label == 0) 
        {
            ++count_0;
            for (int j = 0; j < n; ++j) 
                model->mean_0[j] += X->data[i][j];
            
        } 
        else if (label == 1) 
        {
            ++count_1;
            for (int j = 0; j < n; ++j)
                model->mean_1[j] += X->data[i][j];
        }
    }

    /* NORMALIZE MEANS AND CALCULATE PRIORS */
    model->prior_0 = (double)count_0 / m;
    model->prior_1 = (double)count_1 / m;

    for (int j = 0; j < n; ++j) 
    {
        if (count_0 > 0) 
            model->mean_0[j] /= count_0;
        if (count_1 > 0) 
            model->mean_1[j] /= count_1;
    }

    /* STEP 2: COMPUTE VARIANCES */
    for (int i = 0; i < m; ++i) 
    {
        int label = (int)y->data[i][0];
        
        for (int j = 0; j < n; ++j) 
        {
            if (label == 0) 
            {
                double diff = X->data[i][j] - model->mean_0[j];
                model->var_0[j] += (diff * diff);
            } 
            
            else if (label == 1) 
            {
                double diff = X->data[i][j] - model->mean_1[j];
                model->var_1[j] += (diff * diff);
            }
        }
    }

    /* NORMALIZE VARIANCES AND ADD VARIANCE SMOOTHING (EPSILON) */
    for (int j = 0; j < n; ++j) 
    {            
        if (count_0 > 0) 
            model->var_0[j] = (model->var_0[j] / count_0) + EPSILON;
        if (count_1 > 0) 
            model->var_1[j] = (model->var_1[j] / count_1) + EPSILON;
    }

    model->is_fitted = 1;
    printf("NAIVE BAYES FITTED (PRIOR 0: %.2f, PRIOR 1: %.2f).\n", model->prior_0, model->prior_1);
}

/*
 * FUNCTION: CALCULATE_LOG_LIKELIHOOD
 * MATH: -0.5 * LOG(2 * PI * VAR) - ((X - MEAN)^2 / (2 * VAR))
 * PURPOSE: COMPUTES LOG P(X_i | Y) FOR A SINGLE FEATURE.
 */
static double calculate_log_likelihood(double x, double mean, double var) 
{
    double term1 = -0.5 * log(LOG_2_PI * var); /* WE PRE-COMPUTED LOG_2_PI CONSTANT */
    double diff = x - mean;
    double term2 = -(diff * diff) / (2.0 * var);
  
    return term1 + term2;
}

/*
 * FUNCTION: GAUSSIAN_NB_PREDICT
 * PURPOSE: FOR EACH SAMPLE, COMPUTE LOG POSTERIOR FOR CLASS 0 AND 1, 
 * CHOOSE THE MAXIMUM.
 */
Dataset* gaussian_nb_predict(GaussianNB* model, Dataset* X) 
{
    if (!model->is_fitted) 
    {
        printf("ERROR: MODEL NOT FITTED. CANNOT PREDICT.\n");
        return NULL;
    }

    int m = X->rows;
    int n = X->cols;

    Dataset* predictions = (Dataset*)malloc(sizeof(Dataset));
    predictions->rows = m;
    predictions->cols = 1;
    predictions->data = (double**)malloc(m * sizeof(double*));

    for (int i = 0; i < m; ++i) 
    {
        /* INITIALIZE POSTERIOR WITH LOG(PRIOR) */
        /* LOG(0) IS UNDEFINED, SO WE ADD A TINY NUMBER IF PRIOR IS EXACTLY 0 */
        double log_post_0 = log(model->prior_0 + EPSILON); 
        double log_post_1 = log(model->prior_1 + EPSILON);

        /* ADD LOG-LIKELIHOODS OF EACH FEATURE */
        for (int j = 0; j < n; ++j) 
        {
            log_post_0 += calculate_log_likelihood(X->data[i][j], model->mean_0[j], model->var_0[j]);
            log_post_1 += calculate_log_likelihood(X->data[i][j], model->mean_1[j], model->var_1[j]);
        }

        /* ARGMAX: SELECT CLASS WITH HIGHEST LOG-POSTERIOR */
        predictions->data[i] = (double*)malloc(sizeof(double));
        predictions->data[i][0] = (log_post_1 > log_post_0) ? 1.0 : 0.0;
    }

    return predictions;
}