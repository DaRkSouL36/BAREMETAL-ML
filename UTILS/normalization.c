#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "normalization.h"

/* ========================================================================== */
/* STANDARD SCALER IMPLEMENTATION                                             */
/* ========================================================================== */

StandardScaler* create_standard_scaler(int num_features) 
{
    StandardScaler* scaler = (StandardScaler*)malloc(sizeof(StandardScaler));
    scaler->num_features = num_features;
    scaler->mean = (double*)calloc(num_features, sizeof(double));
    scaler->std = (double*)calloc(num_features, sizeof(double));
    
    return scaler;
}

void free_standard_scaler(StandardScaler* scaler) 
{
    if (scaler) 
    {
        free(scaler->mean);
        free(scaler->std);
        free(scaler);
    }
}

/*
 * FUNCTION: STANDARD_SCALER_FIT
 * MATH: CALCULATES MU (MEAN) AND SIGMA (STD) FOR EACH FEATURE (COLUMN).
 * EQUATION: MU = (1/N) * SUM(X), SIGMA = SQRT((1/N) * SUM((X - MU)^2))
 */
void standard_scaler_fit(StandardScaler* scaler, Dataset* X) 
{
    if (scaler->num_features != X->cols) 
    {
        printf("ERROR: SCALER FEATURE COUNT DOES NOT MATCH DATASET.\n");
        return;
    }

    int n = X->rows;
    int cols = X->cols;

    /* STEP 1: CALCULATE MEAN FOR EACH COLUMN */
    for (int j = 0; j < cols; ++j) 
    {
        double sum = 0.0;
        
        for (int i = 0; i < n; ++i) 
            sum += X->data[i][j];
        
        scaler->mean[j] = sum / n;
    }

    /* STEP 2: CALCULATE STANDARD DEVIATION FOR EACH COLUMN */
    for (int j = 0; j < cols; ++j) 
    {
        double sum_sq_diff = 0.0;
        
        for (int i = 0; i < n; ++i) 
        {
            double diff = X->data[i][j] - scaler->mean[j];
            sum_sq_diff += (diff * diff);
        }
        /* ADD EPSILON TO PREVENT DIVISION BY ZERO IN TRANSFORM */
        scaler->std[j] = sqrt(sum_sq_diff / n) + 1e-8; 
    }
}

/*
 * FUNCTION: STANDARD_SCALER_TRANSFORM
 * MATH: Z = (X - MU) / SIGMA
 * RETURNS A NEW ALLOCATED DATASET TO PRESERVE ORIGINAL DATA.
 */
Dataset* standard_scaler_transform(StandardScaler* scaler, Dataset* X) 
{
    Dataset* transformed = (Dataset*)malloc(sizeof(Dataset));
    transformed->rows = X->rows;
    transformed->cols = X->cols;
    transformed->data = (double**)malloc(X->rows * sizeof(double*));

    for (int i = 0; i < X->rows; ++i) 
    {
        transformed->data[i] = (double*)malloc(X->cols * sizeof(double));
        
        for (int j = 0; j < X->cols; ++j) 
            transformed->data[i][j] = (X->data[i][j] - scaler->mean[j]) / scaler->std[j];
    }
    return transformed;
}

Dataset* standard_scaler_fit_transform(StandardScaler* scaler, Dataset* X) 
{
    standard_scaler_fit(scaler, X);
    
    return standard_scaler_transform(scaler, X);
}

/* ========================================================================== */
/* MIN-MAX SCALER IMPLEMENTATION                                              */
/* ========================================================================== */

MinMaxScaler* create_min_max_scaler(int num_features) 
{
    MinMaxScaler* scaler = (MinMaxScaler*)malloc(sizeof(MinMaxScaler));
    scaler->num_features = num_features;
    scaler->min_val = (double*)malloc(num_features * sizeof(double));
    scaler->max_val = (double*)malloc(num_features * sizeof(double));
    
    return scaler;
}

void free_min_max_scaler(MinMaxScaler* scaler) 
{
    if (scaler) 
    {
        free(scaler->min_val);
        free(scaler->max_val);
        free(scaler);
    }
}

/*
 * FUNCTION: MIN_MAX_SCALER_FIT
 * MATH: FINDS THE MINIMUM AND MAXIMUM VALUES FOR EACH FEATURE COLUMN.
 */
void min_max_scaler_fit(MinMaxScaler* scaler, Dataset* X) 
{
    int n = X->rows;
    int cols = X->cols;

    for (int j = 0; j < cols; ++j) 
    {
        scaler->min_val[j] = X->data[0][j];
        scaler->max_val[j] = X->data[0][j];

        for (int i = 1; i < n; ++i) 
        {
            if (X->data[i][j] < scaler->min_val[j]) 
                scaler->min_val[j] = X->data[i][j];
            if (X->data[i][j] > scaler->max_val[j]) 
                scaler->max_val[j] = X->data[i][j];
        }
        
        /* PREVENT DIVISION BY ZERO IF MIN == MAX (CONSTANT FEATURE) */
        if (scaler->max_val[j] == scaler->min_val[j]) 
            scaler->max_val[j] += 1e-8;
    }
}

/*
 * FUNCTION: MIN_MAX_SCALER_TRANSFORM
 * MATH: X_SCALED = (X - X_MIN) / (X_MAX - X_MIN)
 */
Dataset* min_max_scaler_transform(MinMaxScaler* scaler, Dataset* X) 
{
    Dataset* transformed = (Dataset*)malloc(sizeof(Dataset));
    transformed->rows = X->rows;
    transformed->cols = X->cols;
    transformed->data = (double**)malloc(X->rows * sizeof(double*));

    for (int i = 0; i < X->rows; ++i) 
    {
        transformed->data[i] = (double*)malloc(X->cols * sizeof(double));
        
        for (int j = 0; j < X->cols; ++j) 
            transformed->data[i][j] = (X->data[i][j] - scaler->min_val[j]) / (scaler->max_val[j] - scaler->min_val[j]);
    }
    return transformed;
}

Dataset* min_max_scaler_fit_transform(MinMaxScaler* scaler, Dataset* X) 
{
    min_max_scaler_fit(scaler, X);
    return min_max_scaler_transform(scaler, X);
}