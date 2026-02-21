#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "data_loader.h"

#ifdef __cplusplus
extern "C" 
{
    #endif

    /*
    * STRUCT: STANDARD_SCALER
    * -----------------------
    * HOLDS THE STATE (MEAN AND STD) FOR Z-SCORE NORMALIZATION.
    * EXACT EQUIVALENT OF SKLEARN'S StandardScaler.
    */
    typedef struct 
    {
        int num_features;
        double* mean;
        double* std;
    } StandardScaler;

    /*
    * STRUCT: MIN_MAX_SCALER
    * ----------------------
    * HOLDS THE STATE (MIN AND MAX) FOR MIN-MAX SCALING.
    * EXACT EQUIVALENT OF SKLEARN'S MinMaxScaler.
    */
    typedef struct 
    {
        int num_features;
        double* min_val;
        double* max_val;
    } MinMaxScaler;

    /* -------------------------------------------------------------------------- */
    /* STANDARD SCALER METHODS                                                    */
    /* -------------------------------------------------------------------------- */

    StandardScaler* create_standard_scaler(int num_features);
    void free_standard_scaler(StandardScaler* scaler);

    /* COMPUTES MEAN AND STD FROM THE TRAINING DATA */
    void standard_scaler_fit(StandardScaler* scaler, Dataset* X);

    /* APPLIES THE COMPUTED MEAN AND STD TO TRANSFORM THE DATA */
    Dataset* standard_scaler_transform(StandardScaler* scaler, Dataset* X);

    /* PERFORMS FIT AND TRANSFORM IN ONE STEP FOR EFFICIENCY */
    Dataset* standard_scaler_fit_transform(StandardScaler* scaler, Dataset* X);

    /* -------------------------------------------------------------------------- */
    /* MIN-MAX SCALER METHODS                                                     */
    /* -------------------------------------------------------------------------- */

    MinMaxScaler* create_min_max_scaler(int num_features);
    void free_min_max_scaler(MinMaxScaler* scaler);

    /* COMPUTES MIN AND MAX FROM THE TRAINING DATA */
    void min_max_scaler_fit(MinMaxScaler* scaler, Dataset* X);

    /* APPLIES THE COMPUTED MIN AND MAX TO TRANSFORM THE DATA */
    Dataset* min_max_scaler_transform(MinMaxScaler* scaler, Dataset* X);

    /* PERFORMS FIT AND TRANSFORM IN ONE STEP */
    Dataset* min_max_scaler_fit_transform(MinMaxScaler* scaler, Dataset* X);

    #ifdef __cplusplus
}
#endif

#endif