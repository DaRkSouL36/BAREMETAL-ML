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

    #ifdef __cplusplus
}
#endif

#endif