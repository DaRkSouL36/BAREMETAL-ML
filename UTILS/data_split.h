#ifndef DATA_SPLIT_H
#define DATA_SPLIT_H

#include "data_loader.h"

#ifdef __cplusplus
extern "C" 
{
    #endif

    /*
    * STRUCT: DATA_SPLIT
    * ------------------
    * CONTAINS POINTERS TO THE TRAIN AND TEST SUBSETS.
    */
    typedef struct 
    {
        Dataset* X_train;
        Dataset* y_train;
        Dataset* X_test;
        Dataset* y_test;
    } DataSplit;

    /*
    * FUNCTION: SPLIT_DATASET
    * -----------------------
    * SHUFFLES INDICES AND SPLITS X AND Y INTO TRAIN/TEST SETS.
    * TEST_SIZE: PERCENTAGE (E.G., 0.2 FOR 20%)
    */
    DataSplit split_dataset(Dataset* X, Dataset* y, double test_size);

    #ifdef __cplusplus
}
#endif

#endif