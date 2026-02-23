#ifndef KNN_H
#define KNN_H

#include "../../UTILS/data_loader.h"

#ifdef __cplusplus
extern "C" {
    #endif

    /*
    * STRUCT: KNN_CLASSIFIER
    * ----------------------
    * STORES POINTERS TO THE TRAINING DATA AND THE HYPERPARAMETER K.
    * SINCE KNN IS NON-PARAMETRIC, THE DATA ITSELF IS THE MODEL.
    */
    typedef struct 
    {
        Dataset* X_train;
        Dataset* y_train;
        int k;
        int is_fitted;
    } KNNClassifier;

    /*
    * STRUCT: NEIGHBOR
    * ----------------
    * HELPER STRUCT FOR SORTING DISTANCES.
    * BINDS THE COMPUTED DISTANCE TO THE CORRESPONDING TARGET LABEL.
    */
    typedef struct 
    {
        double distance;
        double label;
    } Neighbor;

    /* CONSTRUCTOR & DESTRUCTOR */
    KNNClassifier* create_knn_classifier(int k);
    void free_knn_classifier(KNNClassifier* model);

    /*
    * FUNCTION: KNN_FIT
    * -----------------
    * MEMORIZES THE TRAINING DATA. NO OPTIMIZATION OCCURS HERE.
    */
    void knn_fit(KNNClassifier* model, Dataset* X_train, Dataset* y_train);

    /*
    * FUNCTION: KNN_PREDICT
    * ---------------------
    * FOR EACH TEST SAMPLE, COMPUTES DISTANCES, SORTS, AND APPLIES MAJORITY VOTE.
    */
    Dataset* knn_predict(KNNClassifier* model, Dataset* X_test);

    #ifdef __cplusplus
}
#endif

#endif