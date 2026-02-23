#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "knn.h"

/*
 * FUNCTION: EUCLIDEAN_DISTANCE
 * MATH: SQRT( SUM( (X_A - X_B)^2 ) )
 * COMPUTES THE STRAIGHT-LINE DISTANCE BETWEEN TWO FEATURE VECTORS.
 */
static double euclidean_distance(double* vector_a, double* vector_b, int num_features) 
{
    double sum_squared_diff = 0.0;
    
    for (int j = 0; j < num_features; ++j) 
    {
        double diff = vector_a[j] - vector_b[j];
        sum_squared_diff += (diff * diff);
    }

    return sqrt(sum_squared_diff);
}

/*
 * FUNCTION: COMPARE_NEIGHBORS
 * PURPOSE: COMPARATOR FUNCTION FOR QSORT.
 * SORTS NEIGHBORS IN ASCENDING ORDER BASED ON DISTANCE.
 */
static int compare_neighbors(const void* a, const void* b) 
{
    Neighbor* n1 = (Neighbor*)a;
    Neighbor* n2 = (Neighbor*)b;
    
    if (n1->distance < n2->distance) 
        return -1;
    if (n1->distance > n2->distance) 
        return 1;
    
        return 0;
}

/*
 * FUNCTION: MAJORITY_VOTE
 * PURPOSE: COUNTS FREQUENCIES OF LABELS IN THE TOP K NEIGHBORS.
 * ASSUMES BINARY CLASSIFICATION (LABELS 0.0 AND 1.0) FOR SIMPLICITY.
 * CAN BE EXTENDED TO MULTI-CLASS BY USING A HASH MAP OR FREQUENCY ARRAY.
 */
static double majority_vote(Neighbor* neighbors, int k) 
{
    int count_zero = 0;
    int count_one = 0;
    
    for (int i = 0; i < k; ++i) 
    {
        if ((int)neighbors[i].label == 0) 
            ++count_zero;
        
        else if ((int)neighbors[i].label == 1) 
            ++count_one;
        
    }
    
    /* RETURN THE MOST FREQUENT LABEL */
    return (count_one > count_zero) ? 1.0 : 0.0;
}

/* -------------------------------------------------------------------------- */
/* CLASS METHODS                                                              */
/* -------------------------------------------------------------------------- */

KNNClassifier* create_knn_classifier(int k) 
{
    if (k <= 0) 
    {
        printf("ERROR: K MUST BE GREATER THAN 0.\n");
        return NULL;
    }
    
    KNNClassifier* model = (KNNClassifier*)malloc(sizeof(KNNClassifier));
    
    model->X_train = NULL;
    model->y_train = NULL;
    model->k = k;
    model->is_fitted = 0;
    
    return model;
}

void free_knn_classifier(KNNClassifier* model) 
{
    if (model) 
    {
        /* WE DO NOT FREE X_TRAIN AND Y_TRAIN HERE AS THE MODEL ONLY HOLDS POINTERS. 
         * THE MEMORY IS MANAGED EXTERNALLY BY THE DATA_LOADER. 
         */
        free(model);
    }
}

/*
 * FUNCTION: KNN_FIT
 * PURPOSE: "LAZY LEARNING" - WE JUST STORE POINTERS TO THE TRAINING DATA.
 */
void knn_fit(KNNClassifier* model, Dataset* X_train, Dataset* y_train) 
{
    if (X_train->rows != y_train->rows) 
    {
        printf("ERROR: X_TRAIN AND Y_TRAIN ROW COUNTS DO NOT MATCH.\n");
        return;
    }
    
    model->X_train = X_train;
    model->y_train = y_train;
    model->is_fitted = 1;
    
    printf("KNN FITTED (MEMORIZED %d SAMPLES WITH K=%d).\n", X_train->rows, model->k);
}

/*
 * FUNCTION: KNN_PREDICT
 * PURPOSE: FOR EVERY TEST SAMPLE, FIND THE K CLOSEST TRAINING SAMPLES AND VOTE.
 * COMPLEXITY: O(N_TEST * N_TRAIN * LOG(N_TRAIN)) DUE TO SORTING.
 */
Dataset* knn_predict(KNNClassifier* model, Dataset* X_test) 
{
    if (!model->is_fitted) 
    {
        printf("ERROR: MODEL NOT FITTED. CALL FIT() FIRST.\n");
        return NULL;
    }
    
    if (model->X_train->cols != X_test->cols) 
    {
        printf("ERROR: FEATURE COUNT MISMATCH BETWEEN TRAIN AND TEST SETS.\n");
        return NULL;
    }

    int num_test = X_test->rows;
    int num_train = model->X_train->rows;
    int num_features = X_test->cols;
    
    /* IF K IS GREATER THAN AVAILABLE TRAINING DATA, CAP IT */
    int k_actual = (model->k > num_train) ? num_train : model->k;

    /* PREPARE OUTPUT DATASET */
    Dataset* predictions = (Dataset*)malloc(sizeof(Dataset));
    predictions->rows = num_test;
    predictions->cols = 1;
    predictions->data = (double**)malloc(num_test * sizeof(double*));

    /* ARRAY TO HOLD DISTANCES FOR A SINGLE TEST POINT TO ALL TRAIN POINTS */
    Neighbor* distances = (Neighbor*)malloc(num_train * sizeof(Neighbor));

    /* ITERATE OVER EACH TEST SAMPLE */
    for (int i = 0; i < num_test; ++i) 
    {    
        /* 1. COMPUTE DISTANCE FROM THIS TEST SAMPLE TO ALL TRAINING SAMPLES */
        for (int j = 0; j < num_train; ++j) 
        {
            distances[j].distance = euclidean_distance(X_test->data[i], model->X_train->data[j], num_features);
            distances[j].label = model->y_train->data[j][0];
        }

        /* 2. SORT DISTANCES IN ASCENDING ORDER */
        qsort(distances, num_train, sizeof(Neighbor), compare_neighbors);

        /* 3. EXTRACT MAJORITY VOTE FROM TOP K NEIGHBORS */
        double predicted_label = majority_vote(distances, k_actual);

        /* 4. STORE PREDICTION */
        predictions->data[i] = (double*)malloc(sizeof(double));
        predictions->data[i][0] = predicted_label;
    }

    free(distances);
    return predictions;
}