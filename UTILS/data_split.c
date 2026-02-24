#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_split.h"

/*
 * FUNCTION: SHUFFLE_INDICES
 * PURPOSE: IMPLEMENTS FISHER-YATES SHUFFLE ALGORITHM
 * MATH: ENSURES EACH PERMUTATION IS EQUALLY PROBABLE
 */
void shuffle_indices(int* indices, int n) 
{
    srand(42); // SEED RANDOM NUMBER GENERATOR
    
    for (int i = n - 1; i > 0; --i) 
    {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

/*
 * FUNCTION: CREATE_SUBSET
 * PURPOSE: HELPER TO COPY DATA FROM INDICES INTO NEW DATASET
 */
Dataset* create_subset(Dataset* source, int* indices, int start, int end) 
{
    int rows = end - start;
    int cols = source->cols;

    Dataset* subset = (Dataset*)malloc(sizeof(Dataset));
    subset->rows = rows;
    subset->cols = cols;
    subset->data = (double**)malloc(rows * sizeof(double*));

    for (int i = 0; i < rows; ++i) 
    {
        subset->data[i] = (double*)malloc(cols * sizeof(double));
        
        int source_idx = indices[start + i];
        for (int j = 0; j < cols; ++j) 
            subset->data[i][j] = source->data[source_idx][j];
        
    }

    return subset;
}

DataSplit split_dataset(Dataset* X, Dataset* y, double test_size) 
{
    /* VALIDATION: ROWS MUST MATCH */
    if (X->rows != y->rows) 
    {
        printf("ERROR: X AND Y ROW COUNTS DO NOT MATCH\n");
        exit(1);
    }

    int n = X->rows;
    int test_count = (int)(n * test_size + 0.5);
    int train_count = n - test_count;

    /* CREATE INDEX ARRAY [0, 1, ..., N-1] */
    int* indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) 
        indices[i] = i;

    /* SHUFFLE INDICES RANDOMLY */
    shuffle_indices(indices, n);

    DataSplit split;

    /* TRAIN SET: INDICES 0 TO TRAIN_COUNT */
    split.X_train = create_subset(X, indices, 0, train_count);
    split.y_train = create_subset(y, indices, 0, train_count);

    /* TEST SET: INDICES TRAIN_COUNT TO END */
    split.X_test = create_subset(X, indices, train_count, n);
    split.y_test = create_subset(y, indices, train_count, n);

    free(indices);
    
    printf("DATA SPLIT COMPLETE: TRAIN (%d), TEST (%d)\n", train_count, test_count);
    return split;
}