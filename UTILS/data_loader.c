#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_loader.h"

#define MAX_LINE_LENGTH 10240

/*
 * FUNCTION: LOAD_CSV
 * PURPOSE: READS A CSV FILE AND POPULATES A DATASET STRUCT
 * LOGIC:
 * 1. OPEN FILE
 * 2. COUNT ROWS AND COLUMNS FIRST (TWO-PASS APPROACH)
 * 3. ALLOCATE MEMORY
 * 4. READ DATA AND PARSE FLOATS
 */
Dataset* load_csv(const char* filename) 
{
    FILE* file = fopen(filename, "r");
    
    if (!file) 
    {
        printf("ERROR: COULD NOT OPEN FILE %s\n", filename);
        return NULL;
    }

    /* PASS 1: DETERMINE DIMENSIONS */
    int rows = 0;
    int cols = 0;
    char line[MAX_LINE_LENGTH];
    
    while (fgets(line, sizeof(line), file)) 
    {
        if (rows == 0) 
        {
            /* COUNT COLUMNS IN FIRST ROW */
            char* token = strtok(line, ",");
            
            while (token) 
            {
                ++cols;
                token = strtok(NULL, ",");
            }
        }
        ++rows;
    }

    /* RESET FILE POINTER TO BEGINNING */
    rewind(file);

    /* ALLOCATE MEMORY FOR DATASET STRUCT */
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->rows = rows;
    dataset->cols = cols;
    
    /* ALLOCATE ROW POINTERS */
    dataset->data = (double**)malloc(rows * sizeof(double*));
    
    for (int i = 0; i < rows; ++i) 
        dataset->data[i] = (double*)malloc(cols * sizeof(double));
    

    /* PASS 2: FILL DATA */
    int r = 0;

    while (fgets(line, sizeof(line), file) && r < rows) 
    {
        int c = 0;
        /* STRTOK MODIFIES STRING, SO WE COPY IF NEEDED, BUT HERE LINE IS DISPOSABLE */
        char* token = strtok(line, ",");
        
        while (token && c < cols) 
        {
            dataset->data[r][c] = atof(token);
            token = strtok(NULL, ",");
            ++c;
        }
        ++r;
    }

    fclose(file);
    
    /* LOG SUCCESS */
    printf("LOADED CSV: %s (%d ROWS, %d COLS)\n", filename, rows, cols);
    return dataset;
}

/*
 * FUNCTION: FREE_DATASET
 * PURPOSE: PREVENTS MEMORY LEAKS BY FREEING 2D ARRAYS
 */
void free_dataset(Dataset* dataset) 
{
    if (dataset) 
    {
        for (int i = 0; i < dataset->rows; ++i) 
            free(dataset->data[i]);
        
        free(dataset->data);
        free(dataset);
    }
}