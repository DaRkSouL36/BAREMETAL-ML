#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" 
{
    #endif

    /*
    * STRUCT: DATASET
    * ---------------
    * REPRESENTS A MATRIX OF DATA LOADED FROM CSV.
    * USED FOR BOTH FEATURES (X) AND TARGETS (Y).
    */
    typedef struct 
    {
        double** data;  /* 2D ARRAY OF DOUBLES */
        int rows;       /* NUMBER OF SAMPLES */
        int cols;       /* NUMBER OF FEATURES */
    } Dataset;

    /*
    * FUNCTION: LOAD_CSV
    * ------------------
    * LOADS A CSV FILE INTO A DATASET STRUCT.
    * ASSUMES NO HEADER ROW IN DATA (OR HANDLES IT EXTERNALLY).
    */
    Dataset* load_csv(const char* filename);

    /*
    * FUNCTION: FREE_DATASET
    * ----------------------
    * FREES MEMORY ALLOCATED FOR THE DATASET.
    */
    void free_dataset(Dataset* dataset);

    #ifdef __cplusplus
}
#endif

#endif