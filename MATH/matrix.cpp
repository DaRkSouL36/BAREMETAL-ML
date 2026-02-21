#include "matrix.h"
#include <cmath>
#include <iomanip>

using namespace std;

/* CONSTRUCTOR: INIT WITH ZERO OR VALUE */
Matrix::Matrix(int r, int c, double initial_val) : rows(r), cols(c) 
{
    data.resize(rows, vector<double>(cols, initial_val));
}

/* CONSTRUCTOR: INIT FROM VECTOR OF VECTORS */
Matrix::Matrix(const vector<vector<double>>& grid) 
{
    rows = grid.size();
    cols = (rows > 0) ? grid[0].size() : 0;
    data = grid;
}

int Matrix::get_rows() const { return rows; }
int Matrix::get_cols() const { return cols; }

/* OPERATOR OVERLOAD FOR DIRECT ACCESS M(row, col) */
double& Matrix::operator()(int r, int c) 
{
    if (r < 0 || r >= rows || c < 0 || c >= cols) 
        throw out_of_range("INDEX OUT OF BOUNDS");
    
    return data[r][c];
}

double Matrix::operator()(int r, int c) const 
{
    if (r < 0 || r >= rows || c < 0 || c >= cols) 
        throw out_of_range("INDEX OUT OF BOUNDS");
    
    return data[r][c];
}

/*
 * FUNCTION: TRANSPOSE
 * MATH: B[i][j] = A[j][i]
 */
Matrix Matrix::transpose() const 
{
    Matrix res(cols, rows);

    for (int i = 0; i < rows; ++i) 
    {
        for (int j = 0; j < cols; ++j)
            res(j, i) = data[i][j];
    }

    return res;
}

/*
 * FUNCTION: DOT (MATRIX MULTIPLICATION)
 * MATH: C[i][j] = SUM(A[i][k] * B[k][j])
 * REQUIREMENT: A_COLS MUST EQUAL B_ROWS
 */
Matrix Matrix::dot(const Matrix& other) const 
{
    if (cols != other.rows) 
        throw invalid_argument("DIMENSION MISMATCH FOR DOT PRODUCT");
    
    Matrix res(rows, other.cols);
    
    for (int i = 0; i < rows; ++i) 
    {
        for (int j = 0; j < other.cols; ++j) 
        {
            for (int k = 0; k < cols; ++k) 
                res(i, j) += data[i][k] * other(k, j);
        }
    }

    return res;
}

/*
 * FUNCTION: INVERSE
 * METHOD: GAUSS-JORDAN ELIMINATION WITH PARTIAL PIVOTING
 * COMPLEXITY: O(N^3)
 * REQUIRED FOR: NORMAL EQUATION
 */
Matrix Matrix::inverse() const 
{
    if (rows != cols) 
        throw invalid_argument("MATRIX MUST BE SQUARE TO INVERT");
    
    int n = rows;
    Matrix aug(n, 2 * n);

    /* CREATE AUGMENTED MATRIX [A | I] */
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
            aug(i, j) = data[i][j];
        
        aug(i, i + n) = 1.0;
    }

    /* GAUSS-JORDAN ELIMINATION */
    for (int i = 0; i < n; ++i) 
    {
        /* PIVOTING: FIND MAX ELEMENT IN CURRENT COLUMN */
        int pivot = i;
        
        for (int j = i + 1; j < n; ++j) 
        {
            if (abs(aug(j, i)) > abs(aug(pivot, i))) 
                pivot = j;
        }

        /* SWAP ROWS */
        swap(aug.data[i], aug.data[pivot]);

        /* CHECK SINGULARITY */
        if (abs(aug(i, i)) < 1e-9) 
            throw runtime_error("MATRIX IS SINGULAR");

        /* NORMALIZE PIVOT ROW */
        double divisor = aug(i, i);

        for (int j = 0; j < 2 * n; ++j) 
            aug(i, j) /= divisor;

        /* ELIMINATE OTHER ROWS */
        for (int k = 0; k < n; ++k) 
        {
            if (k != i) 
            {
                double factor = aug(k, i);

                for (int j = 0; j < 2 * n; ++j) 
                    aug(k, j) -= factor * aug(i, j);
            }
        }
    }

    /* EXTRACT INVERSE FROM RIGHT HALF */
    Matrix inv(n, n);
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
            inv(i, j) = aug(i, j + n);
        
    }

    return inv;
}

Matrix Matrix::operator+(const Matrix& other) const 
{
    if (rows != other.rows || cols != other.cols) 
        throw invalid_argument("DIMENSION MISMATCH (+)");
    
    Matrix res(rows, cols);
    
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j) 
            res(i, j) = data[i][j] + other(i, j);
    }
    
    return res;
}

Matrix Matrix::operator-(const Matrix& other) const 
{
    if (rows != other.rows || cols != other.cols) 
        throw invalid_argument("DIMENSION MISMATCH (-)");
    
    Matrix res(rows, cols);
    
    for (int i = 0; i < rows; ++i)
    {    
        for (int j = 0; j < cols; ++j) 
            res(i, j) = data[i][j] - other(i, j);
    }

    return res;
}

Matrix Matrix::operator*(double scalar) const 
{
    Matrix res(rows, cols);
    
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j) 
            res(i, j) = data[i][j] * scalar;
    }
    
    return res;
}

void Matrix::print() const 
{
    for (int i = 0; i < rows; ++i) 
    {
        for (int j = 0; j < cols; ++j) 
            cout << setw(10) << setprecision(4) << data[i][j] << " ";
        
        cout << "\n";
    }
}