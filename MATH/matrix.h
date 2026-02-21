#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>

using namespace std;

class Matrix 
{
    private:
        vector<vector<double>> data;
        int rows;
        int cols;

    public:
        /* CONSTRUCTORS */
        Matrix(int r, int c, double initial_val = 0.0);
        Matrix(const vector<vector<double>>& grid);

        /* ACCESSORS */
        int get_rows() const;
        int get_cols() const;
        double& operator()(int r, int c);
        double operator()(int r, int c) const;

        /* CORE OPERATIONS */
        Matrix transpose() const;
        Matrix dot(const Matrix& other) const; /* MATRIX MULTIPLICATION */
        Matrix inverse() const;                /* GAUSS-JORDAN ELIMINATION */
        
        /* ELEMENT-WISE OPERATIONS */
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double scalar) const;

        /* UTILITIES */
        void print() const;
        static Matrix identity(int n);
};

#endif