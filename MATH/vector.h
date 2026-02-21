#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <iostream>
#include <stdexcept>

using namespace std;

/*
 * CLASS: VECTOR
 * -------------
 * REPRESENTS A MATHEMATICAL 1D TENSOR (COLUMN VECTOR BY DEFAULT).
 * ESSENTIAL FOR WEIGHT VECTORS IN GRADIENT DESCENT.
 */
class Vector 
{
    private:
        vector<double> data;
        int size;

    public:
        /* CONSTRUCTORS */
        Vector(int n, double initial_val = 0.0);
        Vector(const vector<double>& vec);

        /* ACCESSORS */
        int get_size() const;
        double& operator[](int i);
        double operator[](int i) const;

        /* CORE OPERATIONS */
        double dot(const Vector& other) const;    /* DOT PRODUCT (INNER PRODUCT) */
        double magnitude() const;                 /* L2 NORM (EUCLIDEAN LENGTH) */

        /* ELEMENT-WISE MATHEMATICS */
        Vector operator+(const Vector& other) const;
        Vector operator-(const Vector& other) const;
        Vector operator*(double scalar) const;
        Vector operator/(double scalar) const;

        /* UTILITIES */
        void print() const;
};

#endif