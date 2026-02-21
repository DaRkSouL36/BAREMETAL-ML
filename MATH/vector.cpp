#include "vector.h"
#include <cmath>
#include <iomanip>

using namespace std;

/* CONSTRUCTOR: INITIALIZE WITH GIVEN SIZE AND VALUE */
Vector::Vector(int n, double initial_val) : size(n) 
{
    data.resize(size, initial_val);
}

/* CONSTRUCTOR: INITIALIZE FROM STD::VECTOR */
Vector::Vector(const vector<double>& vec) 
{
    data = vec;
    size = vec.size();
}

/* SIZE ACCESSOR */
int Vector::get_size() const 
{
    return size;
}

/* OPERATOR OVERLOAD FOR ARRAY-LIKE ACCESS */
double& Vector::operator[](int i) 
{
    if (i < 0 || i >= size) 
        throw out_of_range("VECTOR INDEX OUT OF BOUNDS");

    return data[i];
}

double Vector::operator[](int i) const 
{
    if (i < 0 || i >= size) 
        throw out_of_range("VECTOR INDEX OUT OF BOUNDS");
    
    return data[i];
}

/*
 * FUNCTION: DOT PRODUCT
 * MATH: A • B = SUM(A[i] * B[i])
 * MEASURES SIMILARITY OR PROJECTION OF ONE VECTOR ONTO ANOTHER.
 */
double Vector::dot(const Vector& other) const 
{
    if (size != other.size) 
        throw invalid_argument("DOT PRODUCT DIMENSION MISMATCH");
    
    double result = 0.0;
    
    for (int i = 0; i < size; ++i) 
        result += data[i] * other.data[i];
    
    return result;
}

/*
 * FUNCTION: MAGNITUDE (L2 NORM)
 * MATH: ||V|| = SQRT(V • V)
 */
double Vector::magnitude() const 
{
    return sqrt(this->dot(*this));
}

/*
 * FUNCTION: VECTOR ADDITION
 * MATH: C[i] = A[i] + B[i]
 */
Vector Vector::operator+(const Vector& other) const 
{
    if (size != other.size) 
        throw invalid_argument("ADDITION DIMENSION MISMATCH");
    
    Vector res(size);

    for (int i = 0; i < size; ++i) 
        res[i] = data[i] + other.data[i];
    
    return res;
}

/*
 * FUNCTION: VECTOR SUBTRACTION
 * MATH: C[i] = A[i] - B[i]
 * USED HEAVILY IN GRADIENT DESCENT: ERROR = Y_PRED - Y_TRUE
 */
Vector Vector::operator-(const Vector& other) const 
{
    if (size != other.size) 
        throw invalid_argument("SUBTRACTION DIMENSION MISMATCH");
    
    Vector res(size);
    
    for (int i = 0; i < size; ++i) 
        res[i] = data[i] - other.data[i];
    
    return res;
}

/*
 * FUNCTION: SCALAR MULTIPLICATION
 * MATH: C[i] = A[i] * SCALAR
 * USED FOR APPLYING LEARNING RATE: WEIGHTS -= GRADIENT * ALPHA
 */
Vector Vector::operator*(double scalar) const 
{
    Vector res(size);

    for (int i = 0; i < size; ++i) 
        res[i] = data[i] * scalar;
    
    return res;
}

/*
 * FUNCTION: SCALAR DIVISION
 */
Vector Vector::operator/(double scalar) const 
{
    if (scalar == 0) 
        throw runtime_error("DIVISION BY ZERO");
   
    Vector res(size);
    
    for (int i = 0; i < size; ++i) 
        res[i] = data[i] / scalar;
    
    return res;
}

/*
 * FUNCTION: PRINT
 * OUTPUTS VECTOR AS A COLUMN
 */
void Vector::print() const 
{
    cout << "[\n";

    for (int i = 0; i < size; ++i) 
        cout << "  " << setw(10) << setprecision(4) << data[i] << "\n";
    
    cout << "]\n";
}