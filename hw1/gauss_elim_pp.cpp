#include <armadillo>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_NEAR(act, approx, tol) \
    if (!(act)) \
        assert(!(approx) || abs(approx) < (tol)); \
    else \
        assert(fabs(((act)-(approx))/(act)) < (tol))

using namespace std;
using namespace arma;

/**
 * Gaussian elimination with partial pivoting
 *
 * @param A Linear system of equations
 * @param b Linear system solutions
 */
void gauss_elim_wpp_Mut(mat &A, vec &b)
{
    unsigned int row;
    double max, val, mult;

    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        /* find pivot point */
        row = i;
        max = -1; // arbitrary number less than zero
        for (unsigned int j = i; j < A.n_cols; j++)
        {
            val = fabs(A(j,i));
            if (val > max)
            {
                max = val;
                row = j;
            }
            
            /* sanity check */
            assert(A(row,i) != 0);
        }
        
        /* row interchange */
        if (i != row)
            /* swap rows i and "row" */
            for (unsigned int k = 0; k < A.n_cols; k++)
                std::swap(A(i,k), A(row,k));
        std::swap(b(i), b(row));
        
        /* elimination */
        for (unsigned int j = i+1; j < A.n_cols; j++)
        {
            mult = A(j,i)/A(i,i);
            A.row(j) -= mult * A.row(i);
            b(j) -= mult * b(i);
        }
    }
}

/**
 * Given an upper triangular matrix and a solution vector solve for x
 *
 * @param A Upper triangular matrix
 * @param b Solution vector
 * @return x X values
 */
vec backward_subs(const mat& A, const vec& b)
{
    /* TODO: error checking? */
    double sum;
    vec x(b.n_rows);
    x.zeros();
    
    x(x.n_rows-1) = b(b.n_rows-1)/A(A.n_rows-1,A.n_cols-1);
    for (int i = A.n_rows-2; i >= 0; i--)
    {
        sum = 0;
        for (unsigned int j = i+1; j < A.n_rows; j++) sum += A(i,j)*x(j);
        x(i) = (b(i)-sum)/A(i,i);
    }
    
    return x;
}

/**
 * Wrapper to solve a linear system of equations with gaussian elimination and
 * partial pivoting
 *
 * @param A Linear system of equations
 * @param b Linear system of solutions
 * @return x X values
 */
vec solve_gauss_elim_wpp(mat A, vec b)
{
    gauss_elim_wpp_Mut(A, b);
    return backward_subs(A, b);
}



int main(int argc, char* argv[])
{
    /* example 1, section 6.5, pg 391-392 */
    cout << "Example 1, section 6.5, pg 391-392" << endl;
    
    mat::fixed<4, 4> A;
    
    A << 1 << 1 << 0 << 3 << endr
      << 2 << 1 << -1 << 1 << endr
      << 3 << -1 << -1 << 2 << endr
      << -1 << 2 << 3 << -1 << endr;
      
    vec::fixed<4> b;
    b << 8 << endr << 7 << endr << 14 << endr << -7 << endr;
    
    cout << "system of equations" << endl;
    cout << A << endl << b << endl;
    
    vec x;
    
    x = solve_gauss_elim_wpp(A,b);
    cout << "solution..." << endl;
    cout << "x = " << endl << x << endl;
    
    vec::fixed<4> x_actual;
    x_actual << 3 << endr << -1 << endr << 0 << endr << 2 << endr;
    
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_actual(i), x(i), 1e-4);

    return 0;
}
