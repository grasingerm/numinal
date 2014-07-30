#include <armadillo>
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_NEAR(act, approx, tol) \
    if (!act) \
        assert(!approx || abs(approx) < tol); \
    else \
        assert(abs(((act)-(approx))/(act)) < tol)

enum lu_err_code
{
    FACTORIZATION_FAILURE,
    FACTORIZATION_SUCCESS
};

using namespace std;
using namespace arma;

/**
 * Factors a sys of equations into L and U matrices
 * TODO: write a function to perform LU factorization with a permutation matrix
 */
lu_err_code lu_mdoolittle(mat &lsys, mat &L, mat &U)
{
    double m_ji;
    
    /* start L as identity matrix */
    L.eye();
    
    /* copy lsys into U */ /* TODO: is there a more efficent way to do this? */
    for (unsigned int i = 0; i < lsys.n_rows; i++)
        for (unsigned int j = 0; j < lsys.n_cols; j++)
            U(i, j) = lsys(i, j);
    
    for (unsigned int i = 0; i < U.n_rows; i++)
    {
        if (!U(i, i)) return FACTORIZATION_FAILURE;
        /* perform elimination */
        for (unsigned int j = i+1; j < U.n_rows; j++)
        {
            m_ji = U(j, i)/U(i, i);
            U.row(j) -= m_ji * U.row(i);
            L(j, i) = m_ji;
        }
    }
    
    return FACTORIZATION_SUCCESS;
}

/**
 * Factors a sys of equations into U and L matrices; mutates original sys
 */
lu_err_code mlu_mdoolittle(mat &lsys, mat &L)
{
    double m_ji;
    
    /* start L as identity matrix */
    L.eye();
    
    for (unsigned int i = 0; i < lsys.n_rows; i++)
    {
        if (!lsys(i, i)) return FACTORIZATION_FAILURE;
        /* perform elimination */
        for (unsigned int j = i+1; j < lsys.n_rows; j++)
        {
            m_ji = lsys(j, i)/lsys(i, i);
            lsys.row(j) -= m_ji * lsys.row(i);
            L(j, i) = m_ji;
        }
    }
    
    return FACTORIZATION_SUCCESS;
}

/** 
 * Solve system of equations using LU factoriztion
 */
 
vec lu_fb_substitution(mat &L, mat &U, vec &b)
{
    vec y(L.n_rows), x(L.n_rows);   // intense use of heap?
    double sum;

    /* start forward substitution */
    y(0) = b(0);
    
    for (unsigned int i = 0; i < L.n_rows; i++)
    {
        sum = 0;
        for (int j = i-1; j >= 0; j--)
            sum += L(i, j) * y(j);
        y(i) = b(i) - sum;
    }
    
    /* backward substitution */
    x(L.n_rows-1) = y(U.n_rows-1) / U(U.n_rows-1, U.n_cols-1);
    
    for (int i = L.n_rows-2; i >= 0; i--)
    {
        sum = 0;
        for (unsigned int j = i+1; j < U.n_rows; j++)
            sum += U(i, j) * x(j);
        x(i) = (y(i) - sum) / U(i, i);
    }
    
    return x;
}

vec gauss_elimination_pp(mat A, vec b)
{
    vec x(A.n_rows);
    unsigned int row;
    double max, val, sum, mult;

    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        /* find pivot point */
        max = 0;
        for (unsigned int j = i; j < A.n_cols; j++)
        {
            val = fabs(A(j,i));
            if (val > max)
            {
                max = val;
                row = j;
            }
            
            /* sanity check */
            assert(A(j,i) != 0);
        }
        
        /* row interchange */
        if (i != row)
        {
            /* swap rows i and "row" */
            for (unsigned int k = 0; k < A.n_cols; k++)
            {
                val = A(i, k);
                A(i, k) = A(row, k);
                A(row, k) = val;
            }
            
            val = b(i);
            b(i) = b(row);
            b(row) = val;
        }
        
        /* elimination */
        for (unsigned int j = i; j < A.n_cols; j++)
        {
            mult = A(j,i)/A(i,i);
            A.row(j) -= mult * A.row(i);
            b(j) -= b(i);
        } 
    }
    
    /* backward substitution */
    assert(A(A.n_rows-1,A.n_cols-1) != 0);
    x(x.n_rows-1) = b(b.n_rows-1)/A(A.n_rows-1,A.n_cols-1);
    
    cout << A << endl;
    
    for (int i = A.n_rows-2; i >= 0; i--)
    {
        sum = 0;
        for (unsigned int j = i; j < A.n_rows; j++) sum += A(i,j)*x(j);
        x(i) = (b(i)-sum)/A(i,i);
    }
    
    return x;
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
    
    x = gauss_elimination_pp(A,b);
    cout << "solution..." << endl;
    cout << "x = " << endl << x << endl;
    
    vec::fixed<4> x_actual;
    x_actual << 3 << endr << -1 << endr << 0 << endr << 2 << endr;
    
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_actual(i), x(i), 1e-4);

    return 0;
}
