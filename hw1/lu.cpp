#include <armadillo>
#include <iostream>
#include <cassert>
#include <cmath>

enum lu_err_code
{
    FACTORIZATION_FAILURE,
    FACTORIZATION_SUCCESS
};

using namespace std;
using namespace arma;

/**
 * Factors a sys of equations into L and U matrices
 */
lu_err_code lu_mdoolittle(mat &lsys, mat &L, mat &U)
{
    double m_ji;
    
    /* start L as identity matrix */
    L.eye();
    
    /* copy lsys into U */
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
    x(L.n_rows-1) = y(L.n_rows-1) / U(L.n_rows-1, L.n_cols-1);
    
    for (int i = L.n_rows-2; i >= 0; i--)
    {
        sum = 0;
        for (unsigned int j = i+1; j < L.n_rows; j++)
            sum += L(i, j) * x(j);
        x(i) = (y(i) - sum) / L(i, i);
    }
    
    return x;
}

int main(int argc, char* argv[])
{
    mat::fixed<4, 4> A;
    
    A << 1 << 1 << 0 << 3 << endr
      << 2 << 1 << -1 << 1 << endr
      << 3 << -1 << -1 << 2 << endr
      << -1 << 2 << 3 << -1 << endr;
      
    vec::fixed<4> b;
    b << 4 << endr << 1 << endr << -3 << endr << 4 << endr;
      
    mat::fixed<4, 4> L;
    mat::fixed<4, 4> U;
    
    cout << "system of equations" << endl;
    cout << A << endl << b << endl;
    
    if (lu_mdoolittle(A, L, U) == FACTORIZATION_FAILURE)
    {
        cout << "ERROR: Factorization failed." << endl;
        return 1;
    }
    
    cout << "factorization..." << endl;
    cout << "L = " << endl << L << endl;
    cout << "U = " << endl << U << endl;
    
    vec x;
    
    x = lu_fb_substitution(L, U, b);
    cout << "solution..." << endl;
    cout << "x = " << endl << x << endl;

    return 0;
}
