#include "direct_methods.hpp"
#include <algorithm>
#include <armadillo>
#include <cassert>

// #define NDEBUG

using namespace arma;

/**
 * Gaussian elimination with partial pivoting
 *
 * @param A Linear system of equations
 * @param b Linear system solutions
 */
bool gauss_elim_wpp_Mut(mat &A, vec &b)
{
    unsigned int row, n = A.n_rows; /* check for square matrix */
    double max, val, mult;

    for (unsigned int i = 0; i < n; i++)
    {
        /* find pivot point */
        row = i;
        max = -1; // arbitrary number less than zero
        for (unsigned int j = i; j < n; j++)
        {
            val = fabs(A(j,i));
            if (val > max)
            {
                max = val;
                row = j;
            }
            
            /* no unique solution exists */
            if (A(row,i) == 0) return false;
        }
        
        /* row interchange */
        if (i != row)
            /* swap rows i and "row" */
            for (unsigned int k = 0; k < n; k++)
                std::swap(A(i,k), A(row,k));
        std::swap(b(i), b(row));
        
        /* elimination */
        for (unsigned int j = i+1; j < n; j++)
        {
            mult = A(j,i)/A(i,i);
            A.row(j) -= mult * A.row(i);
            b(j) -= mult * b(i);
        }
    }
    
    /* was gaussian elimination successful? */
    return (A(n-1,n-1) != 0) ? true : false;
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
gauss_elim_t solve_gauss_elim_wpp(mat A, vec b)
{
    bool gesuccess = gauss_elim_wpp_Mut(A, b);
    if (gesuccess) return gauss_elim_t (backward_subs(A, b), gesuccess);
 
    vec empty;   
    return gauss_elim_t (empty, gesuccess);
}

/**
 * Factors a sys of equations into U and L matrices
 *
 * @param A System of linear equations
 * @return (Lower diagonal matrix, Upper diagonal matrix)
 */
lu_factor_t lu_doolittle(const mat &A)
{
    unsigned int n = A.n_rows; // TODO: error check/report if n != m
    mat L(n,n), U(n,n);
    double sum;
    
    /* start L as identity matrix */
    L.eye();
    U.zeros();
    
    /* Select L11 and U11 satsifying L11U11 = A11 */
    if (A(0,0) == 0) return lu_factor_t (L, U, false);
    U(0,0) = A(0,0);
    
    for (unsigned int i = 1; i < n; i++) 
    {
        U(0,i) = A(0,i);
        L(i,0) = A(i,0)/U(0,0);
    }
    
    for (unsigned int i = 1; i < n-1; i++)
    {
        // if Aii - sum(Lik * Uki) == 0 -> impossible
        //if (A(i, i) == 0) return lu_factor_t (L, U, false);
        
        for (unsigned int j = i; j < n; j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < i; k++) sum += L(i,k)*U(k,j);
            U(i,j) = A(i,j) - sum;
            
            sum = 0;
            for (unsigned int k = 0; k < i; k++) sum += L(j,k)*U(k,i);
            L(j,i) = (A(j,i)-sum)/U(i,i);
        }
    }
    
    sum = 0;
    for (unsigned int i = 0; i < n-1; i++) sum += L(n-1,i)*U(i,n-1);
    U(n-1,n-1) = A(n-1,n-1) - sum;
    
    return lu_factor_t (L, U, true);
}

/** 
 * Solve system of equations using LU factoriztion
 *
 * @param L Lower diagonal matrix of decomposition
 * @param U Upper diagonal matrix of decomposition
 * @param b Solution vector of system of equations
 * @return x X values
 */
vec lu_fb_subs(const mat& L, const mat& U, const vec& b)
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
