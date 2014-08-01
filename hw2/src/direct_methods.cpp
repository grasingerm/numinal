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
            /* TODO: implement real error checking */
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

/**
 * Factors a sys of equations into U and L matrices; mutates original sys
 * @mutator
 * 
 * @param A System of linear equations, factors into U
 * @return L Lower diagonal matrix
 */
std::array<mat,2> lu_mdoolittle_Mut(mat A)
{
    double m_ji;
    
    /* start L as identity matrix */
    L.eye();
    
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        // if (!A(i, i)) return FACTORIZATION_FAILURE;
        /* perform elimination */
        for (unsigned int j = i+1; j < A.n_rows; j++)
        {
            m_ji = A(j, i)/A(i, i);
            A.row(j) -= m_ji * A.row(i);
            L(j, i) = m_ji;
        }
    }
    
    return std::array<L,U>;
}

/**
 * Factors a sys of equations into U and L matrices; mutates original sys
 * 
 * @param A System of linear equations, factors into U
 * @return (Upper diagonal matrix, Lower diagonal matrix)
 */
mat lu_mdoolittle_Mut(mat &A)
{
    double m_ji;
    
    /* start L as identity matrix */
    L.eye();
    
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        // if (!A(i, i)) return FACTORIZATION_FAILURE;
        /* perform elimination */
        for (unsigned int j = i+1; j < A.n_rows; j++)
        {
            m_ji = A(j, i)/A(i, i);
            A.row(j) -= m_ji * A.row(i);
            L(j, i) = m_ji;
        }
    }
    
    return L;
}

/** 
 * Solve system of equations using LU factoriztion
 */
 
vec lu_fb_subs(const mat &L, const mat &U, const vec &b)
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
