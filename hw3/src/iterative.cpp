#include "iterative.hpp"
#include <cstdlib>
#include <utility>

using namespace arma;

/**
 * Solve a system of equations by the jacobi iterative method
 *
 * @param A Matrix of equations
 * @param b Solutions of equations
 * @param x_o Initial guess at x values
 * @param tol Error tolerance
 * @param max_iter Maximum number of iterations
 * @return tuple(x values, has converged?)
 */
iter_soln_t jacobi
    (const mat& A, const vec& b, vec x_o, const double tol, 
    const unsigned int max_iter)
{
    const unsigned int n = A.n_rows;
    double sum;
    vec x(n);

    for (unsigned int iter = 0; iter < max_iter; iter++)
    {
        for (unsigned int i = 0; i < n; i++)
        {
            sum = 0;
            for (unsigned int j = 0; j < i; j++) sum += A(i,j)*x_o(j);
            for (unsigned int j = i+1; j < n; j++) sum += A(i,j)*x_o(j);
                
            x(i) = (b(i)-sum) / A(i,i);
        }
        
        if (norm(x-x_o, "inf")/norm(x, "inf") < tol) 
            return iter_soln_t (x, true);
            
        x_o = x;
    }
    
    return iter_soln_t(x, false);
}

/**
 * Solve a system of equations by the jacobi iterative method
 *
 * @param A Matrix of equations
 * @param b Solutions of equations
 * @param tol Error tolerance
 * @param max_iter Maximum number of iterations
 * @return tuple(x values, has converged?)
 */
iter_soln_t jacobi
    (const mat& A, const vec& b, const double tol, const unsigned int max_iter)
{
    vec x_o(b.n_rows);
    x_o.zeros();

    return jacobi(A, b, x_o, tol, max_iter);
}

/**
 * Solve a system of equations by the gauss-seidel iterative method
 *
 * @param A Matrix of equations
 * @param b Solutions of equations
 * @param x_o Initial guess at x values
 * @param tol Error tolerance
 * @param max_iter Maximum number of iterations
 * @return tuple(x values, has converged?)
 */
iter_soln_t gauss_seidel
    (const mat& A, const vec& b, vec x_o, const double tol, 
    const unsigned int max_iter)
{
    const unsigned int n = A.n_rows;
    double sum;
    vec x(n);

    for (unsigned int iter = 0; iter < max_iter; iter++)
    {
        for (unsigned int i = 0; i < n; i++)
        {
            sum = 0;
            for (unsigned int j = 0; j < i; j++) sum += A(i,j)*x(j);
            for (unsigned int j = i+1; j < n; j++) sum += A(i,j)*x_o(j);
                
            x(i) = (b(i)-sum) / A(i,i);
        }
        
        if (norm(x-x_o, "inf")/norm(x, "inf") < tol) 
            return iter_soln_t (x, true);
            
        x_o = x;
    }
    
    return iter_soln_t(x, false);
}

/**
 * Solve a system of equations by the gauss-seidel iterative method
 *
 * @param A Matrix of equations
 * @param b Solutions of equations
 * @param tol Error tolerance
 * @param max_iter Maximum number of iterations
 * @return tuple(x values, has converged?)
 */
iter_soln_t gauss_seidel
    (const mat& A, const vec& b, const double tol, const unsigned int max_iter)
{
    vec x_o(b.n_rows);
    x_o.zeros();

    return gauss_seidel(A, b, x_o, tol, max_iter);
}

/**
 * mutates a system of linear equations for jacobi iterative solution method
 * @mutator
 *
 * @param A Matrix of system of equations
 * @param b Solution vector
 */
void reorder_for_jacobi_M(mat& A, vec& b)
{
    unsigned int n = A.n_rows, row;
    double max, abs_val;
    
    for (unsigned int i = 0; i < n; i++)
    {
        /* swap current row for remaining row with max value in same column */
        max = 0;
        for (unsigned int k = 0; k < n; k++)
        {
            abs_val = abs(A(k,i));
            if (abs_val > max)
            {
                max = abs_val;
                row = k;
            }
        }
        
        /* perform swap */
        for (unsigned int j = 0; j < n; j++) std::swap(A(i,j), A(row,j));
        std::swap(b(i), b(row));
    }
}

/**
 * reorder system of linear equations for jacobi iterative solution method
 *
 * @param A Matrix of system of equations
 * @param b Solution vector
 * @return tuple(reordered matrix, reordered solution vector)
 */
std::tuple<mat, vec> reorder_for_jacobi(mat A, vec b)
{
    /* mutate and return copies of A and b */
    reorder_for_jacobi_M(A, b);
    return std::tuple<mat, vec> (A, b);    
}
