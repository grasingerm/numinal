#include "eigen.hpp"
#include "direct_methods.hpp"

using namespace arma;

/* interfaces for file specific functions */
double norm_linf(const mat&, unsigned int&, unsigned int&);

/* implementions */
namespace eigen
{

/**
 * Approximate the dominant eigenvalue and associated eigenvector using power
 * method
 *
 * @param A N x N matrix
 * @param x Initial guess at eigenvector
 * @param tol Error tolerance
 * @param max_iterations Maximum number of iterations to perform
 * @return Approximate eigenvector, approximate eigenvalue, error, has converged
 */
iterative_solution_t power_method
    (const mat& A, vec x, const double tol, const unsigned int max_iterations)
{
    bool has_converged = false;
    unsigned int row, col;
    vec y;
    double y_p, err;
    
    norm_linf(x, row, col);
    x /= x(row);
    for (unsigned int iteration = 0; iteration < max_iterations; iteration++)
    {
        y = A*x;
        
        /* find p so that |y_p| = ||y||_inf */
        norm_linf(y, row, col);
        y_p = y(row);
        
        err = norm_linf(x - y/y_p, row, col);
        x = y/y_p;
        
        has_converged = (err < tol) ? true : false;
        if (has_converged) break;
    }
    
    return iterative_solution_t (x, y_p, err, has_converged);
}

/**
 * Approximate the dominant eigenvalue and associated eigenvector using power
 * method and Aitken's delta squared procedure
 *
 * @param A N x N matrix
 * @param x Initial guess at eigenvector
 * @param tol Error tolerance
 * @param max_iterations Maximum number of iterations to perform
 * @return Approximate eigenvector, approximate eigenvalue, error, has converged
 */
iterative_solution_t power_method_d2
    (const mat& A, vec x, const double tol, const unsigned int max_iterations)
{
    bool has_converged = false;
    unsigned int row, col;
    vec y;
    double y_p, err, mu_0 = 0.0, mu_1 = 0.0, mu_hat;
    
    norm_linf(x, row, col);
    x /= x(row);
    for (unsigned int iteration = 0; iteration < max_iterations; iteration++)
    {
        y = A*x;
        
        /* find p so that |y_p| = ||y||_inf */
        norm_linf(y, row, col);
        y_p = y(row);
        
        mu_hat = mu_0 - (mu_1-mu_0)*(mu_1-mu_0)/(y_p - 2*mu_1 + mu_0);
        
        err = norm_linf(x - y/y_p, row, col);
        x = y/y_p;
        
        has_converged = (err < tol) ? true : false;
        if (has_converged) break;
        
        mu_0 = mu_1;
        mu_1 = y_p;
    }
    
    return iterative_solution_t (x, mu_hat, err, has_converged);
}

/**
 * Approximate an eigenvalue and an associated eigenvector
 *
 * @param A N x N matrix
 * @param x Initial guess at eigenvector
 * @param tol Error tolerance
 * @param max_iterations Maximum number of iterations to perform
 * @return Approximate eigenvector, approximate eigenvalue, error, has converged
 */
iterative_solution_t inverse_power_method
    (const mat& A, vec x, const double tol, const unsigned int max_iterations)
{
    bool has_converged = false;
    vec y;
    mat result(1,1);
    unsigned int row, col;
    double y_p, err;
    double q;
    
    mat I(A.n_rows, A.n_cols);
    I.eye();
    
    result = (x.t() * A * x / (x.t() * x)); /* x'Ax/x'x */
    q = result(0,0);
    
    /* x/x_p */
    norm_linf(x, row, col);
    x /= x(row);
    for (unsigned int iteration = 0; iteration < max_iterations; iteration++)
    {
        y = solve_gauss_elim_wpp(A - q*I, x);
        
        /* find p so that |y_p| = ||y||_inf */
        norm_linf(y, row, col);
        y_p = y(row);
        
        err = norm_linf(x - y/y_p, row, col);
        x = y/y_p;
        
        has_converged = (err < tol) ? true : false;
        if (has_converged) break;
    }
    
    return iterative_solution_t (x, 1/y_p + q, err, has_converged);
}

/**
 * Approximate an eigenvalue and an associated eigenvector
 *
 * @param A N x N matrix
 * @param x Initial guess at eigenvector
 * @param lambda Initial guess at corresponding eigenvalue
 * @param tol Error tolerance
 * @param max_iterations Maximum number of iterations to perform
 * @return Approximate eigenvector, approximate eigenvalue, error, has converged
 */
iterative_solution_t wielandt_deflation
    (const mat& A, const vec& v, const vec& x, const double lambda, 
    const double tol, const unsigned int max_iterations)
{
    bool has_converged;
    double mu, err, sum;
    vec w, u(v.n_rows);
    unsigned int row, col;
    mat B(A.n_rows-1, A.n_cols-1);
    
    norm_linf(v, row, col);
    
    if (row != 0)
        for (unsigned int k = 0; k < row; k++)
            for (unsigned int j = 0; j < row; j++)
                B(k,j) = A(k,j) - v(k)/v(row) * A(row,j);
                
    if (row != 0 && row != v.n_rows-1)
        for (unsigned int k = row; k < v.n_rows-1; k++)
            for (unsigned int j = 0; j < row; j++)
            {
                B(k,j) = A(k+1,j) - v(k+1)/v(row) * A(row,j);
                B(j,k) = A(j,k+1) - v(j)/v(row) * A(row,k+1);
            }
            
    if (row != v.n_rows-1)
        for (unsigned int k = row; k < v.n_rows-1; k++)
            for (unsigned int j = row; j < v.n_rows-1; j++)
                B(k,j) = A(k+1,j+1) - v(k+1)/v(row)*A(row,j+1);
    
    /* this line will need changed if iterative_solution_t changes */      
    std::tie (w, mu, err, has_converged) =
         power_method(B, x, tol, max_iterations);
    
    /* TODO: what is the purpose of this step? */     
    w(row) = 0;
    
    /* calculate this once as it does not change in inner loop */
    sum = 0;
    for (unsigned int i = 0; i < w.n_rows; i++) sum += A(row,i)*w(i);
         
    for (unsigned int k = 0; k < u.n_rows; k++)
        u(k) = (mu - lambda)*w(k) + sum * v(k)/v(row);
        
    return iterative_solution_t (u, mu, err, has_converged);
}

} /* namespace eigen */

/**
 * Calculate the L infinity norm of a matrix and store its location
 *
 * @param A Matrix or vector
 * @param row Integer to store row number of location of the infinity norm
 * @param col Integer to store col number of location of the infinity norm
 * @return L infinity norm
 */
double norm_linf(const mat& A, unsigned int& row, unsigned int& col)
{
    double abs_val;
    double max = 0;
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        for (unsigned int j = 0; j < A.n_cols; j++)
        {
            abs_val = fabs(A(i,j));
            if (abs_val > max)
            {
                max = abs_val;
                row = i;
                col = j;
            }
        }
    }
    
    return max;
}
