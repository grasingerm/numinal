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
 */
iterative_solution_t power_method
    (const mat& A, const vec& x_o, const double tol, 
    const unsigned int max_iterations)
{
    bool has_converged = false;
    unsigned int row, col;
    vec x(x_o), y;
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
 */
iterative_solution_t power_method_d2
    (const mat& A, const vec& x_o, const double tol, 
    const unsigned int max_iterations)
{
    bool has_converged = false;
    unsigned int row, col;
    vec x(x_o), y;
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
 */
iterative_solution_t inverse_power_method
    (const mat& A, const vec& x_o, const double tol, 
    const unsigned int max_iterations)
{
    bool has_converged = false;
    vec x(x_o), y;
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

} /* namespace eigen */

/**
 * Calculate the L infinity norm of a matrix and store its location
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
