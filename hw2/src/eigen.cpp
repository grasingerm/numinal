#include <cmath>
#include <cstdlib>
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
 * Non-mutating wrapper of Householder transformation @see householder_Mut
 *
 * @param A Matrix to transform
 * @return Transformed matrix
 */
mat householder(const mat& A)
{
    mat A_copy(A);
    householder_Mut(A_copy);
    return A_copy;
}

/**
 * Transform a matrix into a symmetric tridiagonal matrix that is similar to the
 * original matrix @mutator
 *
 * @param A Matrix to transform (by reference)
 */
void householder_Mut(mat& A)
{
    unsigned int n = A.n_rows;
    double q, alpha, rsq; /* @var rsq -> 2r^2 */
    double v_dot_u, sum;
    vec v(n), u(n), z(n);

    /* main iteration loop */
    for (unsigned int k = 0; k < n-2; k++)
    {
        /* sum squares of current column */
        q = 0;
        for (unsigned int j = k+1; j < n; j++) q += A(j,k)*A(j,k);
        
        /* calculate alpha */
        if (A(k+1,k) == 0) alpha = -sqrt(q);
        else alpha = -sqrt(q)*A(k+1,k) / fabs(A(k+1,k));
        
        /* calculate 2r^2 */
        rsq = alpha*alpha - alpha*A(k+1,k);
        
        /* set vector 'v' */
        v(k) = 0;
        v(k+1) = A(k+1,k) - alpha;
        for (unsigned int j = k+2; j < n; j++) v(j) = A(j,k);
        
        /* set vector 'u' */
        for (unsigned int j = k; j < n; j++)
        {
            sum = 0;
            for (unsigned int i = k+1; i < n; i++) sum += A(j,i)*v(i);
            u(j) = 1/rsq * sum;
        }
        
        /* calculate v dot product u */
        v_dot_u = dot(v,u);
        
        /* set vector 'z' */
        for (unsigned int j = k; j < n; j++)
            z(j) = u(j) - v_dot_u / (2*rsq) * v(j);
        
        /* iterative transform of A, A_k+1 = A_k - vz' - zv' */
        for (unsigned int m = k+1; m < n-1; m++)
        {
            for (unsigned int j = m+1; j < n; j++)
            {
                A(j,m) -= v(m)*z(j) + v(j)*z(m);
                A(m,j) = A(j,m);
            }
            A(m,m) -= 2*v(m)*z(m);
        }
        A(n-1,n-1) -= 2*v(n-1)*z(n-1);
        
        /* zero rest of k row/col */
        for (unsigned int j = k+2; j < n; j++) { A(k,j) = 0; A(j,k) = 0; }
        
        A(k+1,k) -= v(k+1)*z(k);
        A(k,k+1) = A(k+1,k);
    }
}

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
