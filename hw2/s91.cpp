#include <armadillo>
#include <iostream>
#include <cassert>
#include <tuple>
#include <cstdlib>

using namespace arma;
using namespace std;

/* interface */
namespace eigen
{
    tuple<vec,double> power_method(const mat&, const vec&, const double, const
        unsigned int);
}

double norm_linf(const mat&, unsigned int&, unsigned int&);


/* main program logic */
using namespace eigen;

int main()
{
    double lambda_1;
    mat::fixed<3,3> A;
    vec::fixed<3> x_o;
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
    
    vec x(x_o); /* initialize x to x_o, iterate three times */
    for (unsigned int iter = 0; iter < 3; iter++)
    {   
        /* perform a single iteration, make sure lambda <> 0 */
        tie (x, lambda_1) = power_method(A, x, 1e-3, 1);
        assert(lambda_1 != 0);
        
        cout << "iteration: " << iter+1 << endl;
        cout << "lambda_max: " << lambda_1 << endl;
        cout << "x: " << endl << x << endl;
        cout << endl;
    }
    
    return 0;
}

/* function implementations */

/**
 * Approximate the dominant eigenvalue and associated eigenvector using power
 * method
 */
tuple<vec,double> eigen::power_method(const mat& A, const vec& x_o, const double tol, 
    const unsigned int max_iterations)
{
    unsigned int row, col;
    vec x(x_o), y;
    double y_p, err;
    
    x /= norm_linf(x, row, col);
    for (unsigned int iteration = 0; iteration < max_iterations; iteration++)
    {
        y = A*x;
        y_p = norm_linf(y, row, col);
        
        /* the eigenvalue for eigenvector x is 0, a new guess must be made */
        if (y_p == 0) break;
        
        err = norm_linf(x - y/y_p, row, col);
        x = y/y_p;
        
        if (err < tol) break;
    }
    
    return tuple<vec,double> (x, y_p);
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
