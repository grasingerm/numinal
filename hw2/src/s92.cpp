#include <armadillo>
#include <iostream>
#include <cassert>
#include <tuple>
#include <cstdlib>
#include <vector>

#include "eigen.hpp"
#include "assertion_helper.h"
#include "assignment_helpers.hpp"

using namespace arma;
using namespace std;

/* main program logic */
using namespace eigen;

int main()
{
    double err, lambda;
    bool has_converged;
    mat::fixed<3,3> A;
    vec::fixed<3> x_o, x;
    
    /* ===================================================================== */
    /* Example 1 */
    SECTION("Example 1, pg.561");
    
    vector<double> lambda_expect = { 10.0, 7.2, 6.5, 6.230769, 6.111, 6.054546,
        6.027027, 6.013453, 6.006711, 6.003352, 6.001675, 6.000837 };
    vector<double> x2_expect = { 0.8, 0.75, 0.7307659, 0.7222, 0.718182,
        0.716216, 0.715247, 0.714765, 0.714525, 0.714405, 0.714346, 0.714316 };
    
    A   <<  -4   <<  14   <<  0   <<  endr
        <<  -5   <<  13   <<  0   <<  endr
        <<  -1   <<  0   <<  2   <<  endr;
    
    x_o << 1 << 1 << 1;
    
    cout << "A = " << endl << A << endl
         << "x_o = " << endl << x_o << endl << endl;
         
    SUBSECT("regular power method");
    x = x_o; /* initialize x to x_o, iterate three times */
    for (unsigned int iter = 0; iter < 12; iter++)
    {   
        /* perform a single iteration, make sure lambda <> 0 */
        tie (x, lambda, err, has_converged) = power_method(A, x, 1e-7, 1);
        assert(lambda != 0);
        
        /* check values against those given in the textbook */
        ASSERT_NEAR(lambda, lambda_expect[iter], 1e-3);
        ASSERT_NEAR(x(1), x2_expect[iter], 1e-3);
        
        cout << "iteration: " << iter+1 << endl;
        cout << "err: " << err << endl;
        cout << "lambda_max: " << lambda << endl;
        cout << "x: " << endl << x << endl;
        cout << endl;
    }
    
    SUBSECT("power method with Aitken's delta squared procedure");
    /* check the power method with Aitken's delta squared procedure */
    x = x_o; /* initialize x to x_o, iterate three times */
    /* perform a single iteration, make sure lambda <> 0 */
    tie (x, lambda, err, has_converged) = power_method_d2(A, x, 1e-7, 8);
    assert(lambda != 0);
    
    cout << "iteration: " << 7 << endl;
    cout << "err: " << err << endl;
    cout << "lambda_max: " << lambda << endl;
    cout << "x: " << endl << x << endl;
    cout << endl;
    
    ENDSECT;
    /* end of Example 1 */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* Example 3 */
    SECTION("Example 3, pg. 567");
    
    lambda_expect = { 6.183183, 6.017244, 6.001719, 6.000175, 6.000021, 6.000005 };
    x2_expect = { 0.720727, 0.715518, 0.714409, 0.714298, 0.714287, 0.714286};
    
    A   <<  -4   <<  14   <<  0   <<  endr
        <<  -5   <<  13   <<  0   <<  endr
        <<  -1   <<  0   <<  2   <<  endr;
    
    x_o << 1 << 1 << 1;
    
    cout << "A = " << endl << A << endl
         << "x_o = " << endl << x_o << endl << endl;
         
    for (unsigned int iter = 0; iter < 6; iter++)
    {   
        /* perform n iterations, make sure lambda <> 0 */
        tie (x, lambda, err, has_converged) = 
            inverse_power_method(A, x_o, 1e-9, iter+1);
        assert(lambda != 0);
        
        cout << "iteration: " << iter+1 << endl;
        cout << "err: " << err << endl;
        cout << "lambda: " << lambda << endl;
        cout << "x: " << endl << x << endl;
        cout << endl;
        
        /* check values against those given in the textbook */
        ASSERT_NEAR(lambda, lambda_expect[iter], 1e-2);
        ASSERT_NEAR(x(1), x2_expect[iter], 1e-2); /* within 1% ... */
    }
    
    ENDSECT;
    /* end of Example 3 */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* Example 4 */
    SECTION("Example 4, pg. 569");
    
    lambda = 6;
    
    A   <<  -4   <<  -1   <<  1    <<  endr
        <<  -1   <<  3    <<  -2   <<  endr
        <<  1    <<  -2   <<  3    <<  endr;
    
    vec v(2);
    v << 1 << -1;
    
    x_o << 1 << -1 << 1;
    
    cout << "A = " << endl << A << endl
         << "x_o = " << endl << x_o << endl
         << "v = " << endl << v << endl;
         
    /* perform n iterations, make sure lambda <> 0 */
    tie (x, lambda, err, has_converged) = 
        wielandt_deflation(A, x_o, v, lambda, 1e-4, 25);
    assert(lambda != 0);
    
    cout << "has converged?: " << ((has_converged) ? "yes" : "no") << endl;
    cout << "err: " << err << endl;
    cout << "lambda: " << lambda << endl;
    cout << "x: " << endl << x << endl;
    cout << endl;
    
    /* check values against those given in the textbook */
    ASSERT_NEAR(x(0), -2, 1e-4);
    ASSERT_NEAR(x(1), -1, 1e-4);
    ASSERT_NEAR(x(2), 1, 1e-4);
    ASSERT_NEAR(lambda, 3, 1e-4);
    
    ENDSECT;
    /* end of Example 4 */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* problem 9.2.1a */
    SECTION("problem 9.2.1a");
    
    double lambda_power;
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
    
    cout << "A = " << endl << A << endl
         << "x = " << endl << x_o << endl << endl;
    
    x = x_o;
    for (unsigned int iter = 0; iter < 3; iter++)
    {   
        /* perform a single iteration, make sure lambda <> 0 */
        tie (x, lambda_power, err, has_converged) = power_method(A, x, 1e-9, 1);
        assert(lambda_power != 0);
        
        cout << "iteration: " << iter+1 << endl;
        cout << "err: " << err << endl;
        cout << "lambda_max: " << lambda_power << endl;
        cout << "x: " << endl << x << endl;
        cout << endl;
    }
    
    ENDSECT;
    /* end of problem 9.2.1a */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* problem 9.2.3a */
    SECTION("problem 9.2.3a");
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
    
    for (unsigned int iter = 0; iter < 3; iter++)
    {   
        /* perform a n iterations, make sure lambda <> 0 */
        tie (x, lambda, err, has_converged) = 
            inverse_power_method(A, x_o, 1e-9, iter+1);
        assert(lambda != 0);
        
        cout << "iteration: " << iter+1 << endl;
        cout << "err: " << err << endl;
        cout << "lambda: " << lambda << endl;
        cout << "x: " << endl << x << endl;
        cout << endl;
    }
    
    ENDSECT;
    /* end of problem 9.2.3a */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* problem 9.2.7a */
    SECTION("problem 9.2.7a");
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
         
    /* perform 25 iterations or when error < 1e-4, make sure lambda <> 0 */
    tie (x, lambda_power, err, has_converged) = power_method(A, x_o, 1e-4, 25);
    assert(lambda_power != 0);
    
    cout << "has converged?: " << ((has_converged) ? "yes" : "no") << endl;
    cout << "err: " << err << endl;
    cout << "lambda_max: " << lambda_power << endl;
    cout << "x: " << endl << x << endl;
    cout << endl;
    
    ENDSECT;
    /* end of problem 9.2.7a */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* problem 9.2.9a */
    SECTION("problem 9.2.9a");
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
         
    /* perform 25 iterations or when error < 1e-4, make sure lambda <> 0 */
    tie (x, lambda, err, has_converged) = 
        inverse_power_method(A, x_o, 1e-4, 25);
    assert(lambda != 0);
    
    cout << "has converged?: " << ((has_converged) ? "yes" : "no") << endl;
    cout << "err: " << err << endl;
    cout << "lambda: " << lambda << endl;
    cout << "x: " << endl << x << endl;
    cout << endl;
    
    ENDSECT;
    /* end of problem 9.2.9a */
    /* ===================================================================== */
    
    /* ===================================================================== */
    /* problem 9.2.13a */
    SECTION("problem 9.2.13a");
    
    A   <<  2   <<  1   <<  1   <<  endr
        <<  1   <<  2   <<  1   <<  endr
        <<  1   <<  1   <<  2   <<  endr;
    
    x_o << 1 << -1 << 2;
    
    /* a vector x in the set of Real numbers of an order one less than n */
    vec x_Rminus1(2);
    x_Rminus1 << 1 << 1;
    
    double approx_lambda = lambda_power; /* use results from power method */
         
    cout << "A = " << endl << A << endl
         << "v = " << endl << x_o << endl
         << "x = " << endl << x_Rminus1 << endl;
    
    /* perform 25 iterations or when error < 1e-4, make sure lambda <> 0 */
    tie (x, lambda, err, has_converged) = 
        wielandt_deflation(A, x_o, x_Rminus1, approx_lambda, 1e-4, 25);
    assert(lambda != 0);
    
    cout << "has converged?: " << ((has_converged) ? "yes" : "no") << endl;
    cout << "err: " << err << endl;
    cout << "2nd most dominant lambda: " << lambda << endl;
    cout << "x: " << endl << x << endl;
    cout << endl;
    
    ENDSECT;
    /* end of problem 9.2.13a */
    /* ===================================================================== */
    
    return 0;
}
