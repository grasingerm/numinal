#include <armadillo>
#include "iterative.hpp"
#include "assertion_helpers.hpp"
#include "assignment_helpers.hpp"
#include <cassert>
#include <tuple>

using namespace arma;
using namespace std;

int main()
{
    mat::fixed<4,4> A;
    A   <<  10  <<  -1  <<  2   <<  0   <<  endr
        <<  -1  <<  11  <<  -1  <<  3   <<  endr
        <<  2   <<  -1  <<  10  <<  -1  <<  endr
        <<  0   <<  3   <<  -1  <<  8   <<  endr;
        
    vec::fixed<4> b;
    b << 6 << 25 << -11 << 15;
    
    vec::fixed<4> x, x_o, x_exp;
    x_o << 0 << 0 << 0 << 0;
    x_exp << 1 << 2 << -1 << 1;
    
    SECTION("example 7.3.1, pg 437 using conjugate gradient w/ steepest descent");
    cout << "A = " << endl << A << endl;
    cout << "b = " << endl << b << endl;
    
    bool has_converged;
    tie (x, has_converged) = conj_grad_steepest_desc(A, b, x_o, 1e-3, 500);
    
    cout << "x = " << endl << x << endl;
    
    SUBSECT("testing against textbook values...");
    assert(has_converged);
    for (unsigned int i = 0; i < x.n_rows; i++)
        ASSERT_NEAR(x(i), x_exp(i), 1e-3);
    cout << "... test passed." << endl;
        
    ENDSECT;
    
    SECTION("example 7.3.1, pg 437 using conjugate gradient w/ steepest descent");
    cout << "A = " << endl << A << endl;
    cout << "b = " << endl << b << endl;
    
    tie (x, has_converged) = conj_grad_steepest_desc(A, b, x_o, 1e-3, 500);
    
    cout << "x = " << endl << x << endl;
    
    SUBSECT("testing against textbook values...");
    assert(has_converged);
    for (unsigned int i = 0; i < x.n_rows; i++)
        ASSERT_NEAR(x(i), x_exp(i), 1e-3);
    cout << "... test passed." << endl;
        
    ENDSECT;
    
    SECTION("example 7.3.3, pg 447 using conjugate gradient w/ steepest descent");
    mat::fixed<3,3> A_3;
    A_3     <<      4       <<      3       <<      0       <<     endr
            <<      3       <<      4       <<      -1      <<     endr
            <<      0       <<      -1      <<      4       <<     endr;
            
    vec::fixed<3> b_3;
    b_3     << 24 << 30 << -24;
    
    vec::fixed<3> x_exp_3, x_o_3;
    x_exp_3 << 3 << 4 << -5;
    x_o_3 << 0 << 0 << 0;
    
    vec x_3;
    
    tie (x_3, has_converged) = conj_grad_steepest_desc(A_3, b_3, x_o_3, 1e-5, 500);
    
    cout << "x = " << endl << x_3 << endl;
    
    SUBSECT("testing against textbook values...");
    assert(has_converged);
    for (unsigned int i = 0; i < x_3.n_rows; i++)
        ASSERT_NEAR(x_3(i), x_exp_3(i), 1e-5);
    cout << "... test passed." << endl;
        
    ENDSECT;
}
