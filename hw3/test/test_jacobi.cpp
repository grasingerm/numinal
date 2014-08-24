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
    
    vec::fixed<4> x, x_exp;
    x_exp << 1 << 2 << -1 << 1;
    
    SECTION("example 7.3.1, pg 437");
    cout << "A = " << endl << A << endl;
    cout << "b = " << endl << b << endl;
    
    bool has_converged;
    tie (x, has_converged) = jacobi(A, b, 1e-3, 13);
    
    cout << "x = " << endl << x << endl;
    
    SUBSECT("testing against textbook values...");
    assert(has_converged);
    for (unsigned int i = 0; i < x.n_rows; i++)
        ASSERT_NEAR(x(i), x_exp(i), 1e-3);
    cout << "... test passed." << endl;
        
    ENDSECT;
    
    SECTION("example 7.3.1, pg 437");
    cout << "A = " << endl << A << endl;
    cout << "b = " << endl << b << endl;
    
    tie (x, has_converged) = gauss_seidel(A, b, 1e-3, 6);
    
    cout << "x = " << endl << x << endl;
    
    SUBSECT("testing against textbook values...");
    assert(has_converged);
    for (unsigned int i = 0; i < x.n_rows; i++)
        ASSERT_NEAR(x(i), x_exp(i), 1e-3);
    cout << "... test passed." << endl;
        
    ENDSECT;
    
    SECTION("testing reorder_for_jacobi_M....");
    mat::fixed<4,4> A_2;
    A_2     <<  -1  <<  11  <<  -1  <<  3   <<  endr
            <<  0   <<  3   <<  -1  <<  8   <<  endr
            <<  10  <<  -1  <<  2   <<  0   <<  endr
            <<  2   <<  -1  <<  10  <<  -1  <<  endr;
            
    vec::fixed<4> b_2;
    b_2 << 25 << 15 << 6 << -11;
    
    cout << "system before reorder" << endl << "A = " << endl << A_2 << endl;
    cout << "b = " << endl << b_2 << endl;
    
    reorder_for_jacobi_M(A_2, b_2);
    
    cout << "system after reorder" << endl << "A = " << endl << A_2 << endl;
    cout << "b = " << endl << b_2 << endl;
    
    for (unsigned int i = 0; i < 4; i++)
    {
        ASSERT_NEAR(b(i), b_2(i), 1e-5);
        for (unsigned int j = 0; j < 4; j++)
            ASSERT_NEAR(A(i,j), A_2(i,j), 1e-5);
    }
    
    cout << "... test passed." << endl;
    ENDSECT;
}
