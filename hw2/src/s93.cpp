#include <armadillo>
#include <iostream>
#include <cassert>
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
    /* ===================================================================== */
    /* Example 1 */
    SECTION("Example 1, pg.561");
    
    mat A(4,4), A_exp(4,4);
    
    A <<    4   <<  1   <<  -2  <<  2   <<  endr
      <<    1   <<  2   <<  0   <<  1   <<  endr
      <<    -2  <<  0   <<  3   <<  -2  <<  endr
      <<    2   <<  1   <<  -2  <<  -1  <<  endr;
    
    A_exp   <<    4   <<    -3      <<    0         <<      0       <<  endr
            <<    -3  <<  10./3.    <<  -5./3.      <<      0       <<  endr
            <<  0     <<    -5./3.  <<  -33./25.    <<  68./75.     <<  endr
            <<  0     <<    0       <<  68./75.     <<  149./75.    <<  endr;
    
    SUBSECT("before householder transformation...");        
    cout << "A = " << endl << A << endl;
    householder_Mut(A);
    
    SUBSECT("after householder transformation.");
    cout << "A = " << endl << A << endl;
    
    /* check against values given in textbook */
    ASSERT_ARMA_MATRIX_NEAR(A, A_exp, 1e-4);
    
    ENDSECT;
    /* end of Example 1 */
    /* ===================================================================== */

    /* ===================================================================== */
    /* problem 9.3.1a */
    mat B(3,3);
    
    B   <<  12  <<  10  <<  4   <<  endr
        <<  10  <<  8   <<  -5  <<  endr
        <<  4   <<  -5  <<  3   <<  endr;

    SUBSECT("before householder transformation...");        
    cout << "A = " << endl << B << endl;
    B = householder(B);
    
    SUBSECT("after householder transformation.");
    cout << "A = " << endl << B << endl;
    
    ENDSECT;
    /* end of problem 9.3.1a */
    /* ===================================================================== */

    return 0;
}
