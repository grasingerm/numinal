#include <iostream>
#include <cassert>
#include <tuple>
#include "assertion_helper.h"
#include "direct_methods.hpp"

using namespace arma;
using namespace std;

int main()
{
    mat::fixed<4,4> A;
    mat L, U;
    bool factor_success;
    
    A   <<    1   <<    1   <<  0   <<  3   <<  endr
        <<    2   <<    1   <<  -1  <<  1   <<  endr
        <<    3   <<    -1  <<  -1  <<  2   <<  endr
        <<  -1    <<    2   <<  3   <<  -1  <<  endr;
    
    cout << "A = " << endl << A << endl;
    
    tie (L, U, factor_success) = lu_doolittle(A);
    
    cout << "L = " << endl << L << endl;
    cout << "U = " << endl << U << endl;
    cout << "L*U = " << endl << L*U << endl;
    
    return 0;
}
