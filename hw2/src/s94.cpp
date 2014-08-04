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
    unsigned int index;
    vector<double> lambdas;
    bool success;

    /* ===================================================================== */
    /* 9.4.2a */
    SECTION("problem 9.4.2a");
    
    vec a(3);
    a << 2 << -1 << 3;
    
    vec b(3);
    b << 0 << -1 << -2;
    
    SUBSECT("before iterations");
    cout << "a = " << endl << a << endl << "b = " << endl << b << endl;
    
    tie (lambdas, index, success) = qr_method_Mut(a, b, 1e-9, 2);
    
    SUBSECT("after iterations");
    cout << "a = " << endl << a << endl << "b = " << endl << b << endl;
    
    ENDSECT;
    /* end of problem 9.4.2a */
    /* ===================================================================== */

    /* ===================================================================== */
    /* 9.4.4a */
    SECTION("problem 9.4.4a");
    
    a << 2 << -1 << 3;
    b << 0 << -1 << -2;
    
    cout << "a = " << endl << a << endl << "b = " << endl << b << endl;
    tie (lambdas, index, success) = qr_method(a, b, 1e-5, 1e6);
    
    cout << "eigenvalues: ";
    for (auto lambda : lambdas) cout << lambda << " ";
    cout << endl;
    cout << "success?: " << ((success) ? "yes" : "no") << endl;
    
    ENDSECT;
    /* end of problem 9.4.4a */
    /* ===================================================================== */

    return 0;
}
