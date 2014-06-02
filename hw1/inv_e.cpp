#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;

int main(int argc, char* argv[])
{
    const double inve = 1.0/exp(1.0);
    double eps, error = 1e9, approx_inve = 0;
    double prev_n_fact = 1, n_fact;
    int n = 0;
    
    cout << "Enter a tolerance in which to approximate 1/e to: ";
    cin >> eps;
    
    while (error >= eps)
    {
        n_fact = (n * prev_n_fact) ? n * prev_n_fact : 1;
        prev_n_fact = n_fact;
        
        approx_inve += pow(-1.0, n)/n_fact;
        error = abs(inve - approx_inve);
        cout << "approximate 1/e " << approx_inve << ", n " << n << ", error "
             << error << endl;
        n++;
    }
    
    cout << "SOLUTION CONVERGED." << endl;
    cout << "approximate 1/e " << approx_inve << ", iterations " << n << ", error "
             << error << endl;

    return 0;
}
