#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;

int main(int argc, char* argv[])
{
    double eps, error = 1e9, approx_pi = 0;
    int n = 0;
    
    cout << "Enter a tolerance in which to approximate pi too: ";
    cin >> eps;
    
    while (error >= eps)
    {
        approx_pi += 4 * pow(-1, n) / (2*n + 1);
        error = abs(M_PI - approx_pi);
        cout << "approximate pi " << approx_pi << ", n " << n << ", error "
             << error << endl;
        n++;
    }
    
    cout << "SOLUTION CONVERGED." << endl;
    cout << "approximate pi " << approx_pi << ", iterations " << n << ", error "
             << error << endl;

    return 0;
}
