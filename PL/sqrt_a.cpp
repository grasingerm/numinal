#include <iostream>
#include <cmath>

using namespace std;

int main(int argc, char* argv[])
{
    const int max_iter = 1000;
    int iter = 0;

    double a;
    double guess;
    double eps;
    double x_n, x_m, temp;

    cout << "Enter a value to take the square root of: ";
    cin >> a;
    
    cout << "Enter an initial guess: ";
    cin >> guess;
    
    cout << "Enter a tolerance: ";
    cin >> eps;
    
    cout << "a " << a << ", guess " << guess << ", eps " << eps << endl;
    
    x_n = guess;
    temp = guess;
    do 
    {
        x_m = temp;
        x_n = (x_m + a/x_m) / 2.0;
        temp = x_n;
        
        cout << "error " << abs(x_n-x_m) << ", x_n " << x_n << ", x_m " << x_m
             << ", iteration " << ++iter << endl;
    } while (iter < max_iter && abs(x_n-x_m) > eps);
    
    cout << "SOLUTION CONVERGED" << endl;

    return 0;
}
