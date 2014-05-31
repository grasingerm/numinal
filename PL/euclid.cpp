#include <iostream>

using namespace std;

int gcd(int m, int n)
{
    int temp, diff;

    if (n == m)
        return n;
        
    /* make sure m is greater than n */
    if (n > m)
    {
        temp = m;
        m = n;
        n = temp;
        cout << "m = " << m << ", n = " << n << endl;
    }
    
    diff = m - n;
    m = n;
    n = diff;
    
    cout << "m = " << m << ", n = " << n << endl; 
    
    return gcd(m, n);
}

int main(int argc, char* argv[])
{
    int m, n;
    
    while (1)
    {
        cout << endl << "First integer ";
        cin >> m;
        cout << "Second integer ";
        cin >> n;
        
        if (!m && !n) break;
        
        cout << endl << "Greatest common divisor is " << gcd(m, n) << endl;
    }
    
    return 0;
}
