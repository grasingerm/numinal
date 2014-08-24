#include <armadillo>
#include <iostream>
#include "config.hpp"
#include "iterative.hpp"
#include "assignment_helpers.hpp"

using namespace arma;
using namespace std;

/*
 *
 */
mat create_grid_rectangle(const unsigned int n, const unsigned int m,
    const double h, const double delta)
{
    mat A(n*n,m*m);

    unsigned int center_i, center_j;
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < m; j++)
        {
            center_i = n*i+j;
            center_j = j;
        
            A(center_i, center_j) = -4-h*h;
            A(center_i+1, center_j) = 1;
            A(center_i, center_j+1) = 1;
            if (center_i > 0) A(center_i-1,center_j) = 1;
            if (center_j > 0) A(center_i,center_j-1) = 1;
        }
        
    return delta/h/h * A; /* delta/h^2 * A */
}

/*
 *
 */
int main()
{
    const double delta = 0.01;

    double h;
    for (unsigned int i = 6; i < 32; i+=2)
    {
        h = 1/i;
        cout << "A = " << create_grid_rectangle(i, i, h, delta) << endl;
    }

    return 0;
}
