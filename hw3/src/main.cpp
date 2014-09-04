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
mat create_grid_rectangle(const unsigned int m, const unsigned int n,
    const double h, const double delta)
{
    mat A(n*m,n*m);

    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < m; j++)
        {
            unsigned int center = map(i,j);

            A(center, center) = -4-h*h;
            /* TODO: we can rewrite this in a more efficient way. have loop
                        iterate over interior nodes, this way we do not need
                        to do bounds checking 
            */
            if (i < n-1) A(center, map(i+1,j)) = 1;
            if (j < m-1) A(center, map(i,j+1)) = 1;
            if (i > 0) A(center, map(i-1,j)) = 1;
            if (j > 0) A(center, map(i,j-1)) = 1;
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
    
    

    return 0;
}
