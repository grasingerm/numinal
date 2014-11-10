#ifndef __FINITE_DIFFERENCE_HPP__
#define __FINITE_DIFFERENCE_HPP__

#include <armadillo>

using namespace arma;

mat create_grid_rectangle_helmholtz(const unsigned int m, const unsigned int n,
    const double h, const double delta);
mat create_grid_rectangle_laplace(const unsigned int m, const unsigned int n);
void enforce_bcs(const unsigned int m, const unsigned int n, mat& A,
    vec& b, const double value);

#endif
