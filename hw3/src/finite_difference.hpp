#ifndef __FINITE_DIFFERENCE_HPP__
#define __FINITE_DIFFERENCE_HPP__

#include <armadillo>
#include <tuple>

std::tuple < arma::mat, arma::vec > 
create_grid_rectangle_helmholtz (const unsigned int, const unsigned int, 
    const double, const double, const double);

std::tuple < arma::mat, arma::vec >
create_grid_rectangle_laplace (const unsigned int, const unsigned int,
    const double, const double);

#endif