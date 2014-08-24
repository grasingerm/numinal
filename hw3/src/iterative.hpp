#ifndef __ITERATIVE_HPP__
#define __ITERATIVE_HPP__

#include "numinal_config.hpp"
#ifdef __USE_ARMADILLO__
    #include <armadillo>
#endif

#include <tuple>

typedef std::tuple<arma::vec,bool> iter_soln_t;

iter_soln_t jacobi
    (const arma::mat&, const arma::vec&, arma::vec, const double, 
    const unsigned int);
iter_soln_t jacobi
    (const arma::mat&, const arma::vec&, const double, const unsigned int);
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, arma::vec, const double, 
    const unsigned int);
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, const double, const unsigned int);
    
void reorder_for_jacobi_M(arma::mat&, arma::vec&);
std::tuple<arma::mat,arma::vec> reorder_for_jacobi(arma::mat, arma::vec);

#endif /* __ITERATIVE_HPP__ */
