#ifndef __DIRECT_METHODS_HPP__
#define __DIRECT_METHODS_HPP__

#include <armadillo>

/*
 * Gaussian elimation
 */
void gauss_elim_wpp_Mut(arma::mat&, arma::vec&); // @mutator
arma::vec solve_gauss_elim_wpp(arma::mat, arma::vec);

/*
 * Substitution
 */
arma::vec backward_subs(const arma::mat&, const arma::vec&);

#endif /* __DIRECT_METHODS_HPP__ */
