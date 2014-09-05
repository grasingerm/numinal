#ifndef __ITERATIVE_HPP__
#define __ITERATIVE_HPP__

#include "numinal_config.hpp"
#ifdef __USE_ARMADILLO__
    #include <armadillo>
#endif

#include <tuple>

typedef std::tuple<arma::vec,bool> iter_soln_t;

/* jacobi iterative method */
iter_soln_t jacobi
    (const arma::mat&, const arma::vec&, arma::vec, const double, 
    const unsigned int);
iter_soln_t jacobi
    (const arma::mat&, const arma::vec&, const double, const unsigned int);
    
/* gauss seidel iterative method */
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, arma::vec, const double, 
    const unsigned int);
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, const double, const unsigned int);
/* gauss seidel with relaxation */
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, arma::vec, const double, 
    const unsigned int, const double);
iter_soln_t gauss_seidel
    (const arma::mat&, const arma::vec&, const double, const unsigned int,
    const double);
    
void reorder_for_jacobi_M(arma::mat&, arma::vec&);
std::tuple<arma::mat,arma::vec> reorder_for_jacobi(arma::mat, arma::vec);

/* conjugate gradient method */
iter_soln_t conj_grad_precond
    (const arma::mat&, const arma::vec&, const arma::mat&, arma::vec,
    const double, const unsigned int);
iter_soln_t conj_grad_steepest_desc
    (const arma::mat&, const arma::vec&, arma::vec,
    const double, const unsigned int);

/* TODO: reconsider naming convention, prepend all solution functions perhaps?*/
arma::mat conj_grad_precond_mat_jacobi(const arma::vec&);

#endif /* __ITERATIVE_HPP__ */
