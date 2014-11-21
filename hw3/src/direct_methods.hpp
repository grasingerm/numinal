#ifndef __DIRECT_METHODS_HPP__
#define __DIRECT_METHODS_HPP__

#include <armadillo>
#include <tuple>

/*
 * Method types
 */
namespace lin_solver
{
    enum    lu_solution_t               {   lu_decomp                 };
    enum    gausselimpp_solution_t      {   gauss_elim_wpp            };
}

/*
 * Gaussian elimation
 */
typedef std::tuple<arma::vec,bool> gauss_elim_t;
 
bool gauss_elim_wpp_Mut(arma::mat&, arma::vec&); // @mutator
gauss_elim_t solve_gauss_elim_wpp(arma::mat, arma::vec);

/*
 * Substitution
 */
arma::vec backward_subs(const arma::mat&, const arma::vec&);

/*
 * LU Decomposition
 */
typedef std::tuple<arma::mat,arma::mat,bool> lu_factor_t; 

lu_factor_t lu_doolittle(const arma::mat&);
arma::vec lu_fb_subs(const arma::mat&, const arma::mat&, const arma::vec&);



#endif /* __DIRECT_METHODS_HPP__ */
