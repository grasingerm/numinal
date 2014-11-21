#ifndef __EIGEN_HPP__
#define __EIGEN_HPP__

#include <tuple>
#include <armadillo>
#include <vector>
#include "direct_methods.hpp"

namespace eigen
{
    /* iterative methods */
    typedef std::tuple<arma::vec,double,double,bool> iterative_solution_t;
    typedef std::tuple<std::vector<double>,unsigned int,bool> qr_solution_t;

    iterative_solution_t power_method
        (const arma::mat&, arma::vec, const double, const unsigned int);
        
    iterative_solution_t power_method_d2
        (const arma::mat&, arma::vec, const double, const unsigned int);
        
    iterative_solution_t inverse_power_method
        (const arma::mat&, arma::vec, const double, const unsigned int);
        
    iterative_solution_t inverse_power_method
        (const arma::mat&, arma::vec, const double, const unsigned int,
        lin_solver::lu_solution_t);
        
    iterative_solution_t inverse_power_method
        (const arma::mat&, arma::vec, const double, const unsigned int,
        lin_solver::gausselimpp_solution_t);
        
    iterative_solution_t wielandt_deflation
        (const arma::mat&, const arma::vec&, const arma::vec&, const double, 
        const double, const unsigned int);
        
    qr_solution_t qr_method
        (arma::vec, arma::vec, const double, const unsigned int);
        
    qr_solution_t qr_method_Mut
        (arma::vec&, arma::vec&, const double, const unsigned int);
}

/* transformations */  
arma::mat householder(const arma::mat&);
void householder_Mut(arma::mat&);

#endif /* __EIGEN_HPP__ */
