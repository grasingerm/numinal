#ifndef __EIGEN_HPP__
#define __EIGEN_HPP__

#include <tuple>
#include <armadillo>

namespace eigen
{
    typedef std::tuple<arma::vec,double,double,bool> iterative_solution_t;

    iterative_solution_t power_method
        (const arma::mat&, const arma::vec&, const double, const unsigned int);
        
    iterative_solution_t power_method_d2
        (const arma::mat&, const arma::vec&, const double, const unsigned int);
        
    iterative_solution_t inverse_power_method
        (const arma::mat&, const arma::vec&, const double, const unsigned int);
}

#endif /* __EIGEN_HPP__ */
