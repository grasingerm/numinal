#include <armadillo>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <cstdint>

#include "finite_difference.hpp"
#include "config.hpp"
#include "iterative.hpp"
#include "assignment_helpers.hpp"


#define WRITE_OUT_ERROR_RESULTS(name_, filename_, a_vector_, r_vector_) \
    do \
    { \
        ofstream name_; \
        name_.open(filename_); \
        name_ << setw(13) << "# iter" << setw(15) << "analytical" \
                   << setw(15) << "residual" << endl; \
        for (auto i = uint_fast32_t { 0 }; i < a_vector_.size(); i++) \
            name_ << setw(15) << i << setw(15) << a_vector_[i] \
                       << setw(15) << r_vector_[i] << endl; \
                       \
        name_.close(); \
    } while(0)
    
#define WRITE_OUT_APPROX_RESULTS(name_, filename_, analyt, jacobi, gauss) \
    do \
    { \
        ofstream name_; \
        name_.open(filename_); \
        name_ << setw(13) << "# analyt" << setw(15) << "conj_grad" \
                   << setw(15) << "pre_conj_grad" << endl; \
        for (auto i = uint_fast32_t { 0 }; i < analyt.n_rows; i++) \
            name_ << setw(15) << analyt(i) << setw(15) << jacobi(i) \
                       << setw(15) << gauss(i) << endl; \
                       \
        name_.close(); \
    } while(0)

using namespace arma;
using namespace std;

/*
 *
 */
int main()
{
    const auto max_iterations = uint_fast32_t { 1000000 };
    const auto tol = 1e-4;

    const auto bc_value = 0.;
    auto n = uint_fast32_t { 16 };
    auto m = n;
    
    auto h = 1./n;
 
    /* helmholtz size 16 */
    mat grid;
    vec b;
    tie (grid, b) = create_grid_rectangle_laplace(m,n,h,bc_value);
    mat inv_C(n*m, n*m);
    inv_C.zeros();
    for (auto i = uint_fast32_t { 0 }; i < grid.n_rows; i++)
        inv_C(i,i) = grid(i,i);
    inv_C = inv_C.i();

    bool has_converged;
    auto iter = uint_fast64_t {1};
    
    vec xp(m*n);
    vector<double> pre_conj_grad_error_residual;
    
    do
    {
        xp.zeros();

        cout << "Precond Conjugate Gradient, N=16, iter=" << iter;
    
        tie (xp, has_converged) = 
            conj_grad_precond (grid, b, inv_C, xp, 1.e-9, iter);
        pre_conj_grad_error_residual.push_back(norm(b - grid*xp, "inf"));
        
        cout << ", resid_err=" << pre_conj_grad_error_residual.back() << endl;
    } while (pre_conj_grad_error_residual.back() > tol 
        && ++iter <= max_iterations);

    /* ----------------------------- N = 64 -------------------------------- */
    n = 64;
    m = n;
    
    h = 1./n;
 
    /* helmholtz size 64 */
    tie (grid, b) = create_grid_rectangle_laplace(m,n,h,bc_value);
    inv_C.resize(n*m, n*m);
    inv_C.zeros();
    for (auto i = uint_fast32_t { 0 }; i < grid.n_rows; i++)
        inv_C(i,i) = grid(i,i);
    inv_C = inv_C.i();
    
    xp.resize(m*n);
    pre_conj_grad_error_residual.clear();
    iter = 1;
    
    do
    {
        xp.zeros();

        cout << "Precond Conjugate Gradient, N=64, iter=" << iter;
    
        tie (xp, has_converged) = 
            conj_grad_precond (grid, b, inv_C, xp, 1.e-9, iter);
        pre_conj_grad_error_residual.push_back(norm(b - grid*xp, "inf"));
        
        cout << ", resid_err=" << pre_conj_grad_error_residual.back() << endl;
    } while (pre_conj_grad_error_residual.back() > tol
        && ++iter <= max_iterations);

    return 0;
}
