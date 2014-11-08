#include <armadillo>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
#include <cstdint>

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
mat create_grid_rectangle_helmholtz(const unsigned int m, const unsigned int n,
    const double h, const double delta)
{
    mat A(n*m,n*m);

    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };
        
    const auto delta_div_hsq = delta / (h*h);

    for (auto i = uint_fast32_t {0}; i < n; i++)
        for (auto j = uint_fast32_t {0}; j < m; j++)
        {
            unsigned int center = map(i,j);

            A(center, center) = -(4*delta_div_hsq + 1);
            /* TODO: we can rewrite this in a more efficient way. have loop
                        iterate over interior nodes, this way we do not need
                        to do bounds checking 
            */
            if (i < n-1) A(center, map(i+1,j)) = delta_div_hsq;
            if (j < m-1) A(center, map(i,j+1)) = delta_div_hsq;
            if (i > 0) A(center, map(i-1,j)) = delta_div_hsq;
            if (j > 0) A(center, map(i,j-1)) = delta_div_hsq;
        }

    return A;
}

/*
 *
 */
mat create_grid_rectangle_laplace(const unsigned int m, const unsigned int n)
{
    mat A(n*m,n*m);

    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };

    for (auto i = uint_fast32_t {0}; i < n; i++)
        for (auto j = uint_fast32_t {0}; j < m; j++)
        {
            unsigned int center = map(i,j);

            A(center, center) = -4;
            /* TODO: we can rewrite this in a more efficient way. have loop
                        iterate over interior nodes, this way we do not need
                        to do bounds checking 
            */
            if (i < n-1) A(center, map(i+1,j)) = 1;
            if (j < m-1) A(center, map(i,j+1)) = 1;
            if (i > 0) A(center, map(i-1,j)) = 1;
            if (j > 0) A(center, map(i,j-1)) = 1;
        }

    return A;
}

/*
 *
 */
void enforce_bcs(const unsigned int m, const unsigned int n, mat& A,
    vec& b, const double value)
{
    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };

    for (auto j = uint_fast32_t {0}; j < n; j++)
    {
        auto center = map (0, j);
        (A.row (center)).zeros ();
        A (center,center) = 1.;
        b (center) = value;
        
        center = map (m-1, j);
        (A.row (center)).zeros ();
        A (center,center) = 1.;
        b (center) = value; 
    }
    
    for (auto i = uint_fast32_t {0}; i < m; i++)
    {
        auto center = map (i, 0);
        (A.row (center)).zeros ();
        A (center,center) = 1.;
        b (center) = value;
        
        center = map (i, n-1);
        (A.row (center)).zeros ();
        A (center,center) = 1.;
        b (center) = value; 
    }
}

/*
 *
 */
int main()
{
    const auto max_iterations = uint_fast32_t { 1000000 };

    vector<string> columns;
    vec::fixed<16*16> helm_N16;
    vec::fixed<64*64> helm_N64;

    ifstream helm_N16_data("helmholtz_N-16.csv");
    string line, val;
    
    /* first line is headers */
    getline(helm_N16_data, line);
    while (getline(helm_N16_data, line))
    {
        int i,j;
        stringstream line_stream(line);
        string cell;
        
        while (getline(line_stream, cell, ',')) columns.push_back(cell);
        i = stoi(columns[0])-1;
        j = stoi(columns[1])-1;
        helm_N16(16*i+j) = stod(columns[4]);
        
        columns.clear();
    }
    helm_N16_data.close();

    const auto delta = 0.01;
    auto n = uint_fast32_t { 16 };
    auto m = n;
    
    auto h = 1./n;
 
    /* helmholtz size 16 */
    auto grid = create_grid_rectangle_helmholtz(m,n,h,delta);
    mat inv_C(n*m, n*m);
    inv_C.zeros();
    for (auto i = uint_fast32_t { 0 }; i < grid.n_rows; i++)
        inv_C(i,i) = grid(i,i);
    inv_C = inv_C.i();
    
    vec b(n*m);
    b.fill(1.*delta); /*!< R(x,y) = 1 */
    enforce_bcs(m, n, grid, b, 0.);

    bool has_converged;
    vec x(m*n);
    x.zeros();
    vector<double> conj_grad_error_analytical;
    vector<double> conj_grad_error_residual;
    auto iter = uint_fast64_t {1};
    
    do
    {
        cout << "Conjugate Gradient, N=16, iter=" << iter;
    
        tie (x, has_converged) = 
            conj_grad_steepest_desc (grid, b, x, 1.e-4, 1);
        conj_grad_error_analytical.push_back(norm(x - helm_N16, "inf") / 
            norm(helm_N16, "inf"));
        conj_grad_error_residual.push_back(norm(b - grid*x, "inf"));
        
        cout << ", analyt_err=" << conj_grad_error_analytical.back()
             << ", resid_err=" << conj_grad_error_residual.back() << endl;
    } while (!has_converged && ++iter <= max_iterations);
    
    vec xp(m*n);
    xp.zeros();
    vector<double> pre_conj_grad_error_analytical;
    vector<double> pre_conj_grad_error_residual;
    iter = 1;
    
    do
    {
        cout << "Precond Conjugate Gradient, N=16, iter=" << iter;
    
        tie (xp, has_converged) = 
            conj_grad_precond (grid, b, inv_C, xp, 1.e-4, 1);
        pre_conj_grad_error_analytical.push_back(norm(xp - helm_N16, "inf") / 
            norm(helm_N16, "inf"));
        pre_conj_grad_error_residual.push_back(norm(b - grid*xp, "inf"));
        
        cout << ", analyt_err=" << pre_conj_grad_error_analytical.back()
             << ", resid_err=" << pre_conj_grad_error_residual.back() << endl;
    } while (!has_converged && ++iter <= max_iterations);
    
    WRITE_OUT_APPROX_RESULTS(h16_appox, "h16_cg_approx.dsv", helm_N16, x, xp);
    WRITE_OUT_ERROR_RESULTS(h16_conj_grad, "h16_conj_grad.dsv", 
        conj_grad_error_analytical, conj_grad_error_residual);
    WRITE_OUT_ERROR_RESULTS(h16_precond_conj_grad, "h16_precond_conj_grad.dsv", 
        pre_conj_grad_error_analytical, pre_conj_grad_error_residual);
        
    /* ----------------------------- N = 64 -------------------------------- */
    
    ifstream helm_N64_data("helmholtz_N-64.csv");
    
    /* first line is headers */
    getline(helm_N64_data, line);
    while (getline(helm_N64_data, line))
    {
        int i,j;
        stringstream line_stream(line);
        string cell;
        
        while (getline(line_stream, cell, ',')) columns.push_back(cell);
        i = stoi(columns[0])-1;
        j = stoi(columns[1])-1;
        helm_N64(64*i+j) = stod(columns[4]);
        
        columns.clear();
    }
    helm_N64_data.close();
    
    n = 64;
    m = n;
    
    h = 1./n;
 
    /* helmholtz size 64 */
    grid = create_grid_rectangle_helmholtz(m,n,h,delta);
    inv_C.resize(n*m, n*m);
    inv_C.zeros();
    for (auto i = uint_fast32_t { 0 }; i < grid.n_rows; i++)
        inv_C(i,i) = grid(i,i);
    inv_C = inv_C.i();
    
    b.resize(n*m);
    b.fill(1.*delta); /*!< R(x,y) = 1 */
    enforce_bcs(m, n, grid, b, 0.);
    
    x.resize(m*n);
    x.zeros();
    conj_grad_error_analytical.clear();
    conj_grad_error_residual.clear();
    iter = 1;
    
    do
    {
        cout << "Conjugate Gradient, N=64, iter=" << iter;
    
        tie (x, has_converged) = 
            conj_grad_steepest_desc (grid, b, x, 1.e-4, 1);
        conj_grad_error_analytical.push_back(norm(x - helm_N64, "inf") / 
            norm(helm_N64, "inf"));
        conj_grad_error_residual.push_back(norm(b - grid*x, "inf"));
        
        cout << ", analyt_err=" << conj_grad_error_analytical.back()
             << ", resid_err=" << conj_grad_error_residual.back() << endl;
    } while (!has_converged && ++iter <= max_iterations);
    
    xp.resize(m*n);
    xp.zeros();
    pre_conj_grad_error_analytical.clear();
    pre_conj_grad_error_residual.clear();
    iter = 1;
    
    do
    {
        cout << "Precond Conjugate Gradient, N=64, iter=" << iter;
    
        tie (xp, has_converged) = 
            conj_grad_precond (grid, b, inv_C, xp, 1.e-4, 1);
        pre_conj_grad_error_analytical.push_back(norm(xp - helm_N64, "inf") / 
            norm(helm_N64, "inf"));
        pre_conj_grad_error_residual.push_back(norm(b - grid*xp, "inf"));
        
        cout << ", analyt_err=" << pre_conj_grad_error_analytical.back()
             << ", resid_err=" << pre_conj_grad_error_residual.back() << endl;
    } while (!has_converged && ++iter <= max_iterations);
    
    WRITE_OUT_APPROX_RESULTS(h64_appox, "h64_cg_approx.dsv", helm_N64, x, xp);
    WRITE_OUT_ERROR_RESULTS(h64_conj_grad, "h64_conj_grad.dsv", 
        conj_grad_error_analytical, conj_grad_error_residual);
    WRITE_OUT_ERROR_RESULTS(h64_precond_conj_grad, "h64_precond_conj_grad.dsv", 
        pre_conj_grad_error_analytical, pre_conj_grad_error_residual);

    return 0;
}
