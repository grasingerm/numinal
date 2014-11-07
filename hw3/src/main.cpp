#include <armadillo>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>
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
        for (auto i = 0; i < a_vector_.size(); i++) \
            name_ << setw(15) << i << setw(15) << a_vector_[i] \
                       << setw(15) << r_vector_[i] << endl; \
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

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < m; j++)
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

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < m; j++)
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

    for (auto j = 0; j < n; j++)
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
    
    for (auto i = 0; i < m; i++)
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
    vector<string> columns;
    vec::fixed<16*16> helm_N16;
    vec::fixed<64*64> helm_N64;

    ifstream helm_N16_data("helmholtz_N-16.csv");
    string line, val;
    
    /* first line is headers */
    getline(helm_N16_data, line);
    while (getline(helm_N16_data, line))
    {
        cout << line << endl;
    
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
    
    cout << "analytical solution = " << endl << helm_N16 << endl;

    const double delta = 0.01;
    double h;
    mat grid;
    
    unsigned int n = 16;
    unsigned int m = n;
    
    h = 1./n;
 
    /* helmholtz size 16 */
    grid = create_grid_rectangle_helmholtz(m,n,h,delta);
    vec b(n*m);
    b.fill(1.*delta); /*!< R(x,y) = 1 */
    enforce_bcs(m, n, grid, b, 0.);
    
    ofstream grid_file;
    grid_file.open("grid_N-16.txt");
    grid_file << grid << endl;
    grid_file.close();

    bool has_converged;
    vec x(m*n);
    x.zeros();
    vector<double> jacobi_error_analytical;
    vector<double> jacobi_error_residual;
    do
    {
        tie (x, has_converged) = jacobi (grid, b, x, 1.e-4, 1);
        jacobi_error_analytical.push_back(norm(x - helm_N16, "inf") / 
            norm(helm_N16, "inf"));
        jacobi_error_residual.push_back(norm(b - grid*x, "inf"));
    } while (!has_converged);
    
    cout << "numerical soln (jacobi) = " << endl << x << endl;
    
    x.zeros();
    vector<double> gauss_error_analytical;
    vector<double> gauss_error_residual;
    do
    {
        tie (x, has_converged) = gauss_seidel (grid, b, x, 1.e-4, 1);
        gauss_error_analytical.push_back(norm(x - helm_N16, "inf") / 
            norm(helm_N16, "inf"));
        gauss_error_residual.push_back(norm(b - grid*x, "inf"));
    } while (!has_converged);
    
    cout << "numerical soln (gauss) = " << endl << x << endl;
    
    WRITE_OUT_ERROR_RESULTS(h16_jacobi, "h16_jacobi.dsv", 
        jacobi_error_analytical, jacobi_error_residual);
    WRITE_OUT_ERROR_RESULTS(h16_gauss, "h16_gauss.dsv", gauss_error_analytical, 
        gauss_error_residual);
        
    /* ----------------------------- N = 64 -------------------------------- */
    
    ifstream helm_N64_data("helmholtz_N-64.csv");
    
    /* first line is headers */
    getline(helm_N64_data, line);
    while (getline(helm_N64_data, line))
    {
        cout << line << endl;
    
        int i,j;
        stringstream line_stream(line);
        string cell;
        
        while (getline(line_stream, cell, ',')) columns.push_back(cell);
        i = stoi(columns[0])-1;
        j = stoi(columns[1])-1;
        helm_N64(16*i+j) = stod(columns[4]);
        
        columns.clear();
    }
    helm_N64_data.close();
    
    cout << "analytical solution = " << endl << helm_N64 << endl;
    
    n = 64;
    m = n;
    
    h = 1./n;
 
    /* helmholtz size 16 */
    grid = create_grid_rectangle_helmholtz(m,n,h,delta);
    b.resize(n*m);
    b.fill(1.*delta); /*!< R(x,y) = 1 */
    enforce_bcs(m, n, grid, b, 0.);
    
    grid_file.open("grid_N-64.txt");
    grid_file << grid << endl;
    grid_file.close();

    x.resize(m*n);
    x.zeros();
    jacobi_error_analytical.clear();
    jacobi_error_residual.clear();
    do
    {
        tie (x, has_converged) = jacobi (grid, b, x, 1.e-4, 1);
        jacobi_error_analytical.push_back(norm(x - helm_N64, "inf") / 
            norm(helm_N64, "inf"));
        jacobi_error_residual.push_back(norm(b - grid*x, "inf"));
    } while (!has_converged);
    
    cout << "numerical soln (jacobi) = " << endl << x << endl;
    
    x.zeros();
    gauss_error_analytical.clear();
    gauss_error_residual.clear();
    do
    {
        tie (x, has_converged) = gauss_seidel (grid, b, x, 1.e-4, 1);
        gauss_error_analytical.push_back(norm(x - helm_N64, "inf") / 
            norm(helm_N64, "inf"));
        gauss_error_residual.push_back(norm(b - grid*x, "inf"));
    } while (!has_converged);
    
    cout << "numerical soln (gauss) = " << endl << x << endl;
    
    WRITE_OUT_ERROR_RESULTS(h64_jacobi, "h64_jacobi.dsv", 
        jacobi_error_analytical, jacobi_error_residual);
    WRITE_OUT_ERROR_RESULTS(h64_gauss, "h64_gauss.dsv", gauss_error_analytical, 
        gauss_error_residual);

    return 0;
}
