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
#include "eigen.hpp"

using namespace arma;
using namespace std;
using namespace eigen;

/*
 *
 */
int main ()
{
    const auto max_iterations = uint_fast32_t { 1000000 };
    const auto delta = 0.01;
    const auto bc_value = 0.;
    const auto tol = 1e-4;

    for (auto n = uint_fast32_t { 16 }; n <= 128; n *= 2 )
    {
        mat grid_helmholtz, grid_laplace;
        vec b;

        auto m = n;
        auto h = 1./n;

        tie (grid_helmholtz, b) = 
            create_grid_rectangle_helmholtz (m,n,h,delta,bc_value);
        tie (grid_laplace, b) = create_grid_rectangle_laplace (m,n,h,bc_value);

/*
        householder_Mut (grid_helmholtz);
        householder_Mut (grid_laplace);

        vec a_h = diagvec (grid_helmholtz, 0);
        vec b_h = diagvec (grid_helmholtz, 1);

        cout << "grid_helmholtz = " << endl << grid_helmholtz << endl;
        cout << "a_h = " << endl << a_h << endl;
        cout << "b_h = " << endl << b_h << endl;
        cout << "a_n = " << a_h.n_rows << endl;
        cout << "b_n = " << b_h.n_rows << endl;

        vec a_l = diagvec (grid_laplace, 0);
        vec b_l = diagvec (grid_laplace, 1);

        tie (lambdas_helmholtz, index_helmholtz, success_helmholtz) = 
            qr_method_Mut (a_h, b_h, tol, max_iterations);
        tie (lambdas_laplace, index_laplace, success_laplace) = 
            qr_method_Mut (a_l, b_l, tol, max_iterations);

        cout << "N = " << n << endl;

        cout << setw(30) << "eigenvalues helmholtz: ";
        for (auto lambda : lambdas_helmholtz) cout << lambda << " ";
        cout << endl;

        cout << setw(30) << "eigenvalues laplace: ";
        for (auto lambda : lambdas_laplace) cout << lambda << " ";
        cout << endl << endl;
*/
        auto eig_h = eig_sym ( grid_helmholtz );
        auto eig_l = eig_sym ( grid_laplace );

        cout << "eigenvalues helmholtz, n=" << n << ": " << endl << eig_h << endl; 
        cout << "eigenvalues laplace, n=" << n << ": " << endl << eig_l << endl;
    }

    return 0;
}
