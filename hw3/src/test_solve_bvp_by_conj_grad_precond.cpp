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

using namespace arma;
using namespace std;

/*
 *
 */
int main()
{
    const auto max_iterations = uint_fast32_t { 1000000 };
    const auto delta = 0.01;
    const auto bc_value = 0.;
    
    vector < tuple < uint_fast32_t, uint_fast32_t > > convergence_results;

    for (auto n = uint_fast32_t { 16 }; n <= 128; n *= 2 )
    {
        auto m = n;
        auto h = 1./n;

        mat grid;
        vec b;
        vec x (n*m);
        x.zeros ();

        tie (grid, b) = create_grid_rectangle_helmholtz (m,n,h,delta,bc_value);
        auto inv_C = (diagmat (grid)).i ();
        
        auto iter = uint_fast32_t { 1 };

        bool has_converged;

        do
        {
            cout << "Conjugate grad precond, N=" << n << ", iter=" << iter;
        
            tie (x, has_converged) = conj_grad_precond (grid, b, inv_C, 
                x, 1.e-4, 1);
            auto resid_error = norm(b - grid*x, "inf");
            
            cout << ", resid_err=" << resid_error << endl;
        } while (!has_converged && ++iter <= max_iterations);

        convergence_results.emplace_back (n, iter);
    }

    cout << endl << setw (20) << "n" << setw (20) << "iterations" << endl;
    cout << "----------------------------------------" << endl;
    for (const auto t : convergence_results)
        cout << setw (20) << get<0> (t) << setw(20) << get<1> (t) << endl;
    
    return 0;
}
