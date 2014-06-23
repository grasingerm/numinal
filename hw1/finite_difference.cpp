#include <iostream>
#include <armadillo>
#include <cassert>
#include "assertion_helper.h"

#define HR1 "======================================"
#define HR2 "--------------------------------------"
#define HR3 "______________________________________"
#define HR4 ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"

using namespace std;
using namespace arma;

/**
 * LU factorization for a tridiagonal matrix
 *
 * @param A Matrix A for the linear system
 * @param b Vector b for the linear system
 * @param L Lower diagonal matrix
 * @param U Upper diagonal matrix
 * @return True for success
 */
vec crout_tridiagonal_solve(const mat &A, const vec &b, mat &L, mat &U)
{
    // TODO: can this be more memory efficient?
    vec x(A.n_rows);
    vec z(A.n_rows);
    
    L.zeros();
    U.eye();

    /* LU factorization */
    L(0,0) = A(0,0);
    U(0,1) = A(0,1)/L(0,0);
    z(0) = b(0)/L(0,0);
    
    for (unsigned int i = 1; i < A.n_rows-1; i++)
    {
        L(i,i-1) = A(i,i-1);
        L(i,i) = A(i,i) - L(i,i-1)*U(i-1,i);
        U(i,i+1) = A(i,i+1)/L(i,i);
        z(i) = (b(i)-L(i,i-1)*z(i-1))/L(i,i);
    }
    
    L(A.n_rows-1,A.n_rows-2) = A(A.n_rows-1,A.n_rows-2);
    L(A.n_rows-1,A.n_rows-1) = 
        A(A.n_rows-1,A.n_rows-1) - L(A.n_rows-1,A.n_rows-2)
                                    * U(A.n_rows-2,A.n_rows-1);
    z(A.n_rows-1) = (b(A.n_rows-1) - L(A.n_rows-1,A.n_rows-2)*z(A.n_rows-2)) /
            L(A.n_rows-1,A.n_rows-1);
    
    /* test to make sure factorization was successful */
    /* note: to eliminate this overhead macro define N_DEBUG */
    mat A_computed = L*U;
    ASSERT_ARMA_MATRIX_NEAR(A,A_computed,1e-5);
    
    cout << "z = " << endl << z << endl;
    
    /* solve */
    x(A.n_rows-1) = z(A.n_rows-1);
    for (int i = A.n_rows-2; i >= 0; i--) x(i) = z(i) - U(i,i+1)*x(i+1);
    
    return x;
}

/**
 * Discrete a one dimensional system
 *
 * @param length Length of rod
 * @param N Number of nodes
 * @param west Function pointer to calculate west-node
 * @param center Function pointer to calculate center-node
 * @param east Function pointer to calculate east-node
 * @return Matrix A for linear system of equations
 */
mat discretize_1D(const int length, const int N, double (*const west)
    (const double), double (*const center)(const double), double (*const east)
    (const double))
{
    double delta_x = double(length)/N;
    mat A(N,N);
    A.zeros();
    
    A(0,0) = center(delta_x);
    A(0,1) = west(delta_x);
    
    for (int i = 1; i < N-1; i++)
    {
        A(i,i) = center(delta_x);
        A(i,i-1) = east(delta_x);
        A(i,i+1) = west(delta_x);
    }
    
    A(N-1,N-1) = center(delta_x);
    A(N-1,N-2) = east(delta_x);
    
    return A;
}

/**
 * Calculate solution vector r
 *
 * @param N Number of nodes
 * @param bc1 Boundary condition at node 1
 * @param bc2 Boundary condition at node N
 * @param source Source term
 * @return Solution vector
 */
vec solution_vector_1D(const int N, const double bc1, const double bc2, 
    const double source = 0)
{
    vec r(N);
    r.zeros();
    
    r(0) = bc1 + source;
    for (int i = 0; i < N-1; i++) r(i) = source;
    r(N-1) = bc2 + source;
    
    return r;
}

/**
 * Calculate center-node
 *
 * @param delta_x Delta x
 * @return Node value
 */
inline double center_heat_transfer(const double delta_x)
{
    return 4 - 2/(delta_x*delta_x);
}

/**
 * Calculate west-node
 *
 * @param delta_x Delta x
 * @return Node value
 */
inline double west_heat_transfer(const double delta_x)
{
    return 1/(delta_x*delta_x);
}

/**
 * Calculate east-node
 *
 * @param delta_x Delta x
 * @return Node value
 */
inline double east_heat_transfer(const double delta_x)
{
    return 1/(delta_x*delta_x);
}

// main function
int main(int argc, char* argv[])
{
    /* bar of unit length */
    const unsigned int length = 1;
    const double max_nodes = 8;
    
    /* boundary conditions */
    const double bc1 = 0.0;
    const double bc2 = 1.0;
    
    vec r, x;
    mat A, U, L;
    
    /* test tridiagonal solver */
    cout << "example 5, pg. 409" << endl;

    A.set_size(4,4);
    A   <<  2   <<  -1  <<  0   <<  0   <<  endr
        <<  -1  <<  2   <<  -1  <<  0   <<  endr
        <<  0   <<  -1  <<  2   <<  -1  <<  endr
        <<  0   <<  0   <<  -1  <<  2   <<  endr;
    r.set_size(4);
    r << 1 << endr << 0 << endr << 0 << endr << 1 << endr;
    
    L.zeros(4,4);
    U.zeros(4,4);
    
    x = crout_tridiagonal_solve(A, r, L, U);
    cout << "A = " << endl << A << endl;
    cout << "b = " << endl << r << endl;
    cout << "x = " << endl << x << endl;
    
    vec::fixed<4> x_act;
    x_act << 1 << endr << 1 << endr << 1 << endr << 1 << endr;
    ASSERT_ARMA_VEC_NEAR(x_act,x,1e-5);
    /* end of example 5 */
    
    /* set up function pointers */
    double (*fwest)(const double), (*feast)(const double), \
        (*fcenter)(const double);
    fwest = &west_heat_transfer;
    fcenter = &center_heat_transfer;
    feast = &east_heat_transfer;
    
    cout << HR1 << endl;
    cout << "heat transfer through a rod approximated by finite difference"
        << endl;
    cout << "pde: second partial of f w/ respect to x + 4f = 0" << endl;
    cout << "f(0) = 0; f(L) = 1" << endl << endl;
    
    for (int N = 4; N <= max_nodes; N*=2)
    {
        cout << HR2 << endl;
        cout << N << " nodes." << endl;
        L.set_size(N,N);
        U.set_size(N,N);
        
        A = discretize_1D(length, N, fwest, fcenter, feast);
        r = solution_vector_1D(N, bc1, bc2);
        x = crout_tridiagonal_solve(A, r, L, U);
        cout << "A = " << endl << A << endl;
        cout << "r = " << endl << r << endl;
        cout << "x = " << endl << x << endl;
    }

    return 0;
}
