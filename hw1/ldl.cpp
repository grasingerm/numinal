#include <armadillo>
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_NEAR(act, approx, tol) \
    if (!act) \
        assert(!approx || abs(approx) < tol); \
    else \
        assert(abs(((act)-(approx))/(act)) < tol)

using namespace arma;
using namespace std;

enum factor_err_code
{
    FACTORIZATION_FAILURE,
    FACTORIZATION_SUCCESS
};

bool is_symmetric(const mat &A)
{
    for (unsigned int i = 0; i < A.n_rows; i++)
        for (unsigned int j = i+1; j < A.n_cols; j++)
            if (A(i,j) != A(j,i)) return false;
    return true;
}

factor_err_code ldl_factorization(const mat &A, mat &L, vec &D)
{
    /* sanity checks */
    assert(A.is_square());
    assert(is_symmetric(A));
    assert(A.n_rows == L.n_rows);
    assert(L.is_square());
    assert(L.n_rows == D.n_rows);

    double sum;
    vec v(A.n_rows);
    
    L.eye();
    D.zeros();
    
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        for (unsigned int j = 0; j < i; j++)
            v(j) = L(i,j)*D(j);
            
        sum = 0;
        for (unsigned int j = 0; j < i; j++)
            sum += L(i,j)*v(j);
            
        D(i) = A(i,i) - sum;
        
        for (unsigned int j = i+1; j < A.n_rows; j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < i; k++)
                sum += L(j,k)*v(k);
                
            L(j,i) = (A(j,i)-sum) / D(i);
        }
    }
    
    return FACTORIZATION_SUCCESS;
}

int main(int argc, char* argv[])
{
    mat::fixed<3,3> A;
    A   <<  4   <<  -1      <<  1       <<  endr
        <<  -1  <<  4.25    <<  2.75    <<  endr
        <<  1   <<  2.75    <<  3.5     <<  endr;
        
    mat::fixed<3,3> L;
    vec::fixed<3> D;
    
    mat::fixed<3,3> A_computed;
    
    /**
     * example 4, pg 405
     */
    cout << "Example 4, pg 405" << endl;
    cout << "Matrix to be factorized, A = " << endl << A << endl;
    
    assert( ldl_factorization(A, L, D) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L << endl;
    cout << "D = " << endl << D << endl;
    
    /* compute A with L and D and check to make sure it equals the original */
    A_computed = L * diagmat(D) * trans(L);
    cout << "LDL' = " << endl << A_computed << endl;
    
    cout << "... testing A == LDL' ... ";
    for (unsigned int i = 0; i < A.n_rows; i++)
        for (unsigned int j = 0; j < A.n_cols; j++)
            ASSERT_NEAR(A(i,j), A_computed(i,j), 1e-5);
    cout << " ... passed." << endl;
    /* end of example 4 */

    return 0;
}
