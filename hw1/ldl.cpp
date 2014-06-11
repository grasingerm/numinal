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

/**
 * Returns true if the matrix is symmetric
 *
 * @param Matrix to check, by reference
 * @return True if symmetric
 */
bool is_symmetric(const mat &A)
{
    for (unsigned int i = 0; i < A.n_rows; i++)
        for (unsigned int j = i+1; j < A.n_cols; j++)
            if (A(i,j) != A(j,i)) return false;
    return true;
}

/**
 * Factor a positive definite matrix with LDL'
 *
 * @param A Matrix to factor, by reference
 * @param L Matrix to store lower diagonal, by reference
 * @param D Vector to store diagonal, by reference
 * @return Error code
 */
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

factor_err_code cholesky(const mat &A, mat &L)
{
    double sum;
    L.zeros();

    L(0,0) = sqrt(A(0,0));
    for (unsigned int j = 1; j < A.n_rows; j++)
        L(j,0) = A(j,0)/L(0,0);
        
    for (unsigned int i = 1; i < A.n_cols-1; i++)
    {
        sum = 0;
        for (unsigned int k = 0; k < i; k++)
            sum += L(i,k)*L(i,k);
        L(i,i) = sqrt(A(i,i) - sum);
        
        for (unsigned int j = i+1; j < A.n_rows; j++)
        {
            sum = 0;
            for (unsigned int k = 0; k < A.n_rows-1; k++)
                sum += L(j,k)*L(i,k);
            L(j,i) = (A(j,i) - sum) / L(i,i);
        }
    }
    
    sum = 0;
    for (unsigned int k = 0; k < A.n_rows-1; k++)
        sum += L(A.n_rows-1,k)*L(A.n_rows-1,k);
    
    L(A.n_rows-1,A.n_cols-1) = sqrt(A(A.n_rows-1,A.n_cols-1) - sum);
    
    return FACTORIZATION_SUCCESS;
}


// MAIN
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
    cout << " ... passed." << endl << endl;
    /* end of example 4 */
    
    
    /**
     * problem 4b, pg 410
     */
    mat::fixed<3,3> A_4b;
    A_4b    <<  4   <<  2       <<  2       <<  endr
            <<  2   <<  6       <<  2       <<  endr
            <<  2   <<  2       <<  5       <<  endr;
    
    cout << endl << "=====================================" << endl;
    cout << "Problem 4b pg 410" << endl;
    cout << "Matrix to be factorized, A = " << endl << A_4b << endl;
    
    assert( ldl_factorization(A_4b, L, D) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L << endl;
    cout << "D = " << endl << D << endl;
    
    /* compute A with L and D and check to make sure it equals the original */
    A_computed = L * diagmat(D) * trans(L);
    cout << "LDL' = " << endl << A_computed << endl;
    
    cout << "... testing A == LDL' ... ";
    for (unsigned int i = 0; i < A_4b.n_rows; i++)
        for (unsigned int j = 0; j < A_4b.n_cols; j++)
            ASSERT_NEAR(A_4b(i,j), A_computed(i,j), 1e-5);
    cout << " ... passed." << endl << endl;
    /* end of problem 4b */
    
    /**
     * problem 4d, pg 410
     */
    mat::fixed<4,4> A_4d;
    A_4d    <<  4   <<  1       <<  1       <<  1   << endr
            <<  1   <<  3       <<  0       <<  -1  << endr
            <<  1   <<  0       <<  2       <<  1   << endr
            <<  1   <<  -1      <<  1       <<  4   << endr;
            
    mat::fixed<4,4> L_4;
    vec::fixed<4> D_4;
    mat::fixed<4,4> A_computed_4;
    
    cout << endl << "=====================================" << endl;
    cout << "Problem 4d pg 410" << endl;
    cout << "Matrix to be factorized, A = " << endl << A_4d << endl;
    
    assert( ldl_factorization(A_4d, L_4, D_4) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L_4 << endl;
    cout << "D = " << endl << D_4 << endl;
    
    /* compute A with L and D and check to make sure it equals the original */
    A_computed_4 = L_4 * diagmat(D_4) * trans(L_4);
    cout << "LDL' = " << endl << A_computed_4 << endl;
    
    cout << "... testing A == LDL' ... ";
    for (unsigned int i = 0; i < A_4d.n_rows; i++)
        for (unsigned int j = 0; j < A_4d.n_cols; j++)
            ASSERT_NEAR(A_4d(i,j), A_computed_4(i,j), 1e-5);
    cout << " ... passed." << endl << endl;
    /* end of problem 4b */
    
    /**
     * Problem 6b pg 410
     */
    cout << endl << "=====================================" << endl;
    cout << "Problem 6b" << endl;
    cout << "Matrix to be factorized, A = " << endl << A_4b << endl;
    
    assert( cholesky(A_4b,L) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L << endl;
    cout << "L' = " << endl << trans(L) << endl;
    
    /* compute A with L and check to make sure it equals the original */
    A_computed = L * trans(L);
    cout << "LL' = " << endl << A_computed << endl;
    
    cout << "... testing A == LL' ... ";
    for (unsigned int i = 0; i < A_4b.n_rows; i++)
        for (unsigned int j = 0; j < A_4b.n_cols; j++)
            ASSERT_NEAR(A_4b(i,j), A_computed(i,j), 1e-5);
    cout << " ... passed." << endl << endl;
    /* end of Problem 6b */
    
    /**
     * Problem 6d
     */
    cout << endl << "=====================================" << endl;
    cout << "Problem 6d" << endl;
    cout << "Matrix to be factorized, A = " << endl << A_4d << endl;
    
    assert( cholesky(A_4d,L_4) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L_4 << endl;
    cout << "L' = " << endl << trans(L_4) << endl;
    
    /* compute A with L and check to make sure it equals the original */
    A_computed_4 = L_4 * trans(L_4);
    cout << "LL' = " << endl << A_computed_4 << endl;
    
    cout << "... testing A == LL' ... ";
    for (unsigned int i = 0; i < A_4d.n_rows; i++)
        for (unsigned int j = 0; j < A_4d.n_cols; j++)
            ASSERT_NEAR(A_4d(i,j), A_computed_4(i,j), 1e-5);
    cout << " ... passed." << endl << endl;
    /* end of Problem 6d */
    
    /**
     * example 4 pg 405
     */
    A       <<  4   <<  -1      <<  1       << endr
            <<  -1  <<  4.25    <<  2.75    << endr
            <<  1   <<  2.75    <<  3.5     << endr;
    
    cout << endl << "=====================================" << endl;
    cout << "Example 4 pg 405" << endl;
    cout << "Matrix to be factorized, A = " << endl << A << endl;
    
    assert( cholesky(A,L) == FACTORIZATION_SUCCESS );
    
    cout << "L = " << endl << L << endl;
    cout << "L' = " << endl << trans(L) << endl;
    
    /* compute A with L and check to make sure it equals the original */
    A_computed = L * trans(L);
    cout << "LL' = " << endl << A_computed << endl;
    
    cout << "... testing A == LL' ... ";
    for (unsigned int i = 0; i < A.n_rows; i++)
        for (unsigned int j = 0; j < A.n_cols; j++)
            ASSERT_NEAR(A(i,j), A_computed(i,j), 1e-5);
    cout << " ... passed." << endl << endl;
    /* end of example 4 */

    return 0;
}
