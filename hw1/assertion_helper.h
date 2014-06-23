#ifndef ASSERTION_HELPER
#define ASSERTION_HELPER 1

#define ASSERT_NEAR(act, approx, tol) \
    if (!act) \
        assert(!approx || abs(approx) < tol); \
    else \
        assert(abs(((act)-(approx))/(act)) < tol)
        
#define ASSERT_ARMA_MATRIX_NEAR(A, A_computed, tol) \
    for (unsigned int i = 0; i < (A).n_rows; i++) \
        for (unsigned int j = 0; j < (A).n_cols; j++) \
            ASSERT_NEAR((A)(i,j), (A_computed)(i,j), tol)
            
#define ASSERT_ARMA_VEC_NEAR(b, b_computed, tol) \
    for (unsigned int i = 0; i < b.n_rows; i++) \
        ASSERT_NEAR((b)(i), (b_computed)(i), tol)
        
#endif /* ASSERTION_HELPER */
