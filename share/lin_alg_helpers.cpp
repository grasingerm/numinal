/**
 * Calculate the L infinity norm of a matrix and store its location
 *
 * @param A Matrix or vector
 * @param row Integer to store row number of location of the infinity norm
 * @param col Integer to store col number of location of the infinity norm
 * @return L infinity norm
 */
double norm_linf(const mat& A, unsigned int& row, unsigned int& col)
{
    double abs_val;
    double max = 0;
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        for (unsigned int j = 0; j < A.n_cols; j++)
        {
            abs_val = fabs(A(i,j));
            if (abs_val > max)
            {
                max = abs_val;
                row = i;
                col = j;
            }
        }
    }
    
    return max;
}

/**
 * Calculate the L infinity norm of a matrix
 *
 * @param A Matrix or vector
 * @return L infinity norm
 */
double norm_linf(const mat& A)
{
    double abs_val;
    double max = 0;
    for (unsigned int i = 0; i < A.n_rows; i++)
    {
        for (unsigned int j = 0; j < A.n_cols; j++)
        {
            abs_val = fabs(A(i,j));
            if (abs_val > max) max = abs_val;
        }
    }
    
    return max;
}

/**
 * Calculate the L2 norm of a vector
 *
 * @param u Vector
 * @return L2 norm
 */
double norm_l2(const vec& u)
{
    double sum = 0;
    for (unsigned int i = 0; i < u.n_rows; i++) sum += u(i)*u(i);
    return sqrt(sum);
}
