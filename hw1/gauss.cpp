#include <armadillo>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;
using namespace arma;

#define ASSERT_VECTOR_EQUAL(u, v, tolerance) \
for (int i = 0; i < u.n_rows; i++) \
    assert (abs((u(i)-v(i))/u(i)) <= tolerance)

vec gauss_elimination(mat &aug)
{
    int i, j;
    double temp;
    int rows = aug.n_rows;
    vec x(rows);

    int cols = aug.n_cols;
    for (i = 0; i < cols-2; i++)
    {
        for (j = i; j < rows; j++)
            /* find the first non-zero coeff at column i */
            if (aug(j, i)) break;
        if (i != j)
            /* swap rows i and j */
            for (int k = 0; k < cols; k++)
            {
                temp = aug(i, k);
                aug(i, k) = aug(j, k);
                aug(j, k) = temp;
            }
        /* perform elimination */
        for (j = i+1; j < rows; j++)
            aug.row(j) -= aug(j, i)/aug(i, i) *
                                    aug.row(i);
    }
    
    /* backward substitution */
    double sum;
    
    x(rows-1) = aug(rows-1, rows)/aug(rows-1, rows-1);
    for (i = rows-2; i >= 0; i--)
    {
        sum = 0;
        for (j = i+1; j < rows; j++)
            sum += aug(i, j)*x(j);
        x(i) = (aug(i, rows) - sum) / aug(i, i);
    }

    return x;
}

vec gauss_jordan(mat &aug)
{
    int i, j, k;
    double temp;
    int rows = aug.n_rows;
    vec x(rows);
    
    for (i = 0; i < rows; i++)
    {
        for (j = i; j < rows; j++)
            /* find the first non-zero coeff at column i */
            if (aug(j, i)) break;
        if (i != j)
            /* swap rows i and j */
            for (k = 0; k < aug.n_cols; k++)
            {
                temp = aug(i, k);
                aug(i, k) = aug(j, k);
                aug(j, k) = temp;
            }
        /* perform elimination */
        for (j = i+1; j < rows; j++)
            aug.row(j) -= aug(j, i)/aug(i, i) *
                                    aug.row(i);
        for (j = i-1; j >= 0; j--)
            aug.row(j) -= aug(j, i)/aug(i, i) *
                                    aug.row(i);
    }
    
    /* backward substitution */
    for (i = 0; i < rows; i++)
        x(i) = aug(i, rows)/aug(i, i);

    return x;
}

int main(int argc, char* argv[])
{
    /* guassian elimination */
    mat a(3, 4);
    
    a << 1  << 2  << 1 <<   8 << endr
      << -4 << 7  << 3 <<  19 << endr
      << 11 << -1 << 2 << 15  << endr;
    
    cout << "Augmented matrix before elimation = " << endl << a << endl;
         
    vec x;
    x = gauss_elimination(a);
    
    cout << "Augmented matrix after elimation = " << endl << a << endl;
    cout << "x = " << endl << x << endl;
    
    mat b(4, 5);
    b << 1 << -1 << 2 << -1 << -8 << endr
      << 2 << -2 << 3 << -3 << -20 << endr
      << 1 << 1 << 1 << 0 << -2 << endr
      << 1 << -1 << 4 << 3 << 4 << endr;

    cout << "Augmented matrix before elimation = " << endl << b << endl;

    vec y;
    y = gauss_elimination(b);
    
    cout << "Augmented matrix after elimation = " << endl << b << endl;
    cout << "y = " << endl << y << endl;
    
    /* gauss-jordan */
    mat c(3, 4);
    
    c << 1  << 2  << 1 <<   8 << endr
      << -4 << 7  << 3 <<  19 << endr
      << 11 << -1 << 2 << 15  << endr;
    
    cout << "Augmented matrix before gauss-jordan = " << endl << c << endl;
         
    vec z;
    z = gauss_jordan(c);
    
    cout << "Augmented matrix after gauss-jordan = " << endl << c << endl;
    cout << "z = " << endl << z << endl;
    
    mat d(4, 5);
    d << 1 << -1 << 2 << -1 << -8 << endr
      << 2 << -2 << 3 << -3 << -20 << endr
      << 1 << 1 << 1 << 0 << -2 << endr
      << 1 << -1 << 4 << 3 << 4 << endr;

    cout << "Augmented matrix before gauss-jordan = " << endl << d << endl;

    vec w;
    w = gauss_jordan(d);
    
    cout << "Augmented matrix after gauss-jordan = " << endl << d << endl;
    cout << "w = " << endl << w << endl;
    
    /* sanity checks */
    ASSERT_VECTOR_EQUAL(x, z, 1e-5);
    ASSERT_VECTOR_EQUAL(y, w, 1e-5);

    return 0;
}
