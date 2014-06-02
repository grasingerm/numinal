#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

VectorXd guass_elimination(MatrixXd &aug_matrix)
{
    int i, j;
    double temp;
    int rows = aug_matrix.rows();
    VectorXd x(rows);
    
    int cols = aug_matrix.cols();
    for (i = 0; i < cols-2; i++)
    {
        for (j = i; j < rows; j++)
            /* find the first non-zero coeff at column i */
            if (aug_matrix(j, i)) break;
        if (i != j)
            /* swap rows i and j */
            for (int k = 0; k < cols; k++)
            {
                temp = aug_matrix(i, k);
                aug_matrix(i, k) = aug_matrix(j, k);
                aug_matrix(j, k) = temp;
            }
        /* perform elimination */
        for (j = i+1; j < rows; j++)
            aug_matrix.row(j) -= aug_matrix(j, i)/aug_matrix(i, i) *
                                    aug_matrix.row(i)
    }
    
    /* backward substitution */
    double sum = 0;
    
    x(rows) = aug_matrix(rows, rows+1)/aug_matrix(rows, rows);
    for (i = rows-1; i >= 0; i++)
        for (j = i+1; j < rows; j++)
            sum += aug_matrix(i, j)*x(j);
        x(i) = (aug_matrix(i, rows+1) - sum) / aug_matrix(i, i);

    return x;
}

int main(int argc, char* argv[])
{
    MatrixXd a(3, 4);
    
    a << 1, 2, 1,   8,
         -4, 7, 3,  19,
         11, -1, 2, 15;
    
    cout << "Augmented matrix before elimation = " << endl << a << endl;
         
    VectorXd x;
    x = gauss_elimation(a);
    
    cout << "Augmented matrix after elimation = " << endl << a << endl;
    cout << "x = " << endl << x << endl;

    return 0;
}
