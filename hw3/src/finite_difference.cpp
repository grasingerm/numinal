#include "finite_difference.hpp"

using namespace arma;

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

    for (auto i = uint_fast32_t {0}; i < n; i++)
        for (auto j = uint_fast32_t {0}; j < m; j++)
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

    for (auto i = uint_fast32_t {0}; i < n; i++)
        for (auto j = uint_fast32_t {0}; j < m; j++)
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

    for (auto j = uint_fast32_t {0}; j < n; j++)
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
    
    for (auto i = uint_fast32_t {0}; i < m; i++)
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
