#include "finite_difference.hpp"

using namespace std;
using namespace arma;

/*
 *
 */
tuple < mat, vec > 
create_grid_rectangle_helmholtz (const unsigned int m, const unsigned int n, 
    const double h, const double delta, const double boundary_value)
{
    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };

    mat A (n*m,n*m);
    vec b (n*m);
    b.fill(delta);
        
    const auto delta_div_hsq = delta / (h*h);

    /* set up grid for inner nodes */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
        for (auto j = uint_fast32_t {1}; j < m-1; j++)
        {
            auto center = map (i,j);

            A(center, center) = -(4*delta_div_hsq + 1);
            A(center, map (i+1,j)) = delta_div_hsq;
            A(center, map (i,j+1)) = delta_div_hsq;
            A(center, map (i-1,j)) = delta_div_hsq;
            A(center, map (i,j-1)) = delta_div_hsq;
        }

    /* bottom boundary */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
    {
        auto j = uint_fast32_t {0};
        auto center = map (i,j);

        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i+1,j)) = delta_div_hsq;
        A(center, map (i-1,j)) = delta_div_hsq;
        A(center, map (i,j+1)) = delta_div_hsq;

        b(center) -= boundary_value;
    }

    /* top boundary */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
    {
        auto j = uint_fast32_t {m-1};
        auto center = map (i,j);

        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i+1,j)) = delta_div_hsq;
        A(center, map (i-1,j)) = delta_div_hsq;
        A(center, map (i,j-1)) = delta_div_hsq;

        b(center) -= boundary_value;
    }

    /* west boundary */
    for (auto j = uint_fast32_t {1}; j < m-1; j++)
    {
        auto i = uint_fast32_t {0};
        auto center = map (i,j);

        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j+1)) = delta_div_hsq;
        A(center, map (i,j-1)) = delta_div_hsq;
        A(center, map (i+1,j)) = delta_div_hsq;

        b(center) -= boundary_value;
    }

    /* east boundary */
    for (auto j = uint_fast32_t {1}; j < m-1; j++)
    {
        auto i = uint_fast32_t {n-1};
        auto center = map (i,j);

        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j+1)) = delta_div_hsq;
        A(center, map (i,j-1)) = delta_div_hsq;
        A(center, map (i-1,j)) = delta_div_hsq;

        b(center) -= boundary_value;
    }

    /* southwest corner */
    {
        auto i = uint_fast32_t {0};
        auto j = uint_fast32_t {0};
        auto center = map (i, j);
        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j+1)) = delta_div_hsq;
        A(center, map (i+1,j)) = delta_div_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* southeast corner */
    {
        auto i = uint_fast32_t {n-1};
        auto j = uint_fast32_t {0};
        auto center = map (i, j);
        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j+1)) = delta_div_hsq;
        A(center, map (i-1,j)) = delta_div_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* northwest corner */
    {
        auto i = uint_fast32_t {0};
        auto j = uint_fast32_t {m-1};
        auto center = map (i, j);
        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j-1)) = delta_div_hsq;
        A(center, map (i+1,j)) = delta_div_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* northeast corner */
    {
        auto i = uint_fast32_t {n-1};
        auto j = uint_fast32_t {m-1};
        auto center = map (i, j);
        A(center, center) = -(4*delta_div_hsq + 1);
        A(center, map (i,j-1)) = delta_div_hsq;
        A(center, map (i-1,j)) = delta_div_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    return tuple < mat, vec > (A, b);
}

/*
 *
 */
tuple < mat, vec > 
create_grid_rectangle_laplace(const unsigned int m, const unsigned int n,
    const double h, const double boundary_value)
{
    auto map = [&] (unsigned int i, unsigned int j) -> 
        unsigned int { return i*n + j; };

    mat A (n*m,n*m);
    vec b (n*m);
    b.fill(1.);
        
    const auto inv_hsq = 1 / (h*h);

    /* set up grid for inner nodes */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
        for (auto j = uint_fast32_t {1}; j < m-1; j++)
        {
            auto center = map (i,j);

            A(center, center) = -4*inv_hsq;
            A(center, map (i+1,j)) = inv_hsq;
            A(center, map (i,j+1)) = inv_hsq;
            A(center, map (i-1,j)) = inv_hsq;
            A(center, map (i,j-1)) = inv_hsq;
        }

    /* bottom boundary */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
    {
        auto j = uint_fast32_t {0};
        auto center = map (i,j);

        A(center, center) = -4*inv_hsq;
        A(center, map (i+1,j)) = inv_hsq;
        A(center, map (i-1,j)) = inv_hsq;
        A(center, map (i,j+1)) = inv_hsq;

        b(center) -= boundary_value;
    }

    /* top boundary */
    for (auto i = uint_fast32_t {1}; i < n-1; i++)
    {
        auto j = uint_fast32_t {m-1};
        auto center = map (i,j);

        A(center, center) = -4*inv_hsq;
        A(center, map (i+1,j)) = inv_hsq;
        A(center, map (i-1,j)) = inv_hsq;
        A(center, map (i,j-1)) = inv_hsq;

        b(center) -= boundary_value;
    }

    /* west boundary */
    for (auto j = uint_fast32_t {1}; j < m-1; j++)
    {
        auto i = uint_fast32_t {0};
        auto center = map (i,j);

        A(center, center) = -4*inv_hsq;
        A(center, map (i,j+1)) = inv_hsq;
        A(center, map (i,j-1)) = inv_hsq;
        A(center, map (i+1,j)) = inv_hsq;

        b(center) -= boundary_value;
    }

    /* east boundary */
    for (auto j = uint_fast32_t {1}; j < m-1; j++)
    {
        auto i = uint_fast32_t {n-1};
        auto center = map (i,j);

        A(center, center) = -4*inv_hsq;
        A(center, map (i,j+1)) = inv_hsq;
        A(center, map (i,j-1)) = inv_hsq;
        A(center, map (i-1,j)) = inv_hsq;

        b(center) -= boundary_value;
    }

    /* southwest corner */
    {
        auto i = uint_fast32_t {0};
        auto j = uint_fast32_t {0};
        auto center = map (i, j);
        A(center, center) = -4*inv_hsq;
        A(center, map (i,j+1)) = inv_hsq;
        A(center, map (i+1,j)) = inv_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* southeast corner */
    {
        auto i = uint_fast32_t {n-1};
        auto j = uint_fast32_t {0};
        auto center = map (i, j);
        A(center, center) = -4*inv_hsq;
        A(center, map (i,j+1)) = inv_hsq;
        A(center, map (i-1,j)) = inv_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* northwest corner */
    {
        auto i = uint_fast32_t {0};
        auto j = uint_fast32_t {m-1};
        auto center = map (i, j);
        A(center, center) = -4*inv_hsq;
        A(center, map (i,j-1)) = inv_hsq;
        A(center, map (i+1,j)) = inv_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    /* northeast corner */
    {
        auto i = uint_fast32_t {n-1};
        auto j = uint_fast32_t {m-1};
        auto center = map (i, j);
        A(center, center) = -4*inv_hsq;
        A(center, map (i,j-1)) = inv_hsq;
        A(center, map (i-1,j)) = inv_hsq;

        b(center) -= (boundary_value + boundary_value);
    }

    return tuple < mat, vec > (A, b);
}