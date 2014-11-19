# ------------------ Global setup ------------------------
set terminal pngcairo \
    transparent enhanced font "arial,14" fontscale 1.0 size 1600, 1200
set key right box


# ------------------ Analytical, N16 ---------------------
set title 'Analytical Error Convergence, N=16'
set ylabel 'Error'
set xlabel 'Iterations'
set output 'h16_analytical-error.png'
plot \
    'h16_jacobi.dsv' using 1:2 title 'Jacobi', \
    'h16_gauss.dsv' using 1:2 title 'Gauss-Seidel', \
    'h16_gsor.dsv' using 1:2 title 'Gauss-Sediel SOR=1.35', \
    'h16_conj_grad.dsv' using 1:2 title 'Conjugate Gradient', \
    'h16_precond_conj_grad.dsv' using 1:2 \
        title 'Conjugate Gradient w/ Preconditioner'


# ------------------ Residual, N16 ---------------------
set title 'Residual Error Convergence, N=16'
set ylabel 'Error'
set xlabel 'Iterations'
set output 'h16_residual-error.png'
plot \
    'h16_jacobi.dsv' using 1:2 title 'Jacobi', \
    'h16_gauss.dsv' using 1:2 title 'Gauss-Seidel', \
    'h16_gsor.dsv' using 1:2 title 'Gauss-Sediel SOR=1.35', \
    'h16_conj_grad.dsv' using 1:2 title 'Conjugate Gradient', \
    'h16_precond_conj_grad.dsv' using 1:2 title \
        'Conjugate Gradient w/ Preconditioner'

# ------------------ Analytical, N64 ---------------------
set title 'Analytical Error Convergence, N=64'
set ylabel 'Error'
set xlabel 'Iterations'
set output 'h64_analytical-error.png'
plot \
    'h64_jacobi.dsv' using 1:2 title 'Jacobi', \
    'h64_gauss.dsv' using 1:2 title 'Gauss-Seidel', \
    'h64_gsor.dsv' using 1:2 title 'Gauss-Sediel SOR=1.35', \
    'h64_conj_grad.dsv' using 1:2 title 'Conjugate Gradient', \
    'h64_precond_conj_grad.dsv' using 1:2 \
        title 'Conjugate Gradient w/ Preconditioner'


# ------------------ Residual, N64 ---------------------
set title 'Residual Error Convergence, N=64'
set ylabel 'Error'
set xlabel 'Iterations'
set output 'h64_residual-error.png'
plot \
    'h64_jacobi.dsv' using 1:2 title 'Jacobi', \
    'h64_gauss.dsv' using 1:2 title 'Gauss-Seidel', \
    'h64_gsor.dsv' using 1:2 title 'Gauss-Sediel SOR=1.35', \
    'h64_conj_grad.dsv' using 1:2 title 'Conjugate Gradient', \
    'h64_precond_conj_grad.dsv' using 1:2 title \
        'Conjugate Gradient w/ Preconditioner'