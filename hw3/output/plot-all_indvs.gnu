# ------------------ Global setup ------------------------
set terminal pngcairo \
    transparent enhanced font "arial,14" fontscale 1.0 size 1600, 1200
set key right box

# ------------------ Jacobi ---------------------
set title 'Jacobi, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_jacobi_analytical-error.png'
plot 'h16_jacobi.dsv' using 1:2 title 'analytical error'

set title 'Jacobi, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_jacobi_residual-error.png'
plot 'h16_jacobi.dsv' using 1:3 title 'residual error'

set title 'Jacobi, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_jacobi_analytical-error.png'
plot 'h64_jacobi.dsv' using 1:2 title 'analytical error'

set title 'Jacobi, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_jacobi_residual-error.png'
plot 'h64_jacobi.dsv' using 1:3 title 'residual error'

# ------------------ Gauss-Seidel ---------------------
set title 'Gauss-Seidel, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_gauss_analytical-error.png'
plot 'h16_gauss.dsv' using 1:2 title 'analytical error'

set title 'Gauss-Seidel, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_gauss_residual-error.png'
plot 'h16_gauss.dsv' using 1:3 title 'residual error'

set title 'Gauss-Seidel, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_gauss_analytical-error.png'
plot 'h64_gauss.dsv' using 1:2 title 'analytical error'

set title 'Gauss-Seidel, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_gauss_residual-error.png'
plot 'h64_gauss.dsv' using 1:3 title 'residual error'

# ------------------ Gauss-Seidel SOR=1.35 ---------------------
set title 'Gauss-Seidel SOR=1.35, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_gsor_analytical-error.png'
plot 'h16_gsor.dsv' using 1:2 title 'analytical error'

set title 'Gauss-Seidel SOR=1.35, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_gsor_residual-error.png'
plot 'h16_gsor.dsv' using 1:3 title 'residual error'

set title 'Gauss-Seidel SOR=1.35, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_gsor_analytical-error.png'
plot 'h64_gsor.dsv' using 1:2 title 'analytical error'

set title 'Gauss-Seidel SOR=1.35, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_gsor_residual-error.png'
plot 'h64_gsor.dsv' using 1:3 title 'residual error'

# ------------------ Conjugate Gradient ---------------------
set title 'Conjugate Gradient, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_conj_grad_analytical-error.png'
plot 'h16_conj_grad.dsv' using 1:2 title 'analytical error'

set title 'Conjugate Gradient, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_conj_grad_residual-error.png'
plot 'h16_conj_grad.dsv' using 1:3 title 'residual error'

set title 'Conjugate Gradient, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_conj_grad_analytical-error.png'
plot 'h64_conj_grad.dsv' using 1:2 title 'analytical error'

set title 'Conjugate Gradient, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_conj_grad_residual-error.png'
plot 'h64_conj_grad.dsv' using 1:3 title 'residual error'

# ------------------ Conjugate Gradient w/ Preconditioner ---------------------
set title 'Conjugate Gradient w/ Preconditioner, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_precond_conj_grad_analytical-error.png'
plot 'h16_precond_conj_grad.dsv' using 1:2 title 'analytical error'

set title 'Conjugate Gradient w/ Preconditioner, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_precond_conj_grad_residual-error.png'
plot 'h16_precond_conj_grad.dsv' using 1:3 title 'residual error'

set title 'Conjugate Gradient w/ Preconditioner, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_precond_conj_grad_analytical-error.png'
plot 'h64_precond_conj_grad.dsv' using 1:2 title 'analytical error'

set title 'Conjugate Gradient w/ Preconditioner, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_precond_conj_grad_residual-error.png'
plot 'h64_precond_conj_grad.dsv' using 1:3 title 'residual error'