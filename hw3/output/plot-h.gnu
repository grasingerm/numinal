# ------------------ Global setup ------------------------
set terminal pngcairo \
    transparent enhanced font "arial,14" fontscale 1.0 size 1600, 1200
set key right box

# ------------------ Analytical Error, N=16 ---------------------
set title 'Conjugate Gradient w/ Preconditioner Convergence: \
    Analytical Error, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_part-h_analytical-error.png'
plot 'h16_precond_conj_grad.dsv' using 1:2 title 'eps=1e-4' with line, \
    'h16_precond_conj_grad_1e-6.dsv' using 1:2 title 'eps=1e-6'

# ------------------ Relative Error, N=16 ---------------------
set title 'Conjugate Gradient w/ Preconditioner Convergence: \
    Relative Error, N=16'
set ylabel 'error'
set xlabel 'iterations'
set output 'h16_part-h_relative-error.png'
plot 'h16_precond_conj_grad.dsv' using 1:3 title 'eps=1e-4' with line, \
    'h16_precond_conj_grad_1e-6.dsv' using 1:3 title 'eps=1e-6'

# ------------------ Analytical Error, N=64 ---------------------
set title 'Conjugate Gradient w/ Preconditioner Convergence: \
    Analytical Error, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_part-h_analytical-error.png'
plot 'h64_precond_conj_grad.dsv' using 1:2 title 'eps=1e-4' with line, \
    'h64_precond_conj_grad_1e-6.dsv' using 1:2 title 'eps=1e-6'

# ------------------ Relative Error, N=64 ---------------------
set title 'Conjugate Gradient w/ Preconditioner Convergence: \
    Relative Error, N=64'
set ylabel 'error'
set xlabel 'iterations'
set output 'h64_part-h_relative-error.png'
plot 'h64_precond_conj_grad.dsv' using 1:3 title 'eps=1e-4' with line, \
    'h64_precond_conj_grad_1e-6.dsv' using 1:3 title 'eps=1e-6'