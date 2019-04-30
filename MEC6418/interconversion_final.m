

clc 
syms s t

mu  = 10 + 3*exp(-2*t) + 4*exp(-t/35)

mu_laplace = laplace(mu,t,s)

mu_laplace_carson = mu_laplace * s


mu_laplace_carson_inv = (mu_laplace_carson)^(-1)

mu_t = ilaplace(mu_laplace_carson_inv)