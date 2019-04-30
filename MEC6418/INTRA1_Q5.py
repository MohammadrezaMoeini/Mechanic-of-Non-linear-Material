# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:01:22 2018

@author: momoe
"""

(I,J,K)=I_J_K_4orten_isotropes()

A=11*J + 8*K
A_V=tensor4_to_Voigt_tensor2(A)
print('A_V',A_V)

(alpha, betha, gamma)=alpha_betha_gamma_4orten_cubiques(A)


print('alpha, betha, gamma',alpha, betha, gamma)

A_V=tensor4_to_Voigt_tensor2(A)
print('A_V_c',A_V)
