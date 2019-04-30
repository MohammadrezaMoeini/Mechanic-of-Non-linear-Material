# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:09:06 2018

@author: momoe
"""

from numpy import *
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from scipy.optimize import least_squares
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time



from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import * #(the difference is just for importing)
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Function_LAB_interconversion_25102018 import *
from Viscoelastic_Linear_Function_LAB_1D_interconversion_09112018 import *

from Homogenization_Function_S11_Eshelby_16112018py import *
from Homogenization_Function_S12_diluée_18112018py import *
from Homogenization_Function_S12_Mori_Tanaka_18112018py import *

from Homogenization_Function_S12_Voigt_Reuss_19112018py import *



from sympy.integrals import laplace_transform, inverse_laplace_transform
from sympy.abc import t, s, X
import sympy as sy
from sympy import *
from fractions import Fraction


landa1_mu=2.; landa2_mu=1./35.
mu_t = 10. + 3.*(sy.exp(-landa1_mu*t)) + 4.*(sy.exp(-landa2_mu*t))

def laplace_carson( f, t, s ):
	l = laplace_transform( f, t, s)
	l_c = sy.simplify( s*l[0] )
	
	return l_c

mu_Laplace_Carson = laplace_carson( mu_t, t, s )
#mu_Laplace_Carson_inv = mu_Laplace_Carson**(-1)
#mu_Laplace_inv =sy.simplify( mu_Laplace_Carson_inv/s )  
#mu_t_inv = inverse_laplace_transform( mu_Laplace_inv, t, s )

def inverse_laplace_carson( f, t, s):
	f = sy.simplify( f/s )
	f = sy.apart( f )
	f_c = inverse_laplace_transform( f, t, s )
	
	return f_c

mu_t_inv = inverse_laplace_carson( mu_Laplace_Carson, t, s)

# =============================================================================
# I can't solve this problem in the Python. I wrote a solution in MATLAB
# =============================================================================




mu=[np.array([10.]),
 np.array([3.]),
 np.array([4.])]

w = [2., 1/35.]

w = [2., 1/35.]
mu=[10., 3., 4.]

def L1_1D(C):
    L1=0. + C[0]
    for i in range(1,len(C)):
        L1=L1 + C[i]
    return L1

L1=L1_1D(mu)

def l_2_1D(C,w):
    N=len(w)
    l2=np.zeros((N,1))
    for i in range(0,N):
        l2[i]=np.sqrt(w[i]*C[i+1])
    return l2
l2=l_2_1D(mu,w)

def L3_1D(w):
    N=len(w)
    L3=np.zeros((N,N))
    for i in range(0,N):
        L3[i][i]=w[i]
    return L3

L3=L3_1D(w)



N=len(w)
B=np.identity(N) #ok
A1=1./L1 #ok 
a2=(1./L1) *transpose(l2) #ok

l2_l2=np.zeros((len(l2),len(l2)))
for i in range(0,len(l2)):
    for j in range(0,len(l2)):
        l2_l2[i][j]=l2[i]*l2[j]

A3=L3 - (1./L1)*l2_l2
(U,d,V)=np.linalg.svd(A3)
P=U

A3_star=  np.diag(d) 

# =============================================================================
# Just convert to np.array
# =============================================================================
Pm = np.zeros(shape(P))
for i in range(0,len(P)):
    for j in range(0,len(P)):
        Pm[i][j] = P[i][j]

a2m = np.zeros((len(a2)) )
for i in range(0,len(a2)):
    a2m[i] = a2[i]
#==============================================================================
#def transpose_row_to_column (x):
#    y=np.zeros((len(x),1))
#    
#    for i in range(0,len(x)):
#        y[i]=x[i]
#    return y
#
#a2m_T = transpose_row_to_column (a2m)    
#a2_star = np.matmul(transpose(Pm), transpose(a2m))    
#

a2_star = np.zeros((len(a2)) )
for i in range(0,len(P)):
    for j in range(0,len(P)):
        a2_star[i] = P[i][j] * a2m[i]







#a2=np.zeros(shape(transpose(l2)))
#for i in range(0,N):
#    a2[1,i]=(1./L1) *l2[i]
#
#l2_l2=np.zeros((len(l2),len(l2)))
#for i in range(0,len(l2)):
#    for j in range(0,len(l2)):
#        l2_l2[i][j]=l2[i]*l2[j]
#
#A3=L3 - (1./L1)*l2_l2
#(U,d,V)=np.linalg.svd(A3)
#
#P=U
#A3_star=np.diag(d)
#
#
#





