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

w = [2., 1/35.]
mu=[10., 3., 4.]
k=[0., 0., 0.]

def C_1D(mu):
    C=[]
    for i in range(0,len(k)):
        Ci=np.array([2.*mu[i] ] )
        C.append(Ci)
    return C
     

   
C=C_1D(mu)

(S0, Sm, landa) = interconversion_C_to_S_1D(C,w)

mu0 = 2.*S0
mu1 = 2.*Sm[0]
mu2 = 2.*Sm[1]



from sympy.abc import t
import sympy as sy
from sympy import *
from fractions import Fraction


#mu_t = float(mu0) + float(mu1) - float(mu1)*(sy.exp(-landa1_mu*t)) + float(mu2) - float(mu2)*(sy.exp(-landa2_mu*t))


def mut(mu0, mu, landa):
    'write mu as a list'
    mu_t=0.
    for i in range(0,len(mu)):    
        mu_t =float(mu[i]) -  float(mu[i])*(sy.exp(-landa[i]*t)) + mu_t
        
    return mu_t + float(mu0)
 
mu = [mu1 , mu2]    
mu_t = mut(mu0, mu, landa)








