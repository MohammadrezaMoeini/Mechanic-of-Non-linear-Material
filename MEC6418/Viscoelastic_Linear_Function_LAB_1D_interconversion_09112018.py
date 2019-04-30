# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:07:36 2018

@author: momoe
"""


import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv
from array import *

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


from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import *

#
## =============================================================================
## 1D
## =============================================================================
#
##S01D=0.1*np.array([210.174])
##S11D=0.1*np.array([16.997])
##S1D=[]
##S1D.append(S01D)
##S1D.append(S11D)
##landa=[0.043145]
##(B,A1,A2,A3,M,D)=B_A1_A2_A3(S1D,landa)
#

# =============================================================================
# Using 4-order tensor
# =============================================================================
def alpha_betha_1D_isotropes(A):
    '''tenseur isotrope  ---------> alph et betha [1D]'''
    J=(1./3.)*np.array([1.])
    K=(1./3.)*np.array([2.])
    
    alpha=J*A
    betha=K*A
    
    return(alpha, betha)
    
    

def alphai_bethai_for_S_or_C_1D(C_S):
    alpha=[]; betha=[];
    for i in range(0,len(C_S)):
        (alphai, bethai)=alpha_betha_1D_isotropes(C_S[i])
        alpha.append(alphai)
        betha.append(bethai)
    return alpha, betha
        


def alphs0_betha0_for_S0_or_C0_1D(C0_S0):
    (alpha0, betha0)=alpha_betha_1D_isotropes(C0_S0)
    
    return alpha0, betha0
    
    


def interconversion_S_to_C_1D(S,landa):
    (B,A1,A2,A3,M,D)=B_A1_A2_A3(S,landa)
    
    L1=np.array([1./A1[0]])
    L2=np.matmul(np.array([1./A1[0]]),A2)
    L3=A3 + np.matmul( transpose(A2),  np.array([1./A1[0]])*A2  )
    
    (U,d,V)=np.linalg.svd(L3)
    Dp=diag(d)
    P=U
    
    B_star = np.matmul( transpose(P), np.matmul( B, P ) )
    L3_star = Dp
    L2_star = np.matmul( L2, P )
        
    w=[]
    for i in range(0,len(L3_star)):
        w.append(L3_star[i,i])    
    
    Cn=[]
    for n in range(0,D*M):
        C=np.zeros((D))
        C[0]=L2_star[n]*L2_star[n]/w[n]
        Cn.append(C)
    
    C_total=np.zeros((D,D))
    for n in range(0,D*M):
        C_total=Cn[n] + C_total
    
    C0=L1 - C_total
    
    
    print('''Pay attention, If you have mu and k; put individually.
          They are coeficients of orthogonal tensors and they are independent.
          Then
          mu0 = 2.*C0, mu_i = 2.*C[i]
          ku0 = 3.*C0, k_i = 3.*C[i]
          
          mu(t) = mu0 + mu1 (1.-exp(-landa[1]t)) + mu2 (1.-exp(-landa[2]t)) + ...
          k(t) = k0 + k1 (1.-exp(-landa[1]t)) + k2 (1.-exp(-landa[2]t)) + ...
          
          Check Problem 05_06 for writing mu(t) & k(t)
          
          ''')
    
    return C0, Cn, w





def interconversion_C_to_S_1D(C,w):
    (B,L1,L2,L3,N,D) = B_L1_L2_L3(C,w)
    
    A1=np.array([1./L1[0]])
    A2=np.matmul(np.array([1./L1[0]]),L2)
    A3=L3 - np.matmul( transpose(L2),  np.array([1./L1[0]])*L2  )
    
    (U,d,V)=np.linalg.svd(A3)
    Dp=diag(d)
    P=U
    
    B_star = np.matmul( transpose(P), np.matmul( B, P ) )
    A3_star = Dp
    A2_star = np.matmul( A2, P )
    
    S0=A1
    
    landa=[]
    for i in range(0,len(A3_star)):
        landa.append(A3_star[i,i])

    Sm=[]
    for m in range(0,D*N):
        S=np.zeros((D,D))
        for i in range(0,D):
            for j in range(0,D):
                S[i,j]=A2_star[i,m]*A2_star[j,m]/landa[m]
        Sm.append(S)
    print('''Pay attention, If you have mu and k; put individually.
          They are coeficients of orthogonal tensors and they are independent.
          Then
          mu0 = 2.*S0, mu_i = 2.*S[i]
          ku0 = 3.*S0, ku_i = 3.*S[i]
          
          mu(t) = mu0 + mu1 (1.-exp(-landa[1]t)) + mu2 (1.-exp(-landa[2]t)) + ...
          k(t) = k0 + k1 (1.-exp(-landa[1]t)) + k2 (1.-exp(-landa[2]t)) + ...
          
          Check Problem 05_06 for writing mu(t) & k(t)
          
          ''')
    return S0, Sm, landa







# =============================================================================
# FOR 1D YOU SHOULD WRITE C with alpha and betha, you will ge alpha and betha for S
    #Pay attention the new alpha and betha will be the same notation.
# =============================================================================
#w = [2., 1/35.]
#mu=[10., 3., 4.]
#k=[0., 0., 0.]
#
#def C_1D(mu):
#    C=[]
#    for i in range(0,len(k)):
#        Ci=np.array([2.*mu[i] ] )
#        C.append(Ci)
#    return C
#     
#
#   
#C=C_1D(mu)
#
#(S0, Sm, landa) = interconversion_C_to_S_1D(C,w)
#
#mu0 = 2.*S0
#mu1 = 2.*Sm[0]
#mu2 = 2.*Sm[1]














#       
#S1D=[np.array([0.00025575]),
# np.array([2.18626247e-05]),
# np.array([6.40739349e-12]),
# np.array([8.47393463e-12]),
# np.array([2.062202e-05]),
# np.array([9.86679123e-07]),
# np.array([1.48255479e-05]),
# np.array([4.53150959e-14]),
# np.array([1.50929123e-05]),
# np.array([1.13961298e-05])]
#
#landa= [1.0,
# 0.31622776601683794,
# 0.1,
# 0.03162277660168379,
# 0.01,
# 0.003162277660168379,
# 0.001,
# 0.00031622776601683794,
# 0.0001]

#
#(C01D, Cn1D, w1D)=interconversion_S_to_C_1D(S1D,landa)
#(alpha_C, betha_C)=alphai_bethai_for_S_or_C_1D(Cn1D)
#(alpha0_C, betha0_C)= alphs0_betha0_for_S0_or_C0_1D(C01D)
#
#

#(S01D, Sn1D, lambda1D)=interconversion_C_to_S_1D(S1D,landa)


