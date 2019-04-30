# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:13:29 2018

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

#
#landa1_mu=2.; landa2_mu=1./35.
#mu_t = 10. + 1.2*(sy.exp(-landa1_mu*t)) + 0.5*(sy.exp(-landa2_mu*t))


#def laplace_carson( f, t, s ):
#	l = laplace_transform( f, t, s)
#	l_c = sy.simplify( s*l[0] )
#	
#	return l_c
#
#def inverse_laplace_carson( f, t, s):
#	f = sy.simplify( f/s )
#	f = sy.apart( f )
#	f_c = inverse_laplace_transform( f, t, s )
#	
#	return f_c


#landa1_mu=2.;
#landa2_mu=Rational(1, 35)
#mu_t = 10. + 1.2*(sy.exp(-landa1_mu*t)) + 0.5*(sy.exp(-landa2_mu*t))
#mu_s = laplace_carson( mu_t, t, s )
#mu_inter_t = inverse_laplace_carson( mu_s, t, s)
#
#
##mu_t = 1
##mu_s = laplace_carson( mu_t, t, s )
#




#w = [2., 1/35.]
#mu=[10., 3., 4.]
#k=[0., 0., 0.]
#
#
#def C_1D(k,mu):
#    C=[]
#    for i in range(0,len(k)):
#        Ci=np.array([3.*k[i] + 2.*mu[i] ] )
#        C.append(Ci)
#    return C
#        
#C=C_1D(k,mu)
#
#(B,L1,L2,L3,N,D) = B_L1_L2_L3(C,w)
#    
#A1=np.array([1./L1[0]])


#A2=np.matmul(np.array([1./L1[0]]),L2)
#A3=L3 + np.matmul( transpose(L2),  np.array([1./L1[0]])*L2  )
#    
#(U,d,V)=np.linalg.svd(A3)
#Dp=diag(d)
#P=U
#    
#B_star = np.matmul( transpose(P), np.matmul( B, P ) )
#A3_star = Dp
#A2_star = np.matmul( A2, P )
#    
#S0=A1
#    
#landa=[]
#for i in range(0,len(A3_star)):
#    landa.append(A3_star[i,i])
#
#Sm=[]
#for m in range(0,D*N):
#    S=np.zeros((D,D))
#    for i in range(0,D):
#        for j in range(0,D):
#            S[i,j]=A2_star[i,m]*A2_star[j,m]/landa[m]
#    Sm.append(S)


w = [2., 1/35.]
#mu=[np.array(10.), np.array(3.), np.array(4.)]


mu=[np.array([10.]),
 np.array([3.]),
 np.array([4.])]



k=[0., 0., 0.]
#(S0, Sm, landa)=interconversion_C_to_S_1D(mu,w)



#(B,L1,L2,L3,N,D) = B_L1_L2_L3(mu,w)
#    
#A1=np.array([1./L1[0]])
#A2=np.matmul(np.array([1./L1[0]]),L2)
#A3=L3 + np.matmul( transpose(L2),  np.array([1./L1[0]])*L2  )
#    
#(U,d,V)=np.linalg.svd(A3)
#Dp=diag(d)
#P=U
#    
#B_star = np.matmul( transpose(P), np.matmul( B, P ) )
#A3_star = Dp
#A2_star = np.matmul( A2, P )
#    
#S0=A1
#    
#landa=[]
#for i in range(0,len(A3_star)):
#    landa.append(A3_star[i,i])
#
#Sm=[]
#for m in range(0,D*N):
#    S=np.zeros((D,D))
#    for i in range(0,D):
#        for j in range(0,D):
#            S[i,j]=A2_star[i,m]*A2_star[j,m]/landa[m]
#    Sm.append(S)





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
#
#(C01D, Cn1D, w1D)=interconversion_S_to_C_1D(S1D,landa)
#(alpha_C, betha_C)=alphai_bethai_for_S_or_C_1D(Cn1D)
#(alpha0_C, betha0_C)= alphs0_betha0_for_S0_or_C0_1D(C01D)
#
#



# =============================================================================
# 
# =============================================================================

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


#N=len(w)
#B=np.identity(N)
#A1=1./L1
#a2=(1./L1) *transpose(l2)
#
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
#a2_star = np.matmul( transpose(P), transpose(a2) )
#A3_star = np.matmul( transpose(P), np.matmul( A3, P ) )

# =============================================================================
# 
# =============================================================================

#L3=diag(w)

#L3=diag(w)




# =============================================================================
# We cannot solve this problem with our 3D function, because the matrix is not positive definite
# =============================================================================
w = [2., 1/35.]
mu=[10., 3., 4.]
k=[0., 0., 0.]


def C(k,mu):
    C=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)    
    for i in range(0,len(k)):

        Ci=3.*k[i]*J_V + 2.*mu[i]*K_V
        C.append(Ci)
    return C

C=C(k,mu)

(S0,Sm,landa)=interconversion_C_to_S(C,w)





