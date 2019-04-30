# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:39:41 2018

@author: hp
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


print('==============================================')
print('Problème 07 – 03 : Le schéma auto-cohérent')
print('==============================================')


print('----------------------------------------------')
print('Le cas simple isotrope')
print('----------------------------------------------')



E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
c1=30./100. #30%
k1= Ef/(3.*(1.-2.*vf)) 
mu1= Ef/(2.*(1.+vf)) 




def Auto_cohérent_sph(k0,mu0,k1,mu1,c1,iteration,error):
   
    (k_initial, mu_initial, C_old)=Mori_Tanaka_sphériques(k0,mu0,[k1],[mu1],[c1])

    k=np.zeros((iteration,1))
    mu=np.zeros((iteration,1))
    alpha_SE=np.zeros((iteration,1))
    betha_SE=np.zeros((iteration,1))
    
    k[0]=k_initial
    mu[0]=mu_initial
        
    for i in range(1,iteration):
        
        (alpha_SE[i-1],betha_SE[i-1])=Eshelby_tensor_Spherical_alpha_betha(k[i-1],mu[i-1])
        
        k[i]=k0 + c1*((k[i-1])*(k1-k0)) / (k[i-1] + alpha_SE[i-1]*(k1-k[i-1]))
        
        k[0]
        mu[i]=mu0 + c1*((mu[i-1])*(mu1-mu0)) / (mu[i-1] + betha_SE[i-1]*(mu1 - mu[i-1]))
        
                
        error_k= k[i] - k[i-1]        
        error_mu=mu[i]-mu[i-1]        
        error_vector=np.array([error_k,error_mu])
        norm_error=np.linalg.norm(error_vector)
#        print(norm_error)
        if norm_error< error:       
            
            k_h=float(k[i])
            mu_h=float(mu[i])
            
            print('k_homogen=',k_h,'GPa')
            print('mu_homogen=',mu_h,'GPa')
            print('number of iteration=',i)
            print('norm_error=',norm_error)
            break
        else:
            pass
           
    return k_h, mu_h   
      
(k,mu)=Auto_cohérent_sph(k0,mu0,k1,mu1,c1,150,1e-12)



print('\n')
print('----------------------------------------------')
print('Le cas isotrope transverse')
print('----------------------------------------------')


E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
c1=30./100.
k1= Ef/(3.*(1.-2.*vf)) 
mu1= Ef/(2.*(1.+vf)) 

a1=1.; a2=1.; a3=1.0e4

error=(1e-5)#*(1e9)
excel_file_omega='omega_64points.xls'
excel_file_zeta3='zeta3_8points.xls'
iteration=20

def Auto_cohérent_elp(k0,mu0,k1,mu1,c1,excel_file_zeta3, excel_file_omega,a1,a2,a3, iteration,error):
    
    C_V_initial_MT=Mori_Tanaka_ellipsoidales_ONEPHASE(k0,mu0,k1,mu1,c1,excel_file_zeta3, excel_file_omega,a1,a2,a3)
    C4_initial_MT = tensor2_Voigt_to_tensor4_NOT_Major_Sym(C_V_initial_MT)

    (I,J,K)=I_J_K_4orten_isotropes()
    I_V=tensor4_to_Voigt_tensor2(I)
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    C0_V=3.*k0*J_V + 2.*mu0*K_V
    C1_V=3.*k1*J_V + 2.*mu1*K_V

    C4=[]
    C4.append(C4_initial_MT)
    C_V=[]
    C_V.append(C_V_initial_MT)

    
    for i in range(1,iteration):
        
        SE4i=Eshelby_tensor_ellipsoid_C0Input(excel_file_zeta3, excel_file_omega, C4[i-1], a1, a2, a3)
        SEi_V = tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE4i)
    
        A1= inv( I_V + np.matmul(SEi_V, np.matmul(inv(C_V[i-1]), C1_V-C_V[i-1] )) )

        Ci_V= C0_V + np.matmul(c1*(C1_V-C0_V), A1 )
        C4i= tensor2_Voigt_to_tensor4_NOT_Major_Sym(Ci_V)
    
        C_V.append(Ci_V)
        C4.append(C4i)
    
        ERROR_V = C_V[-1]-C_V[-2]
        max_element_error=abs(ERROR_V.max())
        
        if max_element_error< error: 
        
            C_homogen=C_V[-1]
            
            print('C_homogen=')
            print(C_homogen)
            print('number of iteration=',i)
            print('max_element_error=',max_element_error)
        
            break
        else:
            pass
        
    return C_homogen
    
    
     
C_homogen=Auto_cohérent_elp(k0,mu0,k1,mu1,c1,excel_file_zeta3, excel_file_omega,a1,a2,a3, iteration,error)    



