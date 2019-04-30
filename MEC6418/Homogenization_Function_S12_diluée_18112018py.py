# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:16:38 2018

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



def C_Voigt_k_mu_homogen(k,mu):
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)    
    C_V=3.*k*J_V + 2.*mu*K_V
    return C_V

def SE_sphériques_alpa_betha(k0,mu0):
    alpha_sph_Ef=(3.*k0)/(3.*k0+4.*mu0)
    betha_sph_Ef=(6.*(k0+2.*mu0)) / (5.*(3.*k0+4.*mu0))
    return alpha_sph_Ef, betha_sph_Ef


def diluée_sphériques(k0,mu0,kfr,mufr,cfr):
    
    n=len(kfr) #number of phases
    Ar_alpha = np.zeros((n))
    Ar_betha = np.zeros((n))
    
    (alpha_sph_Ef, betha_sph_Ef)=SE_sphériques_alpa_betha(k0,mu0)

    for i in range(0,n):
        Ar_alpha[i]= k0 / (k0 +  (alpha_sph_Ef)*(kfr[i]-k0) )
        Ar_betha[i]= mu0 / (mu0 + (betha_sph_Ef)*(mufr[i]-mu0) )
    
    sum_k=0.
    sum_mu=0.
    for r in range(0,n):
        sum_k= cfr[r]*(kfr[r]-k0)*Ar_alpha[r] + sum_k
        sum_mu= cfr[r]*(mufr[r]-mu0)*Ar_betha[r] + sum_mu
    
    k_homogen = k0 + sum_k
    mu_homogen = mu0 + sum_mu
    
    C_homogen = C_Voigt_k_mu_homogen(k_homogen,mu_homogen)
    
    return k_homogen, mu_homogen, C_homogen
    


##Example  ------------------------------ 
#kfr=[10.,20.,30]
#mufr=[10.,20.,30]
#cfr=[10,10,10]
#k0=1.
#mu0=1.         
#(k_homogen_sph, mu_homogen_sph, C_homogen_sph)=diluée_sphériques(k0,mu0,kfr,mufr,cfr)    
#        
        
    
        
def diluée_ellipsoidales(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    
    n=len(kfr) #number of phases
    Ar_alpha = np.zeros((n))
    Ar_betha = np.zeros((n))
    
    SE_sph_numerical=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1,a2,a3)
    
    (alpha_elp_Ef, betha_elp_Ef)=alpha_betha_4orten_isotropes(SE_sph_numerical)
    
    for i in range(0,n):
        Ar_alpha[i]= k0 / (k0 +  (alpha_elp_Ef)*(kfr[i]-k0) )
        Ar_betha[i]= mu0 / (mu0 + (betha_elp_Ef)*(mufr[i]-mu0) )
    
    sum_k=0.
    sum_mu=0.
    for r in range(0,n):
        sum_k= cfr[r]*(kfr[r]-k0)*Ar_alpha[r] + sum_k
        sum_mu= cfr[r]*(mufr[r]-mu0)*Ar_betha[r] + sum_mu
    
    k_homogen = k0 + sum_k
    mu_homogen = mu0 + sum_mu
    
    C_homogen = C_Voigt_k_mu_homogen(k_homogen,mu_homogen)
    
    return k_homogen, mu_homogen, C_homogen
    

excel_file_omega='omega_32points.xls'
excel_file_zeta3='zeta3_4points.xls'


##Example  ------------------------------ 
#kfr=[10.,20.,30]
#mufr=[10.,20.,30]
#cfr=[10,10,10]
#k0=1.
#mu0=1.  
#a1=1.;a2=1.;a3=10000.       
#(k_homogen_elp, mu_homogen_elp, C_homogen_elp)=diluée_ellipsoidales(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3)    
#      
#print('k_homogen_sph=',k_homogen_sph)                
#print('k_homogen_elp=',k_homogen_elp)                
#print("-------------------------------")
#print('mu_homogen_sph=',mu_homogen_sph)                
#print('mu_homogen_elp=',mu_homogen_elp)                
#





     


