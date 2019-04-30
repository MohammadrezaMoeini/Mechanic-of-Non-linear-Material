# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:05:24 2018

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

print('========================================================')  
print('========================================================')  
print('Problème 07 – 02 : Composites à fibres longues et alignées')                                                               
print('========================================================')  
print('========================================================')  

E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
cfr=0.3
kfr= Ef/(3.*(1.-2.*vf)) 
mufr= Ef/(2.*(1.+vf)) 

a1=1.; a2=1.; a3=1.0e4




print('-------------------------------------------------------')
print('Mori Tanaka')  
print('-------------------------------------------------------')  

print('Result for ONEPHASE-Reinforcement funcion')  
excel_file_omega='omega_512points.xls'
excel_file_zeta3='zeta3_4points.xls'
C_homogen_ONEPHASE=Mori_Tanaka_ellipsoidales_ONEPHASE(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3)
print('C_homogen_ONEPHASE=')
print(C_homogen_ONEPHASE)

print('\n') 

print('Result for MULTI-Reinforcement funcion') 
E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
cfr=[30./100.]
kfr=[ Ef/(3.*(1.-2.*vf)) ]
mufr=[ Ef/(2.*(1.+vf)) ]
a1=[1.]; a2=[1.]; a3=[1.0e4]

excel_file_omega='omega_512points.xls'
excel_file_zeta3='zeta3_4points.xls'
C_homogen_MT=Mori_Tanaka_ellipsoidales(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3)
print('C_homogen_MT=')
print(C_homogen_MT)

print('\n') 

print('-------------------------------------------------------')
print('Voigt')  
print('-------------------------------------------------------')  

E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
cfr=30./100.
kfr= Ef/(3.*(1.-2.*vf)) 
mufr= Ef/(2.*(1.+vf)) 

C_homogen_Voigt=Voigt(k0,mu0,kfr,mufr,cfr)
print('C_homogen_Voigt=')
print(C_homogen_Voigt)


print('-------------------------------------------------------')
print('Reuss')  
print('-------------------------------------------------------')  

E0=3.0
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

Ef=70.0
vf=0.3
cfr=30./100.
kfr= Ef/(3.*(1.-2.*vf)) 
mufr= Ef/(2.*(1.+vf)) 

C_homogen_Reuss=Reuss(k0,mu0,kfr,mufr,cfr)
print('C_homogen_Reuss=')
print(C_homogen_Reuss)




print('-------------------------------------------------------')
print('Engineering constants')  
print('-------------------------------------------------------') 
n=[0,0,1]
print('direction of fibers=[x,y,z]=', n)


def El_Et_vl_vt_Gl_n_isotrope_transverse(C_V_homogen):
    S_homogen_Voigt = inv(C_V_homogen)
    S_4or = tensor2_Voigt_to_tensor4_NOT_Major_Sym(S_homogen_Voigt)
    
    n=[0,0,1] #it's better to input manually
#    if abs(S_homogen_Voigt[0][0]-S_homogen_Voigt[1][1]) <1.0e-6:
#        n=[0,0,1]
#           
#    if abs(S_homogen_Voigt[0][0]==S_homogen_Voigt[2][2]) <1.0e-6:
#        n=[0,1,0]
#
#    if abs(S_homogen_Voigt[1][1]==S_homogen_Voigt[2][2]) <1.0e-6:
#        n=[1,0,0]
           
    (EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
    
    A=S_4or
    alpha=tensor4_contract4_tensor4(EL,A)    
    betha=tensor4_contract4_tensor4(JT,A)
    
    gamma=tensor4_contract4_tensor4(F,A)
    gammap=tensor4_contract4_tensor4(FT,A)
    
    delta=(1./2.)*tensor4_contract4_tensor4(KT,A)
    deltap=(1./2.)*tensor4_contract4_tensor4(KL,A)

    El=1./alpha
    vt=(delta-betha)/(delta+betha)
    Et=(1.-vt)/betha
    vl=gamma*El/(-np.sqrt(2.))
    Gl=1./(2.*deltap)
    
    return El,Et,vl,vt,Gl,n


def El_Et_vl_vt_Gl_n_isotrope(C_V_homogen):
    S_homogen_Voigt = inv(C_V_homogen)
    S_4or = tensor2_Voigt_to_tensor4_NOT_Major_Sym(S_homogen_Voigt)
    A=S_4or
    n=[1,0,0]
           
    (EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
    
    alpha=tensor4_contract4_tensor4(EL,A)    
    betha=tensor4_contract4_tensor4(JT,A)
    
    gamma=tensor4_contract4_tensor4(F,A)
    gammap=tensor4_contract4_tensor4(FT,A)
    
    delta=(1./2.)*tensor4_contract4_tensor4(KT,A)
    deltap=(1./2.)*tensor4_contract4_tensor4(KL,A)

    El=1./alpha
    vt=(delta-betha)/(delta+betha)
    Et=(1.-vt)/betha
    vl=gamma*El/(-np.sqrt(2.))
    Gl=1./(2.*deltap)
    
    return El,Et,vl,vt,Gl,n



(El_MT,Et_MT,vl_MT,vt_MT,Gl_MT,n_MT)=El_Et_vl_vt_Gl_n_isotrope_transverse(C_homogen_MT)

(El_Voigt,Et_Voigt,vl_Voigt,vt_Voigt,Gl_Voigt,n_Voigt)=El_Et_vl_vt_Gl_n_isotrope(C_homogen_Voigt)

(El_Reuss,Et_Reuss,vl_Reuss,vt_Reuss,Gl_Reuss,n_Reuss)=El_Et_vl_vt_Gl_n_isotrope(C_homogen_Reuss)

print('----------------------- El, Et --------------------------')
print('El_MT=E33=',El_MT,'---','El_Voigt=E33=',El_Voigt)
print('Et_MT=E22=',Et_MT,'---','Et_Reuss=E22=',Et_Reuss)
print('-------------------------------------------------------')

print('----------------------- vl, vt -------------------------')
print('vl_MT=',vl_MT,'---','vl_Voigt=',vl_Voigt,'---','vl_Reuss=',vl_Reuss)
print('vt_MT=',vt_MT,'---','vt_Voigt=',vt_Voigt,'---','vt_Reuss=',vt_Reuss)
print('-------------------------------------------------------')


print('------------------------ Gl, Gt --------------------------')
print('Gl_MT=',Gl_MT,'---','Gl_Voigt=',Gl_Voigt,'---','Gl_Reuss=',Gl_Reuss)
print('-------------------------------------------------------')


print('Reminde: Voigt and Ruess predict isotropic')



