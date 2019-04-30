# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:07:18 2018

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


from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Function_S6_Euler_NC_Rel_Flu_14102018 import *
from Viscoelastic_Linear_Function_S5_01_09102018 import *

print('Prenom: Mohammad Reza')
print('Nom: Moeini')
print('1922545')



print('===========================================================')
print('Exercice 1')
print('===========================================================')


print('''Note:  my "Voigt function" checks the symmetry of input tensor.
      Consequently, every time I use, the symmetry condition would be checked
      and the answer would appear in the console.''' )


# =============================================================================
# changement de base d’un tenseur d’ordre 4
# =============================================================================

# 1- Write your tensour 2-order (voigt notation) as A LIST:

print('2-order tensor ((voigt notation)')
#A_V=[[*, *, *, *, *, *],
#     [*, *, *, *, *, *],
#     [*, *, *, *, *, *],
#     [*, *, *, *, *, *],
#     [*, *, *, *, *, *],
#     [*, *, *, *, *, *]]
#
print('A_V=',A_V)
A=tensor2_Voigt_to_tensor4(A_V) #convert to 4-order tensor

print('P: transformation 2-order tensor')
#********      Pay attention to columns **********
#********      Pay attention to columns **********

#P=[[x'1, y'1, z'1],
#   [x'2, y'2, z'2],
#   [x'3, y'3, z'3]]
#********      Pay attention to columns **********
#********      Pay attention to columns **********
#BE OMIDE HAGH
print('P=',P)
print('------------------------------------')

print('cheching symetric of new 4-order tensor anfter changing the coordinate:')
A_new=Changement_de_base_tensor4(A,P)
#tensor_symmetry(A_new)

print('Answer=')
A_V_new=tensor4_to_Voigt_tensor2(A_new)
print(A_V_new)

#Note: Input and output are "list". use A_V[i][j]
#      and if you want to multiply by something, first convert to array
#      A_Varr=np.array(A_V)

C=Changement_de_base_tensor1(A,P)
C=Changement_de_base_tensor2(A,P)
C=Changement_de_base_tensor4(A,P)


# =============================================================================
# =============================================================================
# # Optimization viscoelastic linear
# =============================================================================
# =============================================================================
  
# =============================================================================
# Input data 
# =============================================================================
def landa_calculator(q,n):
    'n=number of items, q=number of pices that you want to devide the decade'
    landa=[]
    j=0    
    for i in range(0,n):
        x=1./10.**(j)
        landa.append(x)
        j=j+1/q
        
    return landa  
#-------------------------------------CONTROL LANDA
landa=landa_calculator(2.,11)
print('landa=',landa)

sigma=20.
print('sigma=',sigma)

excel_file='04_02_donnees.xls'

(alpha,betha)=alphai_bethai_for_Heviside_stress(excel_file,sigma,landa)



def Sv_fluage_test_alpha_betha(alpha,betha):
    Sv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(alpha)):       
        Si= (alpha[i])*J_V + (betha[i])*K_V
        Sv.append(Si)        

    return Sv  

S=Sv_fluage_test_alpha_betha(alpha,betha)


# =============================================================================
# #************************* RELAXATION **************************************
# =============================================================================
(t,sigma_euler)=relaxation_Euler_implicite(excel_file,k,mu,w,D) 
(t,sigma_crank_nicholson)=relaxation_Crank_Nicholson(excel_file,k,mu,w,D)


# =============================================================================
# #************************* FLUAGE **************************************
# =============================================================================
(t, epsilon_euler)=fluage_Euler_implicite(excel_file,k,mu,landa,D)
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k,mu,landa,D)




#Note: first calculate the variables carefully thentry to draw graph
# =============================================================================
# #************************* sample figure **************************************
# =============================================================================
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_1000, sigma_1000[0,:], 'r-')
plt.plot(t[0:len(t)-1], sigma_euler[0,:], 'bx')
plt.plot(t[0:len(t)-1], sigma_crank_nicholson[0,:], 'g.')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('stress [GPa]')
plt.title('Relaxation')
plt.legend(('Analytical', 'Euler implicite n=%d'%nt, 'Crank_Nicholson n=%d'%nt),
           shadow=True, loc=(0.4, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 100, 0.1, 0.3])
plt.show()


  
    
    