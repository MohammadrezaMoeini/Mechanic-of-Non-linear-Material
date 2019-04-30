# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:05:03 2018

@author: momoe
"""

from Tensors_Functions_S1_S2_S3_21082018 import *

# =============================================================================
# Problème 01 – 02
# =============================================================================
print('===========================================================')
print('Problème 01 – 02')
print('===========================================================')
A_V=[[0.241, -0.202, -0.347, -0.848, 0.959, -0.023],
     [-0.202, -0.678, 0.520, -0.600, 0.186, 0.861],
     [-0.347, 0.520, 0.200, -0.473, -0.598, -0.927],
     [-0.848, -0.600, -0.473, -0.214, -0.338, 0.201],
     [0.959, 0.186, -0.598, -0.338, 0.276, -0.392],
     [-0.023, 0.861, -0.927, 0.201, -0.392, -0.719]]


A=tensor2_Voigt_to_tensor4(A_V)
#print('cheching symetric of 4-order tensor of A_V:')
#a=tensor_symmetry(A)

print('P: transformation 2-order tensor')
P=[[0.654, -0.540, -0.529],
   [-0.082, -0.746, 0.661],
   [-0.752, -0.389, -0.532]]
print(P)
print('------------------------------------')
#
print('cheching symetric of new 4-order tensor anfter changing the coordinate:')
A_new=Changement_de_base_tensor4(A,P)
#tensor_symmetry(A_new)

print('Answer=')
A_V_new=tensor4_to_Voigt_tensor2(A_new)
print(A_V_new)


# =============================================================================
# Problème 01 – 04
# =============================================================================
print('\n*')
print('===========================================================')
print('Problème 01 – 04')
print('===========================================================')
ee=0
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            ee= epsilon_ijk(i,j,k) * epsilon_ijk(i,j,k) + ee

print('epsilon_ijk * epsilon_ijk',' = ', ee)


ee=0
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for m in range(0,3):
                ee= epsilon_ijk(i,j,k) * epsilon_ijk(m,j,k) + ee


print('epsilon_ijk * epsilon_ijm',' = ', ee)

# =============================================================================
# Problème 01 – 05
# =============================================================================
print('\n*')
print('===========================================================')
print('Problème 01 – 05')
print('===========================================================')

def epsilon_ijk_2order_tensor(A):
    b=np.zeros((3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                b[k]=epsilon_ijk(i,j,k)*A[i][j]+b[k]
    return b
                
                
A=[[1,2,3],[2,5,4],[3,4,6]]
print('example second-order tensor',' = ', A)
tensor_symmetry(A)
b=epsilon_ijk_2order_tensor(A)               
print('epsilon_ijk * A_ij',' = ',b)                



# =============================================================================
# Problème 01 – 14
# =============================================================================
print('\n*')
print('===========================================================')
print('Problème 01 – 14')
print('===========================================================')

n=[0,0,1]
(EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
(I,J,K)=I_J_K_4orten_isotropes()


EL_V=tensor4_to_Voigt_tensor2(EL)
J_V=tensor4_to_Voigt_tensor2(J)
K_V=tensor4_to_Voigt_tensor2(K)


JT_V=tensor4_to_Voigt_tensor2(JT)

F_V=tensor4_to_Voigt_tensor2(F) #F is not sym.

KT_V=tensor4_to_Voigt_tensor2(KT)

KL_V=tensor4_to_Voigt_tensor2(KL)

#print(EL_V)
#print(J_V)   
#print(K_V) 
#print(JT_V) 
#print(FT_V) 

ELJ=tensor2_contract2_tensor2(EL_V,J_V)
ELK=tensor2_contract2_tensor2(EL_V,K_V)

JTJ=tensor2_contract2_tensor2(JT_V,J_V)
JTK=tensor2_contract2_tensor2(JT_V,K_V)

FTJ=tensor4_contract4_tensor4(FT,J)
FTK=tensor4_contract4_tensor4(FT,K)

FJ=tensor4_contract4_tensor4(F,J)
FK=tensor4_contract4_tensor4(F,K)

KTJ=tensor2_contract2_tensor2(KT_V,J_V)
KTK=tensor2_contract2_tensor2(KT_V,K_V)


KLJ=tensor2_contract2_tensor2(KL_V,J_V)
KLK=tensor2_contract2_tensor2(KL_V,K_V)


#print('ELJ=',ELJ)
#print('ELK=',ELK)

#print('JTJ=',JTJ)
#print('JTK=',JTK)
#
#print('FTJ=',FTJ)
#print('FTK=',FTK)
#
#print('FJ=',FJ)
#print('FK=',FK)
##gamma = gammap
#print('KTJ=',KTJ)
#print('KTK=',KTK)
#
#print('KLJ=',KLJ)
#print('KLK=',KLK)

# =============================================================================
# Problème 01 – 12
# =============================================================================
print('\n*')
print('===========================================================')
print('Problème 01 – 12')
print('===========================================================')

El=180.0e9; Et=10.0e9; vl=0.28; vt=0.4; Gl=7.0e9; n=[1.0,0.0,0.0]
(S,S_V,C,C_V)=transverse_isotropic_material(El,Et,vl,vt,Gl,n)

theta=pi*37.0/180.0
P=[[cos(theta), -sin(theta), 0.],
   [sin(theta), cos(theta), 0.],
   [0., 0., 1.]]
S_37=Changement_de_base_tensor4(S,P) #S in direction 37 fiber

sigma=np.zeros((3,3))
sigma[0][0]=100.0e6 


print('sigma') 
print(sigma)

epsilon=tensor4_contract2_tensor2(S_37,sigma) 

print('\n')
print('epsilon') 
print(epsilon)

# Pay attention that in voigt notation the shear strain (or stress) is not the same

epsilon_V=tensor2_to_Voigt_tensor1(epsilon)

print('\n')
print('epsilon_Voigt notation=')

print(epsilon_V)
    

C_37=Changement_de_base_tensor4(C,P)
sigma_check=tensor4_contract2_tensor2(C_37,epsilon) 
print('\n')
print('sigma_check: WITH s=C:e')
print(sigma_check)


print('\n*')
print('===========================================================')
print('Problème 01 – 13')
print('===========================================================')

SV=[[5.56e-6,-1.56e-6,-1.56e-6,0,0,0],[-1.56e-6,100e-6,-40e-6,0,0,0],[-1.56e-6,-40e-6,100e-6,0,0,0],[0,0,0,140e-6,0,0],[0,0,0,0,71e-6,0],[0,0,0,0,0,71.4e-6]]
print(SV)
S=tensor2_Voigt_to_tensor4(SV)

(alphac, bethac, gammac, gammapc, deltac, deltapc)=C_alphac_bethac_gammac_gammapc_deltac_deltapc_4orten_isotrope_transverse(S)

n=finding_axis_of_isotropy_transverse(S)
print('n=',n)

print('alphac=', alphac/1000.,'x 1000')
print('bethac', bethac/1000.,'x 1000')
print('gammac', gammac/1000.,'x 1000')
print('gammapc', gammapc/1000.,'x 1000')
print('deltac', deltac/1000.,'x 1000')
print('deltapc', deltapc/1000.,'x 1000')




