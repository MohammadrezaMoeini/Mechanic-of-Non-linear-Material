# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:36:18 2018

@author: momoe
"""
from Tensors_Functions_S1_S2_S3_21082018 import *

#A[0][2]=1
#print(A)
#tensor_symmetry(A)                   

#a=tensor_symmetry(A)                   
#print(a)    
#
#if a==False:
#    print('not ok')
#      
   
#b=np.zeros((3,3,3,3))
#b[0][:][2][1]=1
##b[:][:][2][1]=1
#
#print(b)
#bb=tensor_symmetry(b)                   
#print(bb)      

#print('tensor2_contract1_tensor2',C1)
#print('\n')
#print('tensor2_contract2_tensor2',C2)    

#print('tensor4_contract2_tensor4',C3)

#print('tensor4_contract4_tensor4 = ',C4)

#print('tensor4_contract2_tensor4',C5)

#print('Changement_de_base_tensor1 =',C)


#print('Changement_de_base_tensor2 = \n',C)

#print('Changement_de_base_tensor4 = \n',C)


#print(A)
#print(A_V)
#print(An)

#
## -----------------------------------------------------------------------------
## Examples
## -----------------------------------------------------------------------------
#P=[[0.654,-0.082,-0.752],[-0.54,-0.746,-0.389],[-0.529,0.661,-0.532]];
#A=[1,1,1]
#C=Changement_de_base_tensor1(A,P)
#A=[[1,1,1],[1,1,1],[1,1,1]]
#C=Changement_de_base_tensor2(A,P)
#            
#A=np.zeros((3,3,3,3)); A[:][:][:][:]=1
#C=Changement_de_base_tensor4(A,P)
#
#
#
#
## -----------------------------------------------------------------------------
## Examples
## -----------------------------------------------------------------------------
#A=[[1,2,3],[4,5,6],[7,8,9]]
#B=[[2,2,2],[1,1,1],[1,1,1]]
#C1=tensor2_contract1_tensor2(A,B)
#C2=tensor2_contract2_tensor2(A,B)
#
#A=np.zeros((3,3,3,3)); A[0][0][0][:]=1 
#B=np.zeros((3,3,3,3)); B[0][:][:][:]=1
#C3=tensor4_contract2_tensor4(A,B)
#
#C4=tensor4_contract4_tensor4(A,B)
#
#A=np.zeros((3,3,3,3)); A[0][0][0][:]=1
#B=[[1,2,3],[4,5,6],[7,8,9]]
#C5=tensor4_contract2_tensor2(A,B)
#
#
#
##print('I_4',I_4)
##tensor_symmetry(I_4)
##I_V=tensor4_to_Voigt_tensor2(I_4) 
##print(I_V)
#i=[[1,0,0],[0,1,0],[0,0,1]]
#J=1/3*(produits_tensoriels(i,i))
##J_V=tensor4_to_Voigt_tensor2(J) 
##print(J_V)
#K=I_4-J 
##K_V=tensor4_to_Voigt_tensor2(K)  
##print(K_V)    
#(I,J,K)=I_J_K_4order_tensors()
##print(tensor4_contract2_tensor4(J,K))
##print(tensor4_contract2_tensor4(J,J))
##print(tensor4_contract2_tensor4(K,K))
#
##n=[0,0,1]           
##(EL, JT, F, KT, KL)=EL_JT_F_KT_KL_4order_tensors(n)
#
#
#(a,b,c)=alpha_betha_gamma_cubic_4order_tensors(A)
##print(round(a),round(b),round(c))
#

A=np.zeros((3,3,3,3)); A[:][:][:][:]=1
B=np.zeros((3,3,3,3)); B[:][:][:][:]=2


def tensor4_contract2_tensor4XXX(A,B):
    'A:B=C         Aijkl Bklop =Cijop -----> C tenseur d’ordre 4'
    size_A=np.shape(A)
    C=np.zeros(size_A)
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            for k in range(0,len(A[0][0])):
                for l in range(0,len(A[0][0][0])):
                    for o in range(0,len(B[0][0])):
                        for p in range(0,len(B[0][0][0])):
                            C[i][j][o][p]=C[i][j][o][p]+A[i][j][k][l]*B[o][p][k][l]
    return  C

print(tensor4_contract2_tensor4XXX(A,B))
print('\n--------------')
print(tensor4_contract2_tensor4(A,B))






n=[0,0,1]
(EL, JT, F, KT, KL)=EL_JT_F_KT_KL_4orten_isotropes_transverses(n)
print(EL)
i=[[1,0,0],[0,1,0],[0,0,1]]
nxn=produits_tensoriels(n,n)
print('1=',produits_tensoriels(i,nxn))
print('2=',produits_tensoriels(nxn,i))

print('1+2=',produits_tensoriels(nxn,i)+produits_tensoriels(i,nxn))

X=produits_tensoriels(i,nxn) + produits_tensoriels(nxn,i)
print('X=',X)
x1=produits_tensoriels(i,nxn)
x2=produits_tensoriels(nxn,i)

tensor_symmetry(X)
X_V=tensor4_to_Voigt_tensor2(X)
x1v=tensor4_to_Voigt_tensor2(x1)
x2v=tensor4_to_Voigt_tensor2(x2)

print('X_V=',X_V )
print('x1v',x1v )
print('x2v',x2v )


finding_non_zero_elements_4ordten(X)
print('--------')
finding_non_zero_elements_4ordten(x1)
print('--------')
finding_non_zero_elements_4ordten(x2)









