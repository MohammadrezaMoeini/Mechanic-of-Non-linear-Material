# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:55:46 2018

@author: momoe
"""

#In the name of GOD
#Tensors - Python Codes. (by Mohammadreza Moeini)
import numpy as np
from numpy import *
from math import * 
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv
#from Tensors_Functions_EXPANDED_05092018 import *

#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# MY FUNCTIONS 
# =============================================================================
# =============================================================================
#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# Definition of tensor
# =============================================================================
# =============================================================================
#
def operature_tansoriel_n_linear(t):
    'This function just says the concept of linear, bi-linea, ... of mathemathical operations in domain E^n'
    u=[1,1,1]; alpha=0
    if len(shape(t))==1:
        print('''Tenseur d'ordre 1: ''')
        for i in range(0,len(t)):
            alpha=alpha+t[i]*u[i]
        print('''Un tenseur d’ordre 1 permet de définir une forme linéaire sur E.''')
        print('''qui est égal à: alpha=''',alpha)


    if len(shape(t))==2:
          print('''Tenseur d'ordre 2: ''')
          for i in range(0,len(t)):
              for j in range(0,len(t[0])):
                  alpha=alpha+u[i]*t[i][j]*u[j]
          print('''Un tenseur d’ordre 2 permet de définir une forme bi-linéaire sur ExE.''')
          print('''qui est égal à: alpha=''',alpha)


    if len(shape(t))==3:
          print('''Tenseur d'ordre 3: ''')
          for i in range(0,len(t)):
              for j in range(0,len(t[0])):
                  for k in range(0,len(t[0][0])):
                      alpha=alpha+u[i]*t[i][j][k]*u[j]
          print('''Un tenseur d’ordre 3 permet de définir une forme 3-linéaire sur ExExE.''')
          print('''qui est égal à: alpha=''',alpha)

        
        
    if len(shape(t))==4:
          print('''Tenseur d'ordre 4: ''')
          for i in range(0,len(t)):
              for j in range(0,len(t[0])):
                  for k in range(0,len(t[0][0])):
                      for l in range(0,len(t[0][0][0])):
                          alpha=alpha+u[i]*u[j]*u[k]*u[l]*t[i][j][k][l]
          print('''Un tenseur d’ordre 4 permet de définir une forme 4-linéaire sur ExExExE.''')
          print('''qui est égal à: alpha=''',alpha)



# =============================================================================
# Permutation symbol epsilon_ijk (le tenseur alternateur)
# ============================================================================
def epsilon_ijk(i,j,k):
    ijk=[i,j,k]
    if i==j or j==k or i==k:
        e=0
    elif ijk==[0,1,2] or ijk==[1,2,0] or ijk==[2,0,1]:
        e=1
    else:
        e=-1
    return e


def epsilon_ijk_2order_tensor(A):
    b=np.zeros((3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                b[k]=epsilon_ijk(i,j,k)*A[i][j]+b[k]
    return b
           

# =============================================================================
# Transpose f 4-order tensor
# =============================================================================
def transpose_4orten(A):
    A_T=np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    A_T[i][j][k][l]=A[k][l][i][j]
    return A_T




def finding_non_zero_elements_4ordten(A):
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if A[i][j][k][l]!=0:
                        print ('[',i,j,k,l,']', '=!','0')
    



#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# Semain 1 
# =============================================================================
# =============================================================================
#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# Symmetry of tensor
# =============================================================================
# =============================================================================
def tensor_symmetry(t):
    if len(shape(t))==2:
#        print('------------------------------------')
#        print('Second-order tensor, Checking symetric:')
        for i in range(0,len(t)):
            for j in range(0,len(t[0])):
                if i!=j:
                    if round((t[i][j]))!= round((t[j][i])):
                        print('''It's not Symmetric!''')
                        print ('t[i][j][!=t[j][i]---->','t[',i,j,']=', t[i][j],'!=','t[',j,i,']=', t[j][i])
                        return False
                    else:
                        print('''It's Symmetric''')
                        print('------------------------------------')
                        return True
   
    if len(shape(t))==4:
#        print('------------------------------------')
#        print('Fourth-order tensor, Checking symetric:')
        for i in range(0,len(t)):
            for j in range(0,len(t[0])):
                for k in range(0,len(t[0][0])):
                    for l in range(0,len(t[0][0][0])):
                        if round(t[i][j][k][l],4)!=round(t[j][i][k][l],4):
                            print('''It's not Symmetric (minor symmetry)!''')
                            print ('t[i][j][k][l]!=t[j][i][k][l]---->','t[',i,j,k,l,']=', t[i][j][k][l],'!=','t[',j,i,k,l,']=', t[j][i][k][l])
                            return False
                        if round(t[i][j][k][l],4)!=round(t[i][j][l][k],4):
                            print('''It's not Symmetric (minor symmetry)!''')
                            print ('t[i][j][k][l]!=t[i][j][l][k]---->','t[',i,j,k,l,']=', t[i][j][k][l],'!=','t[',i,j,l,k,']=', t[i][j][l][k])
                            return False
                        if round(t[i][j][k][l],4)!=round(t[j][i][l][k],4):
                            print('''It's not Symmetric (minor symmetry)!''')
                            print ('t[i][j][k][l]!=t[j][i][l][k]---->','t[',i,j,k,l,']=', t[i][j][k][l],'!=','t[',j,i,l,k,']=', t[j][i][l][k])
                            return False
                        
                        if round(t[i][j][k][l],4)!=round(t[k][l][i][j],4):
                            print('''It's not Symmetric (*major symmetry)!''')
                            print ('t[i][j][k][l]!=t[k][l][i][j]---->','t[',i,j,k,l,']=', t[i][j][k][l],'!=','t[',k,l,i,j,']=', t[k][l][i][j])
                            return False
        else:
#            print('''It's Symmetric (minor and major)''')
#            print('*\n')                        
            return True
                        
                        
                    
            
A=np.zeros((3,3))
  
    
# =============================================================================
# =============================================================================
# Produits tensoriels et contractés
# =============================================================================
# =============================================================================
      
#Une fonction qui prend comme arguments 2 tenseurs et qui retourne le produit tensoriel.

#=============================================================================
#AxB=C         Aijkl...Bxyzw...= Cijkl...xyzw... -----> C (A-order+B-order)-tensor
#=============================================================================
def produits_tensoriels(A,B):
    'AxB=C-----> Aijkl...Bxyzw...= Cijkl...xyzw... -----> C (A-order+B-order)-tensor'
    if len(shape(A))==len(shape(B))==1:
        C=np.zeros((len(A),len(B)))
        for i in range(0,len(A)):
            for j in range(0,len(B)):
                C[i][j]=A[i]*B[j]
                
    elif len(shape(A))==2 and len(shape(B))==1:
        print('Please put 2-order and 1-order tensors respectively')
        C=np.zeros((len(A),len(A[0]),len(B)))
        for i in range(0,len(A)):
            for j in range(0,len(A[0])):
                for k in range(0,len(B)):
                    C[i][j][k]=A[i][j]*B[k]
                    
    elif len(shape(A))==len(shape(B))==2:
        C=np.zeros((len(A),len(A[0]),len(B),len(B[0])))
        for i in range(0,len(A)):
            for j in range(0,len(A[0])):
                for k in range(0,len(B)):
                    for l in range(0,len(B[0])):
                        C[i][j][k][l]=A[i][j]*B[k][l]
    return C

    
# =============================================================================
# A.B=C         Ai Bi= C -----> C scalaire
# =============================================================================
def tensor1_contract1_tensor1(A,B):
    'A.B=C         Ai Bi= C -----> C scalaire'
    temp_sum=0
    for i in range(0,len(A)):
            temp_sum=temp_sum + A[i]*B[i]
    return temp_sum

# =============================================================================
# A.B=C             Aij Bjk = Cik -----> C tenseur d’ordre 2
# =============================================================================
def tensor2_contract1_tensor2(A,B):
    'A.B=C             Aij Bjk = Cik -----> C tenseur d’ordre 2'
    size_A = np.shape(A)
    C=np.zeros(size_A)
    for i in range(0,len(A)):
        for j in range(len(A[0])):
            for k in range(len(B[0])):
                C[i][k]=C[i][k]+A[i][j]*B[j][k]
    return  C
                
# =============================================================================
# A:B=C        Aij Bji = C -----> C scalaire
# =============================================================================
def tensor2_contract2_tensor2(A,B):
    'A:B=C        Aij Bji = C -----> C scalaire'
    C=0
    for i in range(0,len(A)):
        for j in range(len(A[0])):
                C=C+A[i][j]*B[j][i]
    return  C
                                               
# =============================================================================
# A.B=C         Aij Bj =Ci -----> C tenseur d’ordre 1
# =============================================================================
def tensor2_contract1_tensor1(A,B):
    'A.B=C         Aij Bj =Ci -----> C tenseur d’ordre 1'
    C=np.zeros(len(A))
    for i in range(0,len(A)):
        for j in range(len(A[0])):
            C[i]=C[i]+A[i][j]*B[j]
    return  C    

# =============================================================================
# A:B=C         Aijkl Bklop =Cijop -----> C tenseur d’ordre 4
# =============================================================================
def tensor4_contract2_tensor4(A,B):
    'A:B=C         Aijkl Bklop =Cijop -----> C tenseur d’ordre 4'
    size_A=np.shape(A)
    C=np.zeros(size_A)
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            for k in range(0,len(A[0][0])):
                for l in range(0,len(A[0][0][0])):
                    for o in range(0,len(B[0][0])):
                        for p in range(0,len(B[0][0][0])):
                            C[i][j][o][p]=C[i][j][o][p]+A[i][j][k][l]*B[k][l][o][p]
    return  C


# =============================================================================
# A::B=C         Aijkl Bklij =C -----> C scalaire
# =============================================================================

def tensor4_contract4_tensor4(A,B):
    'A::B=C         Aijkl Bklij =C -----> C scalaire'
    C=0
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            for k in range(0,len(A[0][0])):
                for l in range(0,len(A[0][0][0])):
                    C=C+A[i][j][k][l]*B[i][j][k][l]
    return  C


# =============================================================================
# A:B=C         Aijkl Bkl =Cij -----> C tenseur d’ordre 2
# =============================================================================
def tensor4_contract2_tensor2(A,B):
    'A:B=C         Aijkl Bkl =Cij -----> C tenseur d’ordre 2'
    C=np.zeros((len(A),len(A[0])))
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            for k in range(0,len(A[0][0])):
                for l in range(0,len(A[0][0][0])):
                    C[i][j]=C[i][j]+A[i][j][k][l]*B[k][l]
    return  C


 
# =============================================================================
# =============================================================================
# Changement de base d'un tenseur 
# =============================================================================
# =============================================================================

# =============================================================================
# Pij Aj =Ci -----> C vecteur dans la nouvelle coordonnée
# =============================================================================
def Changement_de_base_tensor1(A,P):
    'Pij Aj =Ci -----> C vecteur dans la nouvelle coordonnée'
    size_A=np.shape(A)
    C=np.zeros(size_A)
    for i in range(0,3):
        for j in range(0,3):
            C[i]=C[i]+P[j][i]*A[j]
    return C

# =============================================================================
# Pik Pjl Aij =Ckl -----> Ckl dans la nouvelle coordonnée
# =============================================================================
def Changement_de_base_tensor2(A,P):
    'Pik Pjl Aij =Ckl -----> Ckl dans la nouvelle coordonnée'
    size_A=np.shape(A)
    C=np.zeros(size_A)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):                
                    C[k][l]=C[k][l]+P[i][k]*P[j][l]*A[i][j]
    return C

# =============================================================================
# Pim Pjn Pko Plp Aijkl =Cmnop -----> Cmnop dans la nouvelle coordonnée
# =============================================================================
def Changement_de_base_tensor4(A,P):
    'Pim Pjn Pko Plp Aijkl =Cmnop -----> Cmnop dans la nouvelle coordonnée'
    size_A=np.shape(A)
    C=np.zeros(size_A)
    for m in range(0,3):
        for n in range(0,3):
            for o in range(0,3):
                for p in range(0,3):
                    for i in range(0,3):
                        for j in range(0,3):
                            for k in range(0,3):
                                for l in range(0,3):                    
                                    C[m][n][o][p]=C[m][n][o][p]+P[i][m]*P[j][n]*P[k][o]*P[l][p]*A[i][j][k][l]
    return C


# =============================================================================
# =============================================================================
# Notation de Voigt modifié 
# =============================================================================
# =============================================================================

# =============================================================================
# A_3x3 tenseur d’ordre 2 ---[Voigt modifié]--->C_6x1, tenseur d’ordre 1  
# =============================================================================
def tensor2_to_Voigt_tensor1(A):
    'A_3x3 tenseur d’ordre 2 ---[Voigt modifié]--->C_6x1, tenseur d’ordre 1'
    C=np.zeros((6,1))
    C[0]=A[0][0]
    C[1]=A[1][1]
    C[2]=A[2][2]
    C[3]=A[1][2]*sqrt(2.)
    C[4]=A[2][0]*sqrt(2.)
    C[5]=A[0][1]*sqrt(2.)
    return C


# =============================================================================
# C_6x1, tenseur d’ordre 1 [Voigt modifié]-------> A_3x3 tenseur d’ordre 2 
# =============================================================================
def tensor1_Voigt_to_tensor2(A):
    'C_6x1, tenseur d’ordre 1 [Voigt modifié]-------> A_3x3 tenseur d’ordre 2'
    C=np.zeros((3,3))
    for i in range(0,3):
        C[i][i]=A[i]
    
    C[1][2]= A[3]/sqrt(2.) # C_23
    C[2][0]= A[4]/sqrt(2.) # C_31
    C[0][1]= A[5]/sqrt(2.) # C_12
    
    C[2][1]=C[1][2]
    C[0][2]=C[2][0]
    C[1][0]=C[0][1]
    return C

# =============================================================================
# A_3x3x3x3 tenseur d’ordre 4 ---[Voigt modifié]--->C_6x6, tenseur d’ordre 2  
# =============================================================================

def tensor4_to_Voigt_tensor2(A):
    'A_3x3x3x3 tenseur d’ordre 4 ---[Voigt modifié]--->C_6x6, tenseur d’ordre 2'
    condition=tensor_symmetry(A)
    if condition is True:

        C=np.zeros((6,6))
        for i in range(0,3):
            for j in range(0,3):
                C[i][j]=A[i][i][j][j]
            
            for i in range(0,3):
                C[3][i] = sqrt(2.) * A[1][2][i][i]
                C[4][i] = sqrt(2.) * A[2][0][i][i]
                C[5][i] = sqrt(2.) * A[0][1][i][i]
		
        for i in range(0,3):
            C[i][3] = sqrt(2.) * A[i][i][1][2]
            C[i][4] = sqrt(2.) * A[i][i][2][0]
            C[i][5] = sqrt(2.) * A[i][i][0][1]
		
        C[3][3] = 2. * A[1][2][1][2]
        C[4][3] = 2. * A[2][0][1][2]
        C[5][3] = 2. * A[0][1][1][2]
	
        C[3][4] = 2. * A[1][2][2][0]
        C[4][4] = 2. * A[2][0][2][0]
        C[5][4] = 2. * A[0][1][2][0]

        C[3][5] = 2. * A[1][2][0][1]
        C[4][5] = 2. * A[2][0][0][1]
        C[5][5] = 2. * A[0][1][0][1]
	        
        return C

    else:
        print('This function is just for 4-order tensor with minor and major symetic')

# =============================================================================
# A_6x6 tenseur d’ordre 2 [Voigt modifié]------> A_3x3x3x3 tenseur d’ordre 4   
# =============================================================================

def tensor2_Voigt_to_tensor4(A2):
    'A_6x6 tenseur d’ordre 2 [Voigt modifié]------> A_3x3x3x3 tenseur d’ordre 4'
    A4=np.zeros((3,3,3,3))
    for i in range(0,6):
        for j in range(0,6):
            
            if i==0:
                a=0; b=0
            if i==1:
                a=1; b=1
            if i==2:
                a=2; b=2

            if i==3:
                a=1; b=2
            if i==4:
                b=0; a=2
            if i==5:
                a=0; b=1

            if j==0:
                c=0; d=0
            if j==1:
                c=1; d=1
            if j==2:
                c=2; d=2
             
            if j==3:
                c=1; d=2
            if j==4:
                d=0; c=2
            if j==5:
                c=0; d=1

            if a==b and c==d:
                A4[a][b][c][d]=A2[i][j]
                                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                                                       
            if a!=b and c==d:
                A4[a][b][c][d]=A2[i][j]/(sqrt(2.)) 
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry                                

            if a==b and c!=d:
                A4[a][b][c][d]=A2[i][j]/(sqrt(2.))
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                

            if a!=b and c!=d:
                A4[a][b][c][d]=A2[i][j]/(2.)
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                  
      
    return A4 


#MY NOTE (19 Nov 2018 (FINAL REVISER)):
    #If you don't cosider major symmetric it's ok too. 
    #(I checked it without this line. it's ok - BOHAGH 19-11-2018)
    #in the other functions for general tensor (without major sys.)
    #I just delet the sym. condition. you can ALWAYS use that. 
    #for Eshelby tensors

#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# Semain 2
# =============================================================================
# =============================================================================
#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#SYM                       4order_tensors                                       Coeficients
#----                      ---------------                                      ------------------------
#isotropie:                I_J_K_4orten_isotropes()                             alpha_betha_4orten_isotropes(A)
#cubiques:                 J_Ka_Kb_4orten_cubiques()                            alpha_betha_gamma_4orten_cubiques(A)
#isotropes transverses:    EL_JT_F_KT_KL_4orten_isotropes_transverses(n)        alpha_betha_gamma_delta_deltap_4orten_isotrope_transverse(A)

#-------------------------------------------------------------------------------
#                          Obtention des projecteurs
#-------------------------------------------------------------------------------
# =============================================================================
# projecteurs isotropes J et K
# =============================================================================

def delta_kronecker(i,j):
    if i==j:
        return 1.0
    else:
        return 0.0
    
I_4=np.zeros((3,3,3,3))
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                I_4[i][j][k][l]=(0.5)*(delta_kronecker(i,k)*delta_kronecker(j,l) + delta_kronecker(i,l)*delta_kronecker(j,k))

def I_J_K_4orten_isotropes():
    'return(I, J, K)---Fourth-order tensors (Projecteurs isotropes)'
    I=np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    I[i][j][k][l]=(0.5)*(delta_kronecker(i,k)*delta_kronecker(j,l) + delta_kronecker(i,l)*delta_kronecker(j,k))
    i=[[1,0,0],[0,1,0],[0,0,1]]
    J=1./3.*(produits_tensoriels(i,i))
    
    K=I-J   
    return(I,J,K)

# =============================================================================
# projecteurs cubiques J,Ka et Kb.
# =============================================================================
def J_Ka_Kb_4orten_cubiques():
    'return(J, Ka, Kb)---Fourth-order tensors (Projecteurs pour la symétrie cubique)'
    (I,J,K)=I_J_K_4orten_isotropes()
    
    e1=[1.,0,0]; e2=[0,1.,0]; e3=[0,0,1.]
    e1xe1xe1xe1=produits_tensoriels(produits_tensoriels(e1,e1),produits_tensoriels(e1,e1))
    e2xe2xe2xe2=produits_tensoriels(produits_tensoriels(e2,e2),produits_tensoriels(e2,e2))
    e3xe3xe3xe3=produits_tensoriels(produits_tensoriels(e3,e3),produits_tensoriels(e3,e3))
    Z=e1xe1xe1xe1 + e2xe2xe2xe2 + e3xe3xe3xe3
    
    Ka=Z-J
    Kb=I-Z
    return(J,Ka,Kb)
    
# =============================================================================
# projecteurs isotropes transverses EL, JT, F, KT et KL
# =============================================================================    
def EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n):
    'return(EL,JT,F,FT,KT,KL)---Fourth-order tensors (projecteurs isotropes transverses)'
    nxn=produits_tensoriels(n,n)
    nxnxnxn=produits_tensoriels(nxn,nxn)
    i=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    ixi=produits_tensoriels(i,i)
    (I,J,K)=I_J_K_4orten_isotropes()
    EL=nxnxnxn
    JT=(1./2.)*(3.0*J + EL - produits_tensoriels(i,nxn)-produits_tensoriels(nxn,i))
    
    I_TV=np.zeros((6,6))
    it=i-nxn
    I_TV[0:3,0:3]=it
    I_TV[3:6,3:6]=nxn
    IT=tensor2_Voigt_to_tensor4(I_TV)
    
    KT=IT-JT
    KE=(1./6.)*(produits_tensoriels(2.*nxn - it,2.*nxn - it))
    KL=K-KT-KE
    
    F=(sqrt(2.0)/2.0)*(produits_tensoriels(it,nxn))    
    FT=transpose_4orten(F)
    
    return(EL, JT, F, FT, KT, KL)
    
#-------------------------------------------------------------------------------
#                          Obtention des coefficients
#-------------------------------------------------------------------------------
# =============================================================================
# "tenseur isotrope"  ---------> alph et betha [2 constantes indépendantes]
# =============================================================================

def alpha_betha_4orten_isotropes(A):
    '''tenseur isotrope  ---------> alph et betha [2 constantes indépendantes]'''
    (I,J,K)=I_J_K_4orten_isotropes()
    
    alpha=tensor4_contract4_tensor4(J,A)
    betha=(1./5.)*(tensor4_contract4_tensor4(K,A))
    
    return(alpha, betha)
    

def alpha_betha_1D_isotropes(A):
    '''tenseur isotrope  ---------> alph et betha [1D]'''
    J=(1./3.)*np.array([1.])
    K=(1./3.)*np.array([2.])
    
    alpha=J*A
    betha=K*A
    
    return(alpha, betha)




    
# =============================================================================
# "tenseur cubique"  ----------> alph, betha et gamma  [3 constantes indépendantes]
# =============================================================================
def alpha_betha_gamma_4orten_cubiques(A):
    '''tenseur cubique  ----------> alph, betha et gamma  [3 constantes indépendantes]:
    invariant par rotation de k.pi/2 autour des trois axes'''
    (J,Ka,Kb)=J_Ka_Kb_4orten_cubiques()
    
    alpha=tensor4_contract4_tensor4(J,A)
    betha=(1./2.)*(tensor4_contract4_tensor4(Ka,A))
    gamma=(1./3.)*(tensor4_contract4_tensor4(Kb,A))
    
    return(alpha, betha, gamma)
        
# =============================================================================
# "tenseur isotrope transverse"  ----------> alph, betha, gamma, gamma', delta, delta'  [5 constantes indépendantes]
# ============================================================================= 
def finding_axis_of_isotropy_transverse(A):
    'please just input the isotropy_transverse 4order tensor'
    A_V=tensor4_to_Voigt_tensor2(A)
    if A_V[0][0]==A_V[1][1]:
        n=[0,0,1]
           
    if A_V[0][0]==A_V[2][2]:
        n=[0,1,0]

    if A_V[1][1]==A_V[2][2]:
        n=[1,0,0]
        
#    else:
#        print('This function works just for isotropy_transverse 4order tensor' )
#    
    return(n)
    
 
    
def alpha_betha_gamma_gammap_delta_deltap_4orten_isotrope_transverse(A):
    '''tenseur isotrope transverse"  ----------> alph, betha, gamma, gamma', delta, delta'  [5 constantes indépendantes]'''
    n=finding_axis_of_isotropy_transverse(A)
    
    (EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
    
    alpha=tensor4_contract4_tensor4(EL,A)    
    betha=tensor4_contract4_tensor4(JT,A)
    
    gamma=tensor4_contract4_tensor4(F,A)
    gammap=tensor4_contract4_tensor4(FT,A)
    
    delta=(1./2.)*tensor4_contract4_tensor4(KT,A)
    deltap=(1./2.)*tensor4_contract4_tensor4(KL,A)
    
    return(alpha, betha, gamma, gammap, delta, deltap)




def C_alphac_bethac_gammac_gammapc_deltac_deltapc_4orten_isotrope_transverse(A):
    '''tenseur isotrope transverse"  ----------> alphc, bethac, gammac, gammac', deltac, deltac'  [5 constantes indépendantes]'''
    n=finding_axis_of_isotropy_transverse(A)
    
    (EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
    
    alpha=tensor4_contract4_tensor4(EL,A)    
    betha=tensor4_contract4_tensor4(JT,A)
    
    gamma=tensor4_contract4_tensor4(F,A)
    gammap=tensor4_contract4_tensor4(FT,A)
    
    delta=(1./2.)*tensor4_contract4_tensor4(KT,A)
    deltap=(1./2.)*tensor4_contract4_tensor4(KL,A)
    
    deltac=1.0/delta
    deltapc=1.0/deltap
    alphac=betha/(alpha*betha-gamma**2.)
    bethac=alpha/(alpha*betha-gamma**2.)
    gammac=-gamma/(alpha*betha-gamma**2.)
    gammapc=gammac
      
    return(alphac, bethac, gammac, gammapc, deltac, deltapc)





#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# =============================================================================
# Semain 3
# =============================================================================
# =============================================================================
#-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# =============================================================================
# matériau isotrope --->   le tenseur de souplesse et de rigidité
# =============================================================================

def isotropic_S_Voigt_and_4orten(E,v):
    'returns S_V and S (Voigt and fourth-order compliance tensor)'
    G=E/(2.*(1.+v))
    
    S_V=np.zeros((6,6))
    for i in range(0,3):
        for j in range(0,3):
            if i!=j:
                S_V[i][j]=-v/E
            if i==j:
                S_V[i][j]=1.0/E
    
    for i in range(3,6):
        for j in range(3,6):
            if i==j:
                S_V[i][j]=1.0/2.0*G
     
    S=tensor2_Voigt_to_tensor4(S_V)
    
    print('shear modulus:',G)
    return S_V,S
        

# =============================================================================
#    Note
#    S::J  = S_V:J_V    
# =============================================================================


def isotropic_materials(E,v):
    'input 2 independent constants: E,v'
    
    alpha=(1.0-2.0*v)/E;  k=1.0/(3.0*alpha)
    betha=(1.0+v)/E;      mu=1.0/(2.0*betha)
               
    (I,J,K)=I_J_K_4orten_isotropes()
    
    S=(alpha)*J + (betha)*K
    
    C=(1.0/alpha)*J + (1.0/betha)*K
    
    S_V=tensor4_to_Voigt_tensor2(S)
    C_V=tensor4_to_Voigt_tensor2(C)
    

    print('compressibility modulus  k=',k)
    print('shear modulus  mu=G=',mu)
    
    return S,S_V,C,C_V
  
    

def cubic_material(E,v,G):
    'input 3 independent constants: E,v,G'

    alpha=(1.0 - 2.0*v)/E;  k=1.0/(3.0*alpha) 
    betha=(1.0+v)/E;        mua=1.0/(2.0*betha)
    gamma=1.0/2.0*G;        mub=1.0/(2.0*betha)   
    
    (J,Ka,Kb)=J_Ka_Kb_4orten_cubiques()
    
    S=(alpha)*J + (betha)*Ka + (gamma)*Kb 
    
    C=(1.0/alpha)*J + (1.0/betha)*Ka + (1.0/gamma)*Kb

    S_V=tensor4_to_Voigt_tensor2(S)
    C_V=tensor4_to_Voigt_tensor2(C)

    print('compressibility modulus  k=',k)
    print('shear modulus  mua=',mua,'\n', '{changement de forme d’un cube à un prisme}')
    print('shear modulus  mub=',mub, '\n', '{changement des angles droits du prisme}')


def transverse_isotropic_material(El,Et,vl,vt,Gl,n):
    'input 5 independent constants: El,Et,vl,vt,Gl  and   axis of symmetry: n'
    alpha=1.0/El
    betha=(1.0-vt)/(Et)
    gamma=(-sqrt(2.0)*vl)/(El); gammap=gamma
    delta=(1.0+vt)/(Et)
    deltap=1.0/(2.0*Gl)
    
    (EL, JT, F, FT, KT, KL)=EL_JT_F_FT_KT_KL_4orten_isotropes_transverses(n)
    
    S=alpha*EL + betha*JT + gamma*(F+FT) + delta*KT + deltap*KL
       
    deltac=1.0/delta
    deltapc=1.0/deltap
    alphac=betha/(alpha*betha-gamma**2.)
    bethac=alpha/(alpha*betha-gamma**2.)
    gammac=-gamma/(alpha*betha-gamma**2.)
       
    C=alphac*EL + bethac*JT + gammac*(F+FT) + deltac*KT + deltapc*KL
    
    S_V=tensor4_to_Voigt_tensor2(S)
    C_V=tensor4_to_Voigt_tensor2(C)
    
    return S,S_V,C,C_V
  


# =============================================================================
# =============================================================================
# # My Extra functions - New: 19-11-2018
# =============================================================================
# =============================================================================

# =============================================================================
# Checking thermidynamic admissible
# =============================================================================
def checking_positive_definite(A):
    (U,d,V)=np.linalg.svd(A)
    
    for i in range(0,len(d)):
        if d[i] > 0:
            print(d[i], 'eigenvalue: ok')
            pass

        else:
            print(d[i], 'should be strictly positive!!!')
            print('le matériau n’est pas admissible thermodynamiquement')
            
            
            
# =============================================================================
# Voigt Notation for NON-Major Symmetric 4-order Tensors            
# =============================================================================
def tensor4_to_Voigt_tensor2_NOT_Major_Sym(A):
    'A_3x3x3x3 tenseur d’ordre 4 ---[Voigt modifié]--->C_6x6, tenseur d’ordre 2'
    C=np.zeros((6,6))
    for i in range(0,3):
        for j in range(0,3):
            C[i][j]=A[i][i][j][j]
            
        for i in range(0,3):
            C[3][i] = sqrt(2.) * A[1][2][i][i]
            C[4][i] = sqrt(2.) * A[2][0][i][i]
            C[5][i] = sqrt(2.) * A[0][1][i][i]
		
    for i in range(0,3):
        C[i][3] = sqrt(2.) * A[i][i][1][2]
        C[i][4] = sqrt(2.) * A[i][i][2][0]
        C[i][5] = sqrt(2.) * A[i][i][0][1]
		
    C[3][3] = 2. * A[1][2][1][2]
    C[4][3] = 2. * A[2][0][1][2]
    C[5][3] = 2. * A[0][1][1][2]
	
    C[3][4] = 2. * A[1][2][2][0]
    C[4][4] = 2. * A[2][0][2][0]
    C[5][4] = 2. * A[0][1][2][0]

    C[3][5] = 2. * A[1][2][0][1]
    C[4][5] = 2. * A[2][0][0][1]
    C[5][5] = 2. * A[0][1][0][1]
	        
    return C



def tensor2_Voigt_to_tensor4_NOT_Major_Sym(A2):
    'A_6x6 tenseur d’ordre 2 [Voigt modifié]------> A_3x3x3x3 tenseur d’ordre 4'
    A4=np.zeros((3,3,3,3))
    for i in range(0,6):
        for j in range(0,6):
            
            if i==0:
                a=0; b=0
            if i==1:
                a=1; b=1
            if i==2:
                a=2; b=2

            if i==3:
                a=1; b=2
            if i==4:
                b=0; a=2
            if i==5:
                a=0; b=1

            if j==0:
                c=0; d=0
            if j==1:
                c=1; d=1
            if j==2:
                c=2; d=2
             
            if j==3:
                c=1; d=2
            if j==4:
                d=0; c=2
            if j==5:
                c=0; d=1

            if a==b and c==d:
                A4[a][b][c][d]=A2[i][j]
                                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
#                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                                                       
            if a!=b and c==d:
                A4[a][b][c][d]=A2[i][j]/(sqrt(2.)) 
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
#                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry                                

            if a==b and c!=d:
                A4[a][b][c][d]=A2[i][j]/(sqrt(2.))
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
#                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                

            if a!=b and c!=d:
                A4[a][b][c][d]=A2[i][j]/(2.)
                
                A4[b][a][c][d]=A4[a][b][c][d] #minor symmetry
                A4[a][b][d][c]=A4[a][b][c][d] #...
                A4[b][a][d][c]=A4[a][b][c][d] #...                    
#                A4[d][c][a][b]=A4[a][b][c][d] #major symmetry
                  
      
    return A4 



# =============================================================================
# Calculation (E,v) from (k,mu) for isotropic tensors             
# =============================================================================
def E_v_from_k_mu(k,mu):
    E=(9.*k*mu) / (mu + 3.*k)
    v=(3.*k - 2*mu) / (2*mu + 6*k)   
    return E,v

def k_mu_from_alpha_betha(alpha,betha):
    k=1./(3.*alpha)
    mu=1./(2.*betha)
    return k,mu




'Calculation Engineering constants'
'Specificlly for homogenization Problems 07-02 and 07-04'

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
















