# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:07:01 2018

@author: momoe
"""
from Tensors_Functions_S1_S2_S3_21082018 import *
import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from array import *

  

# =============================================================================
# Relaxation C and Creep (Fluage) S
# =============================================================================
def Cv_relaxation_test(k,mu):
    Cv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(k)):       
        Ci= 3.*k[i]*J_V + 2.*mu[i]*K_V
        Cv.append(Ci)
        
    return Cv    


    
def Sv_fluage_test(k,mu):
    Sv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(k)):       
        Si= (k[i]/3.)*J_V + (mu[i]/2.)*K_V
        Sv.append(Si)        

    return Sv  



#----------------- 2D -----------------------------------
def Cv_relaxation_test_2D(k,mu):
    Cv=[]
    I=np.array([[1,0,0],[0,1,0],[0,0,1]])
    J=(1./3.)*np.array([[1,1,0],[1,1,0],[0,0,0]])
    K=I-J
    for i in range(0,len(k)):       
        Ci= 3.*k[i]*J + 2.*mu[i]*K
        Cv.append(Ci)
        
    return Cv   


def Sv_fluage_test_2D(k,mu):
    Sv=[]
    I=np.array([[1,0,0],[0,1,0],[0,0,1]])
    J=(1./3.)*np.array([[1,1,0],[1,1,0],[0,0,0]])
    K=I-J
    for i in range(0,len(k)):       
        Si= (k[i]/3.)*J + (mu[i]/2.)*K
        Sv.append(Si) 
        
    return Sv 

# =============================================================================
# Relaxation B_L1_L2_L3(C,w)
# =============================================================================

def B_L1_L2_L3(C,w):
    'Returns B_ND,ND, L1_DxD, L2_DxND, L3_DNxDN'
    sizeC=shape(C)
    N=len(w); D=sizeC[1]
    
    B=np.identity(N*D)

    L1=np.zeros((D,D))
    for i in range(0,N+1):
        L1=C[i]+L1

    L2_matrix=np.zeros((D,D*N))
    for i in range(0,N):
        L2i = scipy.linalg.cholesky(w[i]*C[i+1], lower=True)
        L2_matrix[0:D,i*D:(i+1)*D]=L2i
       
    L3=np.zeros((N*D,N*D))
    for i in range(0,N):
        for j in range(i*D,(i+1)*D):
            L3[j][j]=w[i]
    if D==1:
        print('Relaxation test: 1-Dimention ---> B,L1,L2,L3')
    if D==3:
        print('Relaxation test: 2-Dimentions ---> B,L1,L2,L3')
    if D==6:
        print('Relaxation test: 3-Dimentions ---> B,L1,L2,L3')

            
    return B,L1,L2_matrix,L3,N,D
    
    

# =============================================================================
# Creep B_L1_L2_L3(C,w)
# =============================================================================

def B_A1_A2_A3(S,landa):
    'Returns B_ND,ND, L1_DxD, L2_DxND, L3_DNxDN'
    sizeS=shape(S)
    N=len(landa); D=sizeS[1]
    
    B=np.identity(N*D)
    
    A1=S[0] #Pay attention to A1

    A2_matrix=np.zeros((D,D*N))
    for i in range(0,N):
        A2i = scipy.linalg.cholesky(landa[i]*S[i+1], lower=True)
        A2_matrix[0:D,i*D:(i+1)*D]=A2i
       
    A3=np.zeros((N*D,N*D))
    for i in range(0,N):
        for j in range(i*D,(i+1)*D):
            A3[j][j]=landa[i]        
                    
    if D==1:
        print('Creep test: 1-Dimention ---> B,A1,A2,A3')
    if D==3:
        print('Creep test: 2-Dimentions ---> B,A1,A2,A3')
    if D==6:
        print('Creep test: 3-Dimentions ---> B,A1,A2,A3')

            
    return B,A1,A2_matrix,A3,N,D


# =============================================================================
# import_time_Dcolumns
# =============================================================================
def import_time_Dcolumns(excel_file,D):
    'Please wrtie excel_file(str) and D=Dimensions'
    'This function returns strain or excel file as D x time_frames matrix' 
    if D==6:
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
        c0,c1,c2,c3,c4,c5,c6,c7,c8=df
        column_0 = df[c0]
        column_1 = df[c1]
        column_2 = df[c2]
        column_3 = df[c3]
        column_4 = df[c4]
        column_5 = df[c5]
        column_6 = df[c6]
       
        t=  np.zeros((len(column_0),1))  
        total_se=np.zeros((D,len(column_0))) #includes stress/strain 6x1 vector IN each time step 
        for i in range(0,len(column_0)):
            t[i]=column_0[i]
            total_se[0][i]=column_1[i]
            total_se[1][i]=column_2[i]
            total_se[2][i]=column_3[i]
            total_se[3][i]=column_4[i]
            total_se[4][i]=column_5[i]
            total_se[5][i]=column_6[i]
            
            
    if D==3:
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
        c0,c1,c2,c3=df
        column_0 = df[c0]
        column_1 = df[c1]
        column_2 = df[c2]
        column_3 = df[c3]
        
        t=  np.zeros((len(column_0),1))  
        total_se=np.zeros((D,len(column_0))) #includes stress/strain 6x1 vector IN each time step 
        for i in range(0,len(column_0)):
            t[i]=column_0[i]
            total_se[0][i]=column_1[i]
            total_se[1][i]=column_2[i]
            total_se[2][i]=column_3[i]
               
    if D==2:
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
        c0,c1,c2,c3=df
        column_0 = df[c0]
        column_1 = df[c1]
        
        t=  np.zeros((len(column_0),1))  
        total_se=np.zeros((D,len(column_0))) #includes stress/strain 6x1 vector IN each time step 
        for i in range(0,len(column_0)):
            t[i]=column_0[i]
            total_se[0][i]=column_1[i]
 
    return t,total_se


# =============================================================================
# #************************* RELAXATION **************************************
# =============================================================================

# =============================================================================
# # Relaxation_Euler_implicite
# =============================================================================
    
def relaxation_Euler_implicite(excel_file,k,mu,w,D):
    
    #  Calculate Relaxation modulus--------------------------
    C=Cv_relaxation_test(k,mu)    
   
    #  B,L1,L2,L3--------------------------
    (B,L1,L2_matrix,L3,N,D)=B_L1_L2_L3(C,w)
    
    #  Read data--------------------------
    (t,epsilon_time)=import_time_Dcolumns(excel_file,D) 
                   
    #Variables---------------------------
    n=len(t)-1
    kesi=np.zeros((N*D,n+1))
    sigma_time=np.zeros((D,n+1)) #numpy.ndarray - (D x frame times)
    h=float(t[-1])/float(n)
    
    L2T_matrix=transpose(L2_matrix)
  
    #  W1, W2--------------------------    
    W1=inv(B + h*np.matmul(B,L3))
    W2=-h*np.matmul( inv(B + h*np.matmul(B,L3)), np.matmul(inv(B),L2T_matrix) )
    
    #  Initial solution--------------------------     
    kesi[:,0]= np.matmul(W2,epsilon_time[:,0])    
    sigma_time[:,0]= np.matmul(L1,epsilon_time[:,0])  + np.matmul(L2_matrix,kesi[:,0])
    
    #  Continue... --------------------------
    for i in range(1,n+1):     
#*****************
        kesi[:,i]=  np.matmul(W1,kesi[:,i-1]) + np.matmul(W2,epsilon_time[:,i])        
        sigma_time[:,i]= np.matmul(L1,epsilon_time[:,i])  + np.matmul(L2_matrix,kesi[:,i])               
#*****************        
    return t, sigma_time

            
# =============================================================================
# # Relaxation_Crank_Nicholson
# =============================================================================
    
def relaxation_Crank_Nicholson(excel_file,k,mu,w,D):
    
    #  Calculate Relaxation modulus--------------------------
    C=Cv_relaxation_test(k,mu)    
   
    #  B,L1,L2,L3--------------------------
    (B,L1,L2_matrix,L3,N,D)=B_L1_L2_L3(C,w)
    
    #  Read data--------------------------
    (t,epsilon_time)=import_time_Dcolumns(excel_file,D) 
                   
    #Variables---------------------------
    n=len(t)-1
    kesi=np.zeros((N*D,n+1))
    sigma_time=np.zeros((D,n+1)) #numpy.ndarray - (D x frame times)
    h=float(t[-1])/float(n)
    
    L2T_matrix=transpose(L2_matrix)
  
    #  W1, W2--------------------------    
    W1=np.matmul( inv(B + (h/2.)*np.matmul(inv(B),L3)) , B - (h/2.)*np.matmul(inv(B),L3) )
    
    W2=np.matmul( inv(B + (h/2.)*np.matmul(inv(B),L3)), - (h/2.)*np.matmul(inv(B),L2T_matrix) )
    
    #  Initial solution--------------------------
    kesi[:,0]=np.matmul(W2,0*epsilon_time[:,0]+epsilon_time[:,1])
    sigma_time[:,0]= np.matmul(L1,epsilon_time[:,0]) + np.matmul(L2_matrix,kesi[:,0])     

    #  Continue... --------------------------
    for i in range(1,n+1):     
#*****************
        kesi[:,i]=  kesi[:,i]=np.matmul(W1,kesi[:,i-1]) + np.matmul(W2,epsilon_time[:,i-1]+epsilon_time[:,i])       
        sigma_time[:,i]= np.matmul(L1,epsilon_time[:,i]) + np.matmul(L2_matrix,kesi[:,i])               
#*****************               
        
    return t, sigma_time


# =============================================================================
# #************************* CREEP (fluage) ************************************
# =============================================================================



# =============================================================================
# # CREEP (Fluage)_Euler_implicite
# =============================================================================
    
def fluage_Euler_implicite(excel_file,k,mu,landa,D):
    
    #  Calculate Creep Compliance--------------------------
    S=Sv_fluage_test(k,mu)    
   
    #  B,A1,A2,A3--------------------------
    (B,A1,A2_matrix,A3,N,D)=B_A1_A2_A3(S,landa)
    
    #  Read data--------------------------
    (t,stress_time)=import_time_Dcolumns(excel_file,D) 
                   
    #Variables---------------------------
    n=len(t)-1
    kesi=np.zeros((N*D,n+1))
    epsilon_time=np.zeros((D,n+1)) #numpy.ndarray - (D x frame times)
    h=float(t[-1])/float(n)
    
    A2T_matrix=transpose(A2_matrix)
  
    #  W1, W2--------------------------    
    W1=inv(B + h*np.matmul(B,A3))
    W2=-h*np.matmul( inv(B + h*np.matmul(B,A3)), np.matmul(inv(B),A2T_matrix) )
    
    #  Initial solution--------------------------     
    kesi[:,0]= np.matmul(W2,stress_time[:,0])    
    epsilon_time[:,0]= np.matmul(A1,stress_time[:,0])  - np.matmul(A2_matrix,kesi[:,0])
    
    #  Continue... --------------------------
    for i in range(1,n+1):     
#*****************
        kesi[:,i]=  np.matmul(W1,kesi[:,i-1]) + np.matmul(W2,stress_time[:,i])        
        epsilon_time[:,i]= np.matmul(A1,stress_time[:,i])  - np.matmul(A2_matrix,kesi[:,i])               
#*****************        
    return t, epsilon_time

            



# =============================================================================
# # Relaxation_Crank_Nicholson
# =============================================================================
    
def fluage_Crank_Nicholson(excel_file,k,mu,landa,D):
    
    #  Calculate Creep Compliance--------------------------
    S=Sv_fluage_test(k,mu)     
   
    #  B,A1,A2,A3--------------------------
    (B,A1,A2_matrix,A3,N,D)=B_A1_A2_A3(S,landa)
    
    #  Read data--------------------------
    (t,stress_time)=import_time_Dcolumns(excel_file,D)
                   
    #Variables---------------------------
    n=len(t)-1
    kesi=np.zeros((N*D,n+1))
    epsilon_time=np.zeros((D,n+1)) #numpy.ndarray - (D x frame times)
    h=float(t[-1])/float(n)
    
    A2T_matrix=transpose(A2_matrix)
  
    #  W1, W2--------------------------    
    W1=np.matmul( inv(B + (h/2.)*np.matmul(inv(B),A3)) , B - (h/2.)*np.matmul(inv(B),A3) )
    
    W2=np.matmul( inv(B + (h/2.)*np.matmul(inv(B),A3)), - (h/2.)*np.matmul(inv(B),A2T_matrix) )
    
    #  Initial solution--------------------------
    kesi[:,0]=np.matmul(W2,stress_time[:,0]+stress_time[:,1])
    epsilon_time[:,0]= np.matmul(A1,stress_time[:,0]) - np.matmul(A2_matrix,kesi[:,0])     

    #  Continue... --------------------------
    for i in range(1,n+1):     
#*****************
        kesi[:,i]=  kesi[:,i]=np.matmul(W1,kesi[:,i-1]) + np.matmul(W2,stress_time[:,i-1]+stress_time[:,i])       
        epsilon_time[:,i]= np.matmul(A1,stress_time[:,i]) - np.matmul(A2_matrix,kesi[:,i])               
#*****************               
        
    return t, epsilon_time





        

