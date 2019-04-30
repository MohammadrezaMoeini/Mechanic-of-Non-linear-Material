# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:41:08 2018

@author: momoe
"""

#okkkkk

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


(J,Ka,Kb)=J_Ka_Kb_4orten_cubiques()

A=10*J + 4*Ka + 7* Kb


A_V=tensor4_to_Voigt_tensor2(A)

theta=pi*45.0/180.0
P=[[cos(theta), -sin(theta), 0.],
   [sin(theta), cos(theta), 0.],
   [0., 0., 1.]]


A_NEW=Changement_de_base_tensor4(A,P)

A_V_new=tensor4_to_Voigt_tensor2(A_NEW)

