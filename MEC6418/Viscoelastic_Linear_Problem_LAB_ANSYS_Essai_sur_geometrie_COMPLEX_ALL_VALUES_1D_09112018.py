# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 22:59:51 2018

@author: momoe
"""



FINISH
/CLEAR
/COM,  Structural
!
/FILNAME,geom_complex,1
!
! Ce programme permet de calculer les deformations pour l'eprouvette de geometrie complexe testee dans le cadre du cours
! MEC6418
!
! Ce fichier prend en entree les proprietes du materiau et donne en sortie les deformations dans le plan au centre des jauges.
! La jauge 1 se trouve en haut à droite du trou et fait un angle de 45 degres par rapport à y.
! => 22 EPELX  22 EPELY  22 EPELXY

! La jauge 2 se trouve en bas, au centre de l'eprouvette et est alignee selon y.
! => 5 EPELX  5 EPELY  5 EPELXY
!
!
FINISH
/Prep7
ET,1,shell281
MPTEMP,1,0
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Proprietes du materiau 																						 !
! Partie elastique																								 !
!																												 !
MPDATA,EX,1,,4130.919044305096, !!!! Entrer la valeur de E0															 !
MPDATA,PRXY,1,,0.3959705815395929, !!!! Entrer la valeur de nu0													     !
!   																											 !
!                                                                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!													     !
! Module de relaxation en compressibilite : alpha_k_ansys														 !
!																												 !
TB,PRONY,1,1,8,BULK     !!!! Ne pas oublier d'entrer le nombre de paires				                		 !
TBTEMP,0																										 !


TBDATA,, 1.0802036367050221e-10 , 0.9999968826907845 ,
TBDATA,, 3.119391674838021e-10 , 3.162277660114432 ,
TBDATA,, 0.12073488154602724 , 9.99999999951225 ,
TBDATA,, 0.23567976369496196 , 31.028800330069608 ,
TBDATA,, 1.2610499618651727e-06 , 96.16300524646329 ,
TBDATA,, 1.0809507392988596e-08 , 316.2276979794547 ,
TBDATA,, 1.821809539504018e-11 , 999.9999982002053 ,
TBDATA,, 0.6435643915321558 , 3162.277660159233 ,
                                                                                                               !
!                                                                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!														 !
! Module de relaxation en cisaillement	: alpha_mu_ansys														 !
!																												 !
TB,PRONY,1,1,8,SHEAR    !!!! Ne pas oublier d'entrer le nombre de paires					                	 !
TBTEMP,0																										 !																										 !

TBDATA,, 8.451756405175003e-08 , 0.9999968826907845 ,
TBDATA,, 1.259386559446016e-07 , 3.162277660114432 ,
TBDATA,, 0.26330600333220394 , 9.99999999951225 ,
TBDATA,, 6.71797158719467e-05 , 31.028800330069608 ,
TBDATA,, 0.1681224926174577 , 96.16300524646329 ,
TBDATA,, 3.872416741414733e-13 , 316.2276979794547 ,
TBDATA,, 0.15660679255946372 , 999.9999982002053 ,
TBDATA,, 0.07512571922804222 , 3162.277660159233 ,


                                                                                                             !
!																												 !
!fin des donnees pour les proprietes																			 !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!
!
SECTYPE,1,shell
secdata, 4.49,1,0,5
BLC4, -(71.12)/2, -127/2, 71.12, 127,


CYL4, 41.275, 61.711, 31.75,, 31.75, ,
CYL4, -41.275, 61.711, 31.75,, 31.75, ,
CYL4, 41.275, -61.711, 31.75,, 31.75, ,
CYL4, -41.275, -61.711, 31.75,, 31.75, ,
ASBA,1,all,,,


BLC4, -(71.12)/2, -203.2/2, 26.035, 39.889,
BLC4, -(71.12)/2, 203.2/2-39.889, 26.035, 39.889,
BLC4, (71.12)/2-26.035, 203.2/2-39.889, 26.035, 39.889,
BLC4, (71.12)/2-26.035, -203.2/2, 26.035, 39.889,
ASBA,6,all,,,


CYL4, 0, 0, 20.32,, 20.32, ,

k,101, 0, -203.2/2+69.92-2.5,0
k,102, 0, -203.2/2+70.59+0.5,0
k,103, 3, -203.2/2+70.59+0.5,0
k,104, 3, -203.2/2+70.59-2.5,0
k,105, 3, -203.2/2+70.59-5.5,0
k,106, 0, -203.2/2+70.59-5.5,0
k,107, -3, -203.2/2+70.59-5.5,0
k,108, -3, -203.2/2+70.59-2.5,0
k,109, -3, -203.2/2+70.59+0.5,0
A,101,102,103,104
A,101,104,105,106
A,101,106,107,108
A,101,108,109,102
k,111, (71.12/2-18.67)-2, (203.2/2-74.35)-3+1,0
k,112, (71.12/2-18.67)-2, (203.2/2-74.35)-1.5+1,0
k,113, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-1.5+1,0
k,114, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-3+1,0
k,115, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-6+1.5+1,0
k,116, (71.12/2-18.67)-2, (203.2/2-74.35)-6+1.5+1,0
k,117, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-6+1.5+1,0
k,118, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-3+1,0
k,119, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-1.5+1,0
A,111,112,113,114
A,111,114,115,116
A,111,116,117,118
A,111,118,119,112

ASBA,5,all,,,

k,101, 0, -203.2/2+70.59-2.5,0
k,102, 0, -203.2/2+70.59+0.5,0
k,103, 3, -203.2/2+70.59+0.5,0
k,104, 3, -203.2/2+70.59-2.5,0
k,105, 3, -203.2/2+70.59-5.5,0
k,106, 0, -203.2/2+70.59-5.5,0
k,107, -3, -203.2/2+70.59-5.5,0
k,108, -3, -203.2/2+70.59-2.5,0
k,109, -3, -203.2/2+70.59+0.5,0
k,111, (71.12/2-18.67)-2, (203.2/2-74.35)-3+1,0
k,112, (71.12/2-18.67)-2, (203.2/2-74.35)-1.5+1,0
k,113, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-1.5+1,0
k,114, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-3+1,0
k,115, (71.12/2-18.67)+1-1.5, (203.2/2-74.35)-6+1.5+1,0
k,116, (71.12/2-18.67)-2, (203.2/2-74.35)-6+1.5+1,0
k,117, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-6+1.5+1,0
k,118, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-3+1,0
k,119, (71.12/2-18.67)-5+1.5, (203.2/2-74.35)-1.5+1,0
A,101,104,103,102
A,106,105,104,101
A,107,106,101,108
A,108,101,102,109
A,111,114,113,112
A,116,115,114,111
A,117,116,111,118
A,118,111,112,119
smrtsize,1
AATT,1,,1,,1
amesh,all
!arefine,all
nsel,S,loc,y,-127/2
D,all,ALL,0
nsel,all
nsel,S,loc,y,127/2
CM,charge,NODE
*get,nnode,node,,count,max
nsel,all

FINISH
/SOLU
!*
ANTYPE,4
!*
TRNOPT,FULL
LUMPM,0
NSUBST,20,30,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0 !0 rampe 1 step
TIME,100 !temps fin du LS
F, charge, FY, 350/nnode,
lswrite,1

NSUBST,20,700,
OUTRES,ERASE
OUTRES,ALL,1
KBC,1
F, charge, FY, 350/nnode,
TIME,1999 !temps fin du LS
lswrite,2

NSUBST,10,50,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0
TIME,2098 !temps fin du LS
F, charge, FY, 200/nnode,
lswrite,3


NSUBST,20,700,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0
TIME,2999 !temps fin du LS
F, charge, FY, 200/nnode,
lswrite,4

NSUBST,10,50,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0
TIME,3098 !temps fin du LS
F, charge, FY, 0/nnode,
lswrite,5

NSUBST,20,700,
OUTRES,ERASE
OUTRES,ALL,1
KBC,1
TIME,4501 !temps fin du LS
F, charge, FY, 0/nnode,
lswrite,6

LSSOLVE,1,6,1
FINISH
/Post26
ANSOL,2,5,EPEL,X,strainx
ANSOL,3,5,EPEL,Y,strainy
ANSOL,4,5,EPEL,XY,strainxy
ANSOL,5,22,EPEL,X,strainx
ANSOL,6,22,EPEL,Y,strainy
ANSOL,7,22,EPEL,XY,strainxy

LINES,100
PRVAR, 5, 6, 7
/output

PRVAR, 2, 3, 4
/output




