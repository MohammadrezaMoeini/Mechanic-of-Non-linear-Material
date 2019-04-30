# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:03:16 2018

@author: momoe
"""

FINISH
/CLEAR
/COM,  Structural
!
/FILNAME,geom_simple,1
!
! Ce programme pour tester parametres visco
! sur une eprouvette de geometrie simple.
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
MPDATA,EX,1,,4130.91904431, !!!! Entrer la valeur de E0															 !
MPDATA,PRXY,1,,0.39597058, !!!! Entrer la valeur de nu0													     !
!   																											 !
!                                                                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!													     !
! Module de relaxation en compressibilite : alpha_k_ansys														 !
!																												 !
TB,PRONY,1,1,8,BULK     !!!! Ne pas oublier d'entrer le nombre de paires				                		 !
TBTEMP,0		
																								 !
TBDATA,, 1.0802036367050221e-10 , 0.9999968826907845 ,
TBDATA,, 3.119391674838021e-10 , 3.162277660114432 ,
TBDATA,, 0.12073488154602724 , 9.99999999951225 ,
TBDATA,, 0.23567976369496196 , 31.028800330069608 ,
TBDATA,, 1.2610499618651727e-06 , 96.16300524646329 ,
TBDATA,, 1.0809507392988596e-08 , 316.2276979794547 ,
TBDATA,, 1.821809539504018e-11 , 999.9999982002053 ,
TBDATA,, 0.6435643915321558 , 3162.277660159233 ,

!                                                                                                                !
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
!                                                                                                                !
!																												 !
!fin des donnees pour les proprietes																			 !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Charge 283 N <=> 5 MPa
LT=165-25*2
LG=57
RD=76
WT=19
WG=12.6

SECTYPE,1,shell
secdata, 4.49,1,0,5
BLC4, -WG/2, -LG/2, WG, LG,
BLC4, -WT/2, LG/2, WT, (LT-LG)/2,
BLC4, -WT/2, -LT/2, WT, (LT-LG)/2,
CYL4,(WG/2+RD),LG/2,76
CYL4,-(WG/2+RD),LG/2,76
CYL4,-(WG/2+RD),-LG/2,76
CYL4,(WG/2+RD),-LG/2,76
FLST,2,3,5,ORDE,2
FITEM,2,1
FITEM,2,-3
FLST,3,4,5,ORDE,2
FITEM,3,4
FITEM,3,-7
ASBA,P51X,P51X
AGLUE,ALL
NUMCMP,ALL

smrtsize,1
AATT,1,,1,,1
AMESH,1
AMESH,2,3,1

nsel,S,loc,y,-LT/2
D,all,ALL,0
nsel,all
nsel,S,loc,y,LT/2
CM,charge,NODE
*get,nnode,node,,count,
nsel,all

FINISH
/SOLU
!*
ANTYPE,4
!*
TRNOPT,FULL
LUMPM,0
!NSUBST,20,30,
NSUBST,50,2000,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0 !0 rampe 1 step
TIME,5.5 !temps fin du LS
F, charge, FY, 283/nnode,      ! Vérifier force pour obtenir 5MPA max.
lswrite,1

!NSUBST,20,700,
NSUBST,50,2000,
OUTRES,ERASE
OUTRES,ALL,1
KBC,1
F, charge, FY, 283/nnode,
TIME,905 !temps fin du LS
lswrite,2

!NSUBST,20,30,
NSUBST,50,2000,
OUTRES,ERASE
OUTRES,ALL,1
KBC,0
F, charge, FY, 0/nnode,
TIME,910.5 !temps fin du LS
lswrite,3

!NSUBST,20,700,
NSUBST,50,2000,
OUTRES,ERASE
OUTRES,ALL,1
KBC,1
F, charge, FY, 0/nnode,
TIME,4000 !temps fin du LS
lswrite,4

lssolve,1,4,1

FINISH
/Post26

ANSOL,2,895,EPEL,X,strainx
ANSOL,3,895,EPEL,Y,strainy
ANSOL,4,895,EPEL,XY,strainxy

PLVAR,3,

LINES,100
PRVAR, 2, 3, 4


