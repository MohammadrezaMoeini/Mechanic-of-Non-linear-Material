
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
MPDATA,EX,1,,4130.919044305096, !!!! Entrer la valeur de E0															 !
MPDATA,PRXY,1,,0.3959705815395929, !!!! Entrer la valeur de nu0													     !
!   																											 !
!                                                                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!													     !
! Module de relaxation en compressibilite : alpha_k_ansys														 !
!																												 !
TB,PRONY,1,1,6,BULK     !!!! Ne pas oublier d'entrer le nombre de paires				                		 !
TBTEMP,0
	
! Rajouter autant de paires que necessaire, 3 paires par lignes maximimum                                        !
																									 !
TBDATA,,1.9690929208446657e-05,1.0945794310200747,
 
TBDATA,,0.12073488154602137,0.034146544309014766,
 
TBDATA,,0.23567976369495028,0.032228123206906996, 

TBDATA,,1.2610499620561115e-06,0.003333096352912752, 

TBDATA,,1.080950739140102e-08,0.0031622783405423544,

TBDATA,,0.6435643915321727,0.00031622776601775295,

       			            		         
!                                                                                                                !
!                                                                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!														 !
! Module de relaxation en cisaillement	: alpha_mu_ansys														 !
!																												 !
TB,PRONY,1,1,39,SHEAR    !!!! Ne pas oublier d'entrer le nombre de paires					                	 !
TBTEMP,0

TBDATA,,1.9690929208446657e-05,1.0945794310200747,

TBDATA,,0.06735432041807045,1.0945794310200756,

TBDATA,,0.06735432041807059,1.0945794310200754,

TBDATA,,0.06735432041807052,1.094579431020075,

TBDATA,,1.690351281019262e-08,1.0000031173189334,

TBDATA,,1.6903512810089685e-08,0.31622777376861483,

TBDATA,,1.6903512810087412e-08,0.31622777376861455

TBDATA,,1.6903512810085748e-08,0.31622777376861433,

TBDATA,,1.6903512810104743e-08,0.3162277737686141,

TBDATA,,2.5187731188888507e-08,0.3162277660222327,

TBDATA,,2.5187731188892795e-08,0.1000000034410692,

TBDATA,,2.5187731188911896e-08,0.10000000344106913,

TBDATA,,2.5187731188924565e-08,0.1000000034410691,	

TBDATA,,2.5187731188917157e-08,0.10000000344106907,

TBDATA,,0.05266120066644074,0.10000000000487747,

TBDATA,,0.052661200666440734,0.03414654430901497,

TBDATA,,0.052661200666440734,0.034146544309014946,

TBDATA,,0.05266120066644082,0.034146544309014926,

TBDATA,,0.05266120066644082,0.03414654430901486,

TBDATA,,1.3435943174390545e-05,0.010399009446897239,

TBDATA,,1.3435943174388809e-05,0.010000205725896284,

TBDATA,,1.3435943174388258e-05,0.010000205725896256,
 
TBDATA,,1.3435943174392275e-05,0.01000020572589625,
  
TBDATA,,1.3435943174390784e-05,0.010000205725896223,
  
TBDATA,,0.033624498523489456,0.01000020572589618,
   
TBDATA,,0.033624498523491565,0.0033330963529131633, 
 
TBDATA,,0.033624498523492474,0.003333096352912961,
  
TBDATA,,0.03362449852349151,0.003333096352912959, 
  
TBDATA,,0.03362449852349358,0.003333096352912957,   
																																																			 
TBDATA,,0.03132135851189209,0.0010000000000001208, 
   
TBDATA,,0.031321358511892286,0.0003328540215766033, 
   
TBDATA,,0.03132135851189263,0.0003328540215765991, 
   
TBDATA,,0.03132135851189265,0.000332854021576599, 
   
TBDATA,,0.03132135851189278,0.00033285402157659885,    

TBDATA,,0.015025143845608404,0.00011212577272333476,
    
TBDATA,,0.015025143845607047,0.00010266942121170823, 
   
TBDATA,,0.015025143845608366,0.00010266942121170822,
    
TBDATA,,0.015025143845610297,0.0001026694212117081, 
   
TBDATA,,0.0150251438456092,0.00010266942121170567,   
                                               
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


