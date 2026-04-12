***,C2  ground state
print,orbitals
  
 geometry={ang
C
C  1 b1
}

b1 = 1.0

 set,charge=0 
basis={
!  cc-pVTZ  EMSL  Basis Set Exchange Library   5/26/09 9:20 AM
! Elements                             References
! --------                             ----------
! H     : T.H. Dunning, Jr. J. Chem. Phys. 90, 1007 (1989).
! He    : D.E. Woon and T.H. Dunning, Jr. J. Chem. Phys. 100, 2975 (1994).
! Li - Ne: T.H. Dunning, Jr. J. Chem. Phys. 90, 1007 (1989).
! Na - Mg: D.E. Woon and T.H. Dunning, Jr.  (to be published)
! Al - Ar: D.E. Woon and T.H. Dunning, Jr.  J. Chem. Phys. 98, 1358 (1993).
! Ca     : J. Koput and K.A. Peterson, J. Phys. Chem. A, 106, 9595 (2002).
! Sc - Zn: N.B. Balabanov and K.A. Peterson, J. Chem. Phys, 123, 064107 (2005)
! Ga - Kr: A.K. Wilson, D.E. Woon, K.A. Peterson, T.H. Dunning, Jr., J. Chem. Phys., 110, 7667 (1999)
! 
! CARBON       (10s,5p,2d,1f) -> [4s,3p,2d,1f]
! CARBON       (10s,5p,2d,1f) -> [4s,3p,2d,1f]
s, C , 8236.0000000, 1235.0000000, 280.8000000, 79.2700000, 25.5900000, 8.9970000, 3.3190000, 0.9059000, 0.3643000, 0.1285000
c, 1.10, 0.0005310, 0.0041080, 0.0210870, 0.0818530, 0.2348170, 0.4344010, 0.3461290, 0.0393780, -0.0089830, 0.0023850
c, 1.10, -0.0001130, -0.0008780, -0.0045400, -0.0181330, -0.0557600, -0.1268950, -0.1703520, 0.1403820, 0.5986840, 0.3953890
c, 8.8, 1.0000000
c, 10.10, 1.0000000
p, C , 18.7100000, 4.1330000, 1.2000000, 0.3827000, 0.1209000
c, 1.5, 0.0140310, 0.0868660, 0.2902160, 0.5010080, 0.3434060
c, 4.4, 1.0000000
c, 5.5, 1.0000000
d, C , 1.0970000, 0.3180000
c, 1.1, 1.0000000
c, 2.2, 1.0000000
f, C , 0.7610000
c, 1.1, 1.0000000
}
 
 {hf;wf,12,1,0;orbprint,6}
  
 {multi,maxit=100;
 wf,12,1,0;
 expec2,lzz;orbprint,6;}
  
{rs2c,maxit=100,g3;wf,12,1,0;print,ref=1}

{optg,maxit=100}
E_corr(1)=ENERGY
theory(1)="CASPT2"
b(1)="VTZ"
{frequencies;thermo}
E_ZPE(1)=ZPE
E_Enth(1)=HTOTAL-ENERGY




 
