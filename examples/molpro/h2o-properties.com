***,H2O properties
r=1.85                                            !define initial bond distance
theta=104                                         !define initial bond angle (=2 theta)
geometry={o;h1,o,r;h2,o,r,h1,theta}               !geometry input
Basis={
s,1,v;c,1.6;                                      !van Duijneveld 13s, first 6 contracted
p,1,v;c,1.4;                                      !van Duijneveld 8p, first 4 contracted
d,1,3d;f,1,2f                                     !Dunning optimized 3d, 2f
s,2,v;c,1.4;p,2,3p}                               !8s3p basis for H

gexpec,qm,ef1,ef2,fg,,0,0,0.5                     !Field gradient at (0,0,0.5)

hf                                                !closed-shell SCF

pop                                               !Mulliken population analysis using scf density

{dma                                              !distributed multipole analysis using scf density
limit,,3};                                        !limit to rank 3

multi                                             !full valence CASSCF

{pop;                                             !Mulliken population analysis using mcscf density
individual}                                       !give occupations of individual basis functions

{dma;                                             !distributed multipole analysis using mcscf density
limit,,3}                                         !limit to rank 3

{ci                                               !MRCI with casscf reference
natorb,2350.2
dm,2350.2}                                        !save MRCI density matrix on record 2350, file 2

{pop;density,2350.2;                              !Mulliken population analysis using MRCI density
individual}                                       !give occupations of individual basis functions

{property;
density,2350.2;                                   !use property program to compute expectation values
orbital,2350.2;
qm;ef,1;ef,2;fg,,,,0.5}                           !same operators as before
