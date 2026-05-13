orphan  

# Converging atoms

DIRAC presently does not perform an open-shell Hartree--Fock in a strict
sense, rather an average-of-configuration calculation, which amounts to
the optimization of the average energy of a set of configurations (or
determinants) generated from the specification of a given number of open
shells and their electron occupations. In the case of atoms, a
convergence problem can occur when the inner open-shell orbitals are
more stable than outer "closed" shell. From DIRAC21 onwards, we have
atomic supersymmetry and can specify the occupation number of each
$`\kappa`$ value. by using the keyword `SCF_.KPSELE`.

As examples of such calculations, we consider the Nb and Np atoms.

----------------------------------------------------- Example 1: Nb
\[Kr\]4d<sup>4</sup>5s<sup>1</sup>-----------------------------------------------------
:

    **DIRAC
    .WAVE FUNCTION
    .ANALYZE
    **HAMILTONIAN
    .X2C
    **INTEGRALS
    *READIN
    .UNCONTRACT
    **WAVE FUNCTION
    .SCF
    *SCF
    .CLOSED SHELL
     18 18
    .OPEN SHELL
     2
     4/10,0
     1/2,0
    .KPSELE                                                                                             
     5                                                                                                   
     -1  1 -2  2 -3                                                                                      
      8  6 12  4  6                                                                                      
      0  0  0  4  6                                                                                      
      2  0  0  0  0   
    **ANALYZE
    .MULPOP
    *MULPOP
    .VECPOP
     1..oo
     1..oo
    **MOLECULE 
    *BASIS
    .DEFAULT
     dyall.2zp
    *END OF

---------------------------------------------------------- Example 2: Np
\[Rn\]5f<sup>4</sup>6d<sup>1</sup>7s<sup>2</sup>----------------------------------------------------------
:

    **DIRAC                                                                                             
    .WAVE FUNCTION                                                                                      
    .ANALYZE                                                                                            
    **HAMILTONIAN                                                                                       
    .X2C                                                                                                
    **INTEGRALS                                                                                     
    *READIN                                                                                             
    .UNCONTRACT                                                                                         
    **WAVE FUNCTION                                                                                     
    .SCF                                                                                                
    *SCF                                                                                                
    .CLOSED SHELL                                                                                       
     44 44                                                                                              
    .OPEN SHELL                                                                                         
    2                                                                                                   
    4/0,14                                                                                              
    1/10,0                                                                                              
    .KPSELE                                                                                             
    7                                                                                                   
    -1  1  -2  2 -3  3 -4                                                                               
    14 10  20 12 18  6  8                                                                               
     0  0   0  0  0  6  8                                                                               
     0  0   0  4  6  0  0                                                                               
    **ANALYZE                                                                                           
    .MULPOP                                                                                             
    *MULPOP                                                                                             
    .VECPOP                                                                                             
     1..oo                                                                                              
     1..oo
    **MOLECULE 
    *BASIS                     
    .DEFAULT             
     dyall.2zp   
    *END OF
