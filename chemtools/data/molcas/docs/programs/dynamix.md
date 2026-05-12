<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/dynamix.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.12. DYNAMIX

[previous](<dmrgscf.html> "4.2.11. DMRGSCF") | [next](<embq.html> "4.2.13. EMBQ ¤") | [index](<../../genindex.html> "General Index")

# 4.2.12. DYNAMIX¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * General keywords

    * Input examples

  * Dynamixtools




The DYNAMIX program performs molecular dynamics (MD) simulations in Molcas. Here the nuclei are moved according to the classical Newton’s equations which are solved numerically using the velocity Verlet algorithm [[78](<../../references.html#id101> "W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson. J. Chem. Phys., 76 \(1982\) 637-649.")]. The algorithm requires coordinates, velocities and forces as input. DYNAMIX can be used with any electronic structure method in Molcas. Also environmental effects can be taken into account in the MD simulation: the solvent can be considered implicitly using the reaction field keyword in GATEWAY or explicitly in hybrid QM/MM calculation which requires the ESPF program.

When multiple electronic states are involved in a MD simulation, a trajectory surface hopping (TSH) algorithm allows non-adiabatic transitions between different states. This TSH algorithm evaluates the change of the wavefunction along the trajectory and induces a hop if certain criteria a met (for further details read the RASSI section). In the current implementation the surface hopping algorithm can be used only with state averaged CASSCF wavefunction. However, an extension for CASPT2 and other methods are in preparation.

The Tully algorithm is available in a separate module SURFACEHOP.

## 4.2.12.1. Dependencies¶

The coordinates and the forces are required by the DYNAMIX program. DYNAMIX reads the initial coordinates from the RUNFILE and updates them in each iteration. In addition DYNAMIX depends on the ALASKA program, since it generates forces.

## 4.2.12.2. Files¶

### 4.2.12.2.1. Input files¶

velocity.xyz
    

Contains the initial velocities of the MD simulation.

### 4.2.12.2.2. Output files¶

RUNFILE
    

Trajectory information such as current time, velocities, etc. are stored in this file.

md.xyz
    

The coordinates for each step of the MD trajectory are saved here.

md.energies
    

The potential, kinetic and total energies are written to this file. In case of multiple electronic states, the energies of all roots are saved.

## 4.2.12.3. Input¶

This section describes the input syntax of DYNAMIX in the Molcas program package. In general a MD simulation requires a DoWhile or ForEach loop which contains several programs to compute the energy and ALASKA for subsequent gradient computation. The DYNAMIX input begins with the program name, and is followed by the only compulsory keyword VELV which specifies the velocity Verlet algorithm:
    
    
    &DYNAMIX
    VELV
    

### 4.2.12.3.1. General keywords¶

VELVerlet
    

This keyword specifies the velocity Verlet algorithm [[78](<../../references.html#id101> "W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson. J. Chem. Phys., 76 \(1982\) 637-649.")] to solve Newton’s equations of motion. It’s the only compulsory keyword in the program.

DTime
    

Defines the \\(\delta t\\) which is the time step in the MD simulation and which is used for the integration of Newton’s equations of motion. The program expects the time to be given in floating point format and in atomic unit of time (1 a.u. of time = \\(2.42\cdot10^{-17}\\) s). (Default = 10).

VELOcities
    

Specifies how the initial velocities are generated. This keyword is followed by an integer on the next line. The internal unit of the velocities is [bohr\\(\cdot\\)(a.u. of time)\\(^{-1}\\)].

**0** — Zero velocities. (Default)

**1** — The velocities are read from the file $Project.velocity.xyz in $WorkDir. This file contains velocities in the xyz format given in the same order as the atoms in coordinate file. The unit of the velocities is [bohr\\(\cdot\\)(a.u. of time)\\(^{-1}\\)].

**2** — This option allows to read in mass-weighted velocities from the file $Project.velocity.xyz in [bohr\\(\cdot\sqrt{\text{a.m.u.}}\cdot\\)(a.u. of time)\\(^{-1}\\)].

**3** — This option takes random velocities from a Maxwell–Boltzmann distribution, at a given temperature, assuming that every component of the velocity can be considered as an independent gaussian random variable.

THERmostat
    

Regulates the control of the temperature by scaling the velocities. The option is an integer given on the next line.

**0** — No velocity scaling. (Default)

**1** — The velocities are scaled in order to keep the total energy constant.

**2** — The velocities are scaled according to the Nosé–Hoover chain of thermostats algorithm, used to perform molecular symulation at constant temperature, resulting in statistics belonging to the canonical ensemble (NVT).

TEMPerature
    

Defines the numerical value of the temperature, which is used together with the Nosé–Hoover chain of thermostats to perform molecular dynamics at constant temperature. (Default = 298.15 K)

HOP
    

Enables the trajectory surface hopping algorithm if the integer given in the next line is bigger than 0. The integer also specifies how many non-adiabatic transitions are allowed between electronic states.

OUT
    

Enables dynamics in reduced dimensionality. This keyword is followed by an integer on the next line, which defines the number of nuclear coordinates to project out from the trajectory (default 0). The coordinates to project out are then read from the files out.00X.xyz, in the xyz format given in the same order as the atoms in coordinate file. The projection is performed in mass-weighted coordinates and can be applied directly to normal modes for instance. Note: In case of several coordinates to project out, these are first orthogonalised (in mass-weighted coordinates).

IN
    

Enables dynamics in reduced dimensionality. This keyword is followed by an integer on the next line, which defines the number of nuclear coordinates to keep in in the trajectory (default 3 * number of atoms). The coordinates to keep in are then read from the files in.00X.xyz, in the xyz format given in the same order as the atoms in coordinate file. The projection is performed in mass-weighted coordinates and can be applied directly to normal modes for instance. Note: In case of several coordinates to keep in, these are first orthogonalised (in mass-weighted coordinates).

RESTART
    

This keyword allows to restart the trajectory at a given time. The time is given on the next line in atomic units.

H5RESTART
    

This keyword allows to restart a trajectory calculation from an HDF5 file. The name of the restart file is given on the next line.

### 4.2.12.3.2. Input examples¶

The following example shows the input for an excited state CASSCF molecular dynamics simulation of a methaniminium cation using the DYNAMIX program. The DoWhile loop allows 1000 steps with 10 a.u. of time step size which leads to a total duration of 242 fs. In the RASSCF program the second root is selected for gradient calculation using the keyword MDRLXR. This input assumes that the a JOBIPH file with orbitals is already given. In each iteration the JOBIPH is updated to achieve a fast convergence of the CASSCF wavefunction. A Nosé–Hoover chain of thermostats, enabled with THERmo=2, is used to reproduce dynamics at constant temperature, where the initial velocities are taken from a Maxwell–Boltzmann distribution at 300 K.
    
    
    &GATEWAY
     COORD
     6
     angstrom
     C  0.00031448  0.00000000  0.04334060
     N  0.00062994  0.00000000  1.32317716
     H  0.92882820  0.00000000 -0.49115611
     H -0.92846597  0.00000000 -0.49069213
     H -0.85725321  0.00000000  1.86103989
     H  0.85877656  0.00000000  1.86062860
     BASIS= 3-21G
     GROUP= nosym
    
    >> EXPORT MOLCAS_MAXITER=1000
    >> DOWHILE
    
    &SEWARD
    
    >> IF ( ITER = 1 )
    
    &RASSCF
     LUMORB
     FileOrb= $Project.GssOrb
     Symmetry= 1
     Spin= 1
     nActEl= 2 0 0
     Inactive= 7
     RAS2= 2
     CIroot= 3 3 1
    
    >> COPY $Project.JobIph $Project.JobOld
    
    >> ENDIF
    
    &RASSCF
     JOBIPH; CIRESTART
     Symmetry= 1
     Spin= 1
     nActEl= 2 0 0
     Inactive= 7
     RAS2= 2
     CIroot= 3 3 1
     MDRLXR= 2
    
    >> COPY $Project.JobIph $Project.JobOld
    
    &ALASKA
    
    &DYNAMIX
     VELVer
     DT= 10.0
     VELO= 3
     THER= 2
     TEMP=300
     HOP= 1
    
    >> END DO
    

## 4.2.12.4. Dynamixtools¶

This tool can be found into the Tools/ folder and it will provide some general tools to manage molecular dynamics calculations. At the moment it can be used to generate intial conditions (geometries and momenta) based on a frequency calculation using several sampling methods. It is working with a freq.molden file (.h5 support coming soon…).

From the command prompt:
    
    
    $ python3 dynamixtools.py -h
    usage: dynamixtools.py [-h] [-s SEED] [-l LABEL] [-i I] [-c CONDITION] [-t TEMP] [-v] [-T] [-D] [-m METHOD]
    
    optional arguments:
    -h, --help            show this help message and exit
    -s SEED, --seed SEED  indicate the SEED to use for the generation of randoms
    -l LABEL, --label LABEL
                          label for your project (default is "geom")
    -i I, --input I       path of the frequency h5 or molden file
    -c CONDITION, --condition CONDITION
                          number of initial conditions (default 1)
    -t TEMP, --temperature TEMP
                          temperature in kelvin for the initial conditions
    -v, --verbose         more verbose output
    -T, --TEST            keyword use to test the routines
    -D, --DIGIT           keyword to suppress the counter in the filename (needed for debug)
    -m METHOD, --method METHOD
                          Keyword to specify the sampling method:
                          1 Initial conditions based on the molecular vibrational frequencies and energies sampled from a Boltzmann distribution (Default).
                          2 Thermal normal mode sampling where the cumulitative distribution function for a classical boltzmann distribution at temperature T is used to approximate the energy of each mode.
                          3 Wigner distribution for the ground vibrational state, n=0.
    

Having a water.freq.molden file, this is the command to generate 200 initial conditions using 3435432 as seed and a temperature of 300 kelvin:
    
    
    $ python3 dynamixtools.py -i water.freq.molden -t 300 -c 200 -s 3435432
    

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * [4.2.3. CASPT2](<caspt2.html>)
      * [4.2.4. CASVB](<casvb.html>)
      * [4.2.5. CCSDT](<ccsdt.html>)
      * [4.2.6. CHCC](<chcc.html>)
      * [4.2.7. CHT3](<cht3.html>)
      * [4.2.8. CMOCORR ¤](<cmocorr.html>)
      * [4.2.9. CPF](<cpf.html>)
      * [4.2.10. DIMERPERT ¤](<dimerpert.html>)
      * [4.2.11. DMRGSCF](<dmrgscf.html>)
      * 4.2.12. DYNAMIX
      * [4.2.13. EMBQ ¤](<embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<espf.html>)
      * [4.2.15. EXPBAS](<expbas.html>)
      * [4.2.16. EXTF](<extf.html>)
      * [4.2.17. FALCON ¤](<falcon.html>)
      * [4.2.18. FALSE](<false.html>)
      * [4.2.19. FFPT](<ffpt.html>)
      * [4.2.20. GATEWAY](<gateway.html>)
      * [4.2.21. GENANO](<genano.html>)
      * [4.2.22. GEO ¤](<geo.html>)
      * [4.2.23. GRID_IT](<grid_it.html>)
      * [4.2.24. GUESSORB](<guessorb.html>)
      * [4.2.25. GUGA](<guga.html>)
      * [4.2.26. GUGACI](<gugaci.html>)
      * [4.2.27. GUGADRT](<gugadrt.html>)
      * [4.2.28. LEVEL](<level.html>)
      * [4.2.29. LOCALISATION](<localisation.html>)
      * [4.2.30. LOPROP](<loprop.html>)
      * [4.2.31. MBPT2](<mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<mckinley.html>)
      * [4.2.33. MCLR](<mclr.html>)
      * [4.2.34. MCPDFT](<mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<mknemo.html>)
      * [4.2.36. MOTRA](<motra.html>)
      * [4.2.37. MPPROP](<mpprop.html>)
      * [4.2.38. MPSSI](<mpssi.html>)
      * [4.2.39. MRCI](<mrci.html>)
      * [4.2.40. MULA](<mula.html>)
      * [4.2.41. NEMO ¤](<nemo.html>)
      * [4.2.42. NEVPT2](<nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<poly_aniso.html>)
      * [4.2.45. QMSTAT](<qmstat.html>)
      * [4.2.46. QUATER](<quater.html>)
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * [4.2.52. SEWARD](<seward.html>)
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<dmrgscf.html> "4.2.11. DMRGSCF") | [next](<embq.html> "4.2.13. EMBQ ¤") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/dynamix.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
