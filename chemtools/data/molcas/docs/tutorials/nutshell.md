<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/nutshell.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.1. Quickstart Guide for Molcas

[previous](<tut.html> "3. Short Guide to Molcas") | [next](<pbtutorials.html> "3.2. Problem Based Tutorials") | [index](<../genindex.html> "General Index")

# 3.1. Quickstart Guide for Molcas¶

  * Introduction

  * Molcas Environment Setup

  * Customization of Molcas Execution

  * Molcas Command-Line Help System

  * Input Structure and EMIL Commands

  * Basic Examples

    * Simple Calculation on Water

    * Geometry Optimization

    * GASSCF method

    * Solvation Effects

  * Analyzing Results: Output Files and the LUSCUS Program

    * LUSCUS: Grid and Geometry Visualization




## 3.1.1. Introduction¶

Running Molcas requires a small number of operations. This section of the manual, entitled “Quickstart Guide for Molcas” is aimed at those users who want to immediately run a simple Molcas calculation in order to become familiar with the program. Basic hints are included which set the proper environment, build simple input files, run a calculation, and subsequently extract information from the resulting output.

## 3.1.2. Molcas Environment Setup¶

The environment variable (MOLCAS) and Molcas driver (molcas) must be defined in order to run Molcas. The MOLCAS environment variable points to the root directory of the Molcas installation and can be defined by the bash shell command
    
    
    export MOLCAS=/home/molcas/molcas.version
    

The location of the Molcas driver is defined at installation time and is typically located in /usr/local/bin or $HOME/bin. Check to ensure that this directory is included in your path. Otherwise, the path can be extended by the following command:
    
    
    export PATH=$PATH:$HOME/bin
    

In addition, the variable MOLCAS_NPROCS is needed to run Molcas in parallel. This specifies the number of MPI processes that will be used.

It may be also convenient to define environment variables such as WorkDir which points to a directory for intermediate files and Project to define the name of a project:
    
    
    export Project=MyMolecule
    

Molcas will provide default values if they are not explicitly defined. For a discussion of other Molcas environment variables, please see [Section 3.3.2](<tut_emil.html#tut-sec-environment>). All environment variables can either be defined explicitly or entered in a shell script which can be subsequently executed.

## 3.1.3. Customization of Molcas Execution¶

Molcas has flexible control of organizing filenames and directories used during a calculation. The default values used for customization can be altered either by shell variables or a resource file molcasrc which is more preferable. A command molcas setuprc provides guided help if to create such file.

The terminology used in this chapter:

  * `LOG`: the output and error files produced by Molcas.

  * `ProjectName`: the Project name used for file naming.

  * `RUNFILE`: a file used in a calculation will be named as `ProjectName`.Runfile,

  * `WorkDirName`: the WorkDir name used as the directory for temporary/binary files produced by Molcas.

  * `Scratch`: the scratch disk area which provides a path to a parent directory for `WorkDirName`s.

The WorkDir variable used in the Molcas manual is constructed as `Scratch`/`WorkDirName`,

  * `CurrDir`: the submit directory where the Molcas command was issued.

Note, that in this tutorial, it is assumed that the input file is located in `CurrDir`,

  * `OutputDir`: the output directory which is used for storage of extra output files, such as Orbital files and molden files.




It is quite important to understand, that if a user performs two consecutive runs of molcas, using the same scratch area (WorkDir) and project name, Molcas will try to reuse intermediate data, e.g. integrals and orbitals, in order to make a restart of a calculation. This can save time, but can also be can be dangerous if two consecutive calculations are not compatible.

Assuming that molcasrc does not exist, and no environment is set, the command molcas inputfile will use the following defaults:

  * `LOG` is printed to the screen,

  * `OutputDir` and `CurrDir` are defined to be the same directory,

  * `ProjectName` is s taken as the name of inputfile by removing the suffix (before the last . (dot) character),

  * `Scratch` is defined as /tmp/,

  * and `WorkDirName` is defined from the `ProjectName` plus a random suffix.




For example, when a user issues the following commands:
    
    
    cd /home/joe/projects/water
    vi H2O.DFT.input
    molcas H2O.DFT.input
    

the following files will be generated:
    
    
    /home/joe/projects/water/H2O.DFT.ScfOrb
    /home/joe/projects/water/H2O.DFT.scf.molden
    ...
    /tmp/H2O.DFT.15014/H2O.DFT.RunFile
    ...
    

If a flag -f is used in a Molcas command, `LOG` files will be stored in the `CurrDir` directory with a name `ProjectName`.log and `ProjectName`.err.

`ProjectName` can either be set in a shell script running Molcas or included directly into the Molcas command:
    
    
    molcas Project=water H2O.DFT.input
    

will change the default value for `ProjectName` to water.

If the MOLCAS_WORKDIR environment variable is set either as part of Molcas command or is included in the molcasrc file, the name of WorkDir will NOT be random, but determined by the `ProjectName`.

Example:
    
    
    cd /home/joe/projects/water
    vi H2O.DFT.input
    molcas MOLCAS_WORKDIR=/tmp Project=water -f H2O.DFT.input
    

will generate the following files:
    
    
    /home/joe/projects/water/water.log
    /home/joe/projects/water/water.ScfOrb
    ...
    /tmp/water/water.RunFile
    ...
    

For More options to control the behavior of Molcas, run the command molcas setuprc script. The file molcasrc can be used to set global preferences for the Molcas package and/or to set user preferences. The setuprc script creates a molcasrc file (HOME/.Molcas) in a users home directory.

The following molcasrc file for uses the /scratch area as a parent for WorkDirs and Project name generated for the the name of the input file, then removes WorkDir before a calculation followed by subsequent retains of this file when the calculation finished:
    
    
    # Version 1.0
    MOLCAS_MEM=256
    MOLCAS_WORKDIR=/scratch
    MOLCAS_NEW_WORKDIR=YES
    MOLCAS_KEEP_WORKDIR=YES
    MOLCAS_PROJECT=NAME
    

Once the molcasrc is created, it is usually not necessary to use shell script or environment variables to run Molcas.

## 3.1.4. Molcas Command-Line Help System¶

Just by typing molcas help you get access to Molcas Command-Line Help System. There are different options:

  * molcas help produces a list of available programs and utilities.

  * molcas help module yields the list of keywords of the program MODULE.

  * molcas help module keyword offers the detailed description of the keyword.

  * molcas help -t text displays a list of keywords that contain the text word in their description.




## 3.1.5. Input Structure and EMIL Commands¶

Molcas has a modular program structure. The easiest way to run calculations is to prepare an input file in which the different programs are executed sequentially when the the module name (&module) is provided. The keywords of module name then follow, with each entry on a separate line or several entries on one line, separated by ;. In addition to specific program module keywords, Molcas incorporates certain commands (See section on EMIL Commands.) that allow operations such as looping over the modules, allowing partial execution, changing variables, and substituting certain Unix commands.

## 3.1.6. Basic Examples¶

### 3.1.6.1. Simple Calculation on Water¶

Start by preparing a file containing the cartesian coordinates of a water molecule.
    
    
    3
    angstrom
     O       0.000000  0.000000  0.000000
     H       0.758602  0.000000  0.504284
     H       0.758602  0.000000 -0.504284
    

which is given the name water.xyz. In the same directory we prepare the input for the Molcas run. We can name it water.input.

In addition to using an editor to insert atomic coordinates into a file, a coordinate file can be obtained by using a graphical interface program, for example, the LUSCUS module as shown later in this guide.
    
    
    &GATEWAY
     coord=water.xyz
     basis=sto-3g
    &SEWARD
    &SCF
    

The GATEWAY program module combines the molecular geometric of water (In this case, from the external file, water.xyz) and the basis set definition. The SEWARD program module then computes the integrals, and SCF program modules completer the calculation by computing the Hartree–Fock wave function.

To run the calculation, the following command is used:
    
    
    molcas water.input -f
    

The file water.log now contains output from the calculation, and the water.err includes any error messages. In the same directory, other files, including water.scf.molden or water.lus (if the keyword grid_it is added at end of input file) that help to analyze the results graphically with the external graphical viewer LUSCUS or MOLDEN program. Examples of their use are demonstrated below.

In the case of an open-shell calculation (UHF or UDFT), the SCF program is again used. Below, two examples are shown:

  1. A UDFT calculation yielding an approximate doublet by setting the charge to +1, even if they are not pure spin functions:
         
         &GATEWAY
          coord=water.xyz
          basis=sto-3g
         &SEWARD
         &SCF
          charge=+1
          uhf; ksdft=b3lyp
         

  2. A triplet state (using keyword ZSPIn to specify that there are two more \\(\alpha\\) than \\(\beta\\) electrons) states:
         
         &GATEWAY
          coord=water.xyz
          basis=sto-3g
         &SEWARD
         &SCF
          zspin=2
          uhf; ksdft=b3lyp
         




### 3.1.6.2. Geometry Optimization¶

In the next example, a DFT/B3LYP geometry optimization is performed on the ground state of the water molecule. Notice that, after `&gateway` has defined the coordinates and basis set definition, the EMIL commands >>> Do while and >>> EndDo are employed to form a loop with the SEWARD, SLAPAF, and SCF programs until convergence of geometry optimization is reached. Program SEWARD computes the integrals in atomic basis, SCF computes the DFT energy, and the program SLAPAF controls the geometry optimization and uses the module ALASKA to compute the gradients of the energy with respect to the degrees of freedom. SLAPAF generates the new geometry to continue the iterative structure optimization process and checks to determine convergence parameters are satisfied notifying Molcas and stopping the loop.
    
    
    &GATEWAY
     coord=water.xyz
     basis=ANO-S-MB
    >>> Do While
      &SEWARD
      &SCF
        ksdft=b3lyp
      &SLAPAF
    >>> EndDo
    

The above example illustrates the default situation of optimizing to a minimum geometry without any further constraint. If other options are required such as determining a transition state, obtaining a states crossing, or imposing a geometry constraint, specific input should be added to program SLAPAF.

Figure 3.1.6.1 The acrolein molecule.¶

One of the most powerful aspects of Molcas is the possibility of computing excited states with multiconfigurational approaches. The next example demonstrates a calculation of the five lowest singlet roots in a State-Average (SA) CASSCF calculation using the RASSCF program. It also illustrates the addition of the CASPT2 program to determine dynamical correlation which provides accurate electronic energies at the CASPT2 level. The resulting wave functions are used in the RASSI module to calculate state-interaction properties such as oscillator strengths and other properties.
    
    
    &gateway
    Coord
     8
    Acrolein coordinates in angstrom
     O     -1.808864   -0.137998    0.000000
     C      1.769114    0.136549    0.000000
     C      0.588145   -0.434423    0.000000
     C     -0.695203    0.361447    0.000000
     H     -0.548852    1.455362    0.000000
     H      0.477859   -1.512556    0.000000
     H      2.688665   -0.434186    0.000000
     H      1.880903    1.213924    0.000000
    Basis=ANO-S-MB
    Group=Nosym
    &SEWARD
    &RASSCF
      nactel  = 6 0 0
      inactive= 12
      ras2    = 5
      ciroot  = 5 5 1
    &CASPT2
      multistate=5 1 2 3 4 5
    &RASSI
      Nr of Job=1 5; 1 2 3 4 5
      EJob
    

Notice that the Group with the option Nosym has been used to prevent GATEWAY from identifying the symmetry of the molecule (\\(C_s\\) in this case). Otherwise, the input of the RASSCF program will have to change to incorporate the classification of the active space into the corresponding symmetry species. Working with symmetry will be skipped at this stage, although its use is very convenient in many cases. A good strategy is to run only GATEWAY and let the program guide you.

The RASSCF input describes the active space employed, composed by six active electrons distributed in five active orbitals. By indicating twelve inactive orbitals (always doubly occupied), information about the total number of electrons and the distribution of the orbitals is then complete. Five roots will be obtained in the SA-CASSCF procedurei, and all them will be computed at the CASPT2 level to obtain the transition energies at the higher level of theory. Further, the RASSI will compute the transition properties, in particular, transition dipole moments and oscillator strengths.

### 3.1.6.3. GASSCF method¶

In certain cases it is useful/necessary to enforce restrictions on electronic excitations within the active space beyond the ones accessible by RASSCF. These restrictions are meant to remove configurations that contribute only marginally to the total wave function. In Molcas this is obtained by the GASSCF approach [[18](<../references.html#id192> "D. Ma, G. Li Manni, L. Gagliardi. J. Chem. Phys., 135 \(2011\) 044128.")]. In GASSCF an arbitrary number of active spaces may be chosen. All intra-space excitations are allowed (Full-CI in subspaces). Constraints are imposed by user choice on inter-space excitations. This method, like RASSCF, allows restrictions on the active space, but they are more flexible than in RASSCF. These restrictions are particularly useful when the cost of using the full CI expansion of the active space is beyond reach. These restrictions allow GASSCF to be applied to larger and more complex systems at affordable cost. Instead of a maximum number of holes in RAS1 and particles in RAS3, accumulated minimum and maximum numbers of electrons are specified for GAS1, GAS1+GAS2, GAS1+GAS2+GAS3, etc. in order to define the desired CI expansion. The GAS scheme reduces to CAS or RAS when one or three spaces are chosen and restrictions on electron excitations are adequately imposed. When and how to use the GAS approach? We consider three examples: (1) an organometallic material with separated metal centers and orbitals not delocalized across the metal centers. One can include the near degenerate orbitals of each center in its own GAS space. This implies that one may choose as many GAS spaces as the number of multiconfigurational centers. (2) Lanthanide or actinide metal compounds where the \\(f\\)-electrons require a MC treatment but they do not participate in bonding neither mix with \\(d\\) orbitals. In this case one can put the \\(f\\) orbitals and their electrons into one or more separated GAS spaces and not allow excitations from and/or to other GAS spaces. (3) Molecules where each bond and its correlating anti-bonding orbital could form a separate GAS space as in GVB approach. Finally, if a wave function with a fixed number of holes in one or more orbitals is desired, without interference of configurations where those orbitals are fully occupied the GAS approach is the method of choice instead of the RAS approach. There is no rigorous scheme to choose a GAS partitioning. The right GAS strategy is system-specific. This makes the method versatile but at the same time it is not a black box method. An input example follow:
    
    
    &RASSCF
    nActEl
     6 0 0
    FROZen
    0 0 0 0 0 0 0 0
    INACTIVE
    2 0 0 0 2 0 0 0
    GASScf
    3
     1 0 0 0 1 0 0 0
    2 2
     0 1 0 0 0 1 0 0
    4 4
     0 0 1 0 0 0 1 0
    6 6
    DELEted
    0 0 0 0 0 0 0 0
    

In this example the entire active space counts six active electrons and six active orbitals. These latter are partitioned in three GAS spaces according to symmetry consideration and in the spirit of the GVB strategy. Each subspace has a fixed number of electrons, _two_ , and no interspace excitations are allowed. This input shows clearly the difference with the RAS approach.

### 3.1.6.4. Solvation Effects¶

Molcas incorporates the effects of the solvent using several models. The most common is the cavity-based reaction-field Polarizable Continuum Model (PCM) which is invoked by adding the keyword RF-input to the SEWARD code and is needed to compute the proper integrals.
    
    
    &GATEWAY
      coord=CH4.xyz
      Basis=ANO-S-MB
    &SEWARD
      RF-Input
       PCM-Model
       Solvent=Water
      End of RF-Input
    &RASSCF
      Nactel=8 0 0
      Inactive=1
      Ras2=8
    &CASPT2
      rfpert
    

The reaction field is computed in a self-consistent manner by the SCF or RASSCF codes and added as a perturbation to the Hamiltonian in the CASPT2 method with the keyword RFPErt.

## 3.1.7. Analyzing Results: Output Files and the LUSCUS Program¶

Molcas provides a great deal of printed information in output files, and the printing level is controlled by the environmental variable MOLCAS_PRINT. By default this value is set to two, but can be modified by environmental variable MOLCAS_PRINT Typical Molcas output contains the program header and input information, conditions of the calculation, the number of steps to achieve convergence, the energies and wave functions, and final results, including in many cases the molecular orbital coefficients as well as an analysis of the properties for the computed states.

### 3.1.7.1. LUSCUS: Grid and Geometry Visualization¶

Molcas developers have developed a graphical interface that can be used both to create input for the Molcas program and to analyze the results in a graphical manner by visualizing molecular orbitals, density plots, and other output properties.

The first version of the code has the name GV (stands for Grid Viewer, or Geometry Visualization. By an accident, the name also matches the nicknames of the main developers). GV program uses a very limited set of graphic libraries, and thus has very primitive user interface.

The next generation of GV program has the name LUSCUS. Luscus re-uses the code of GV, and so GV users can use the same key combinations to operate with LUSCUS. At the same time, LUSCUS provides a user-friendly interface, and contains many new options, compared to GV.

LUSCUS can be obtained from <http://luscus.sourceforge.net/>, or from <https://www.molcas.org/LUSCUS>.

LUSCUS can read the files only in one format: Luscus internal format (.lus). This format contains two sections: XYZ cartesian coordinates, and XML formated data. It means that a standard XYZ file is a valid file in LUSCUS format.

Files with different formats, e.g. molden files, can be understood by LUSCUS since they can be converted to LUSCUS format by a corresponding plug-in. For instance, opening a file with the extension .molden, LUSCUS automatically runs a plug-in to convert a file from molden format to LUSCUS format. Saving a LUSCUS file as a Molcas orbital file will automatically run a converter from LUSCUS format to Orbital format.

  * luscus xyz_file: reads coordinates from a cartesian coordinate file.

A molecule can be visualized and modified with the use of the left-button of the mouse and the keyboard. Below are some of the most useful commands.

Left mouse click | Select atoms (if two, a bond is selected, if three bond angle, if four a dihedral angle  
---|---  
Left mouse + Shift click | Mark/unmark atoms to/from the group  
Middle mouse/Space | Remove selection, or marking  
Insert key | Insert atom  
PageUp, PageDown | Alter type of selected atom or bond  
Delete/Supress key | Delete a selected atom  
+/- | Change a value of selected bond/angle in steps  
Backspace | Undo last action  
Home | Set selected atom to center of coordinates  
F8 key | Find or apply symmetry  
  * luscus molden_file: reads (check the comment about plug-in) from MOLDEN files such as wavefunction.molden, freq.molden, and geo.molden.

Note that Molcas produces molden files with several extensions, so it is recommended to visualize these files by using LUSCUS.

  * luscus grid_file: reads coordinates and densities and molecular orbitals from a binary grid_file.

This file is generated by GRID_IT and, by default, placed in the $WorkDir directory with the name $Project.lus. The program allows displaying total densities, molecular orbitals, and charge density differences.

If Molcas and Luscus are installed locally, LUSCUS can also be called from user input as shown in the following example:
        
        &GATEWAY
           coord = acrolein.xyz
           basis = ANO-L-MB
        &SEWARD
        &SCF
        &GRID_IT
        ALL
        
        * running external GUI program luscus
        
        ! luscus $Project.lus
        
        * User has to select active space and save GvOrb file!
        
        &RASSCF
        Fileorb=$CurrDir/$Project.GvOrb
        

Note, that in the example above, the GRID_IT program will generate a $Project.lus file which LUSCUS then uses, eliminating the need for defining $Project.lus and allowing this file to be overwritten. RASSCF will read starting orbitals from the $Project.GvOrb file.




### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tut.html>)
    * 3.1. Quickstart Guide for Molcas
      * 3.1.1. Introduction
      * 3.1.2. Molcas Environment Setup
      * 3.1.3. Customization of Molcas Execution
      * 3.1.4. Molcas Command-Line Help System
      * 3.1.5. Input Structure and EMIL Commands
      * 3.1.6. Basic Examples
      * 3.1.7. Analyzing Results: Output Files and the LUSCUS Program
    * [3.2. Problem Based Tutorials](<pbtutorials.html>)
    * [3.3. Program Based Tutorials](<tutorials.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut.html> "3. Short Guide to Molcas") | [next](<pbtutorials.html> "3.2. Problem Based Tutorials") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/nutshell.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
