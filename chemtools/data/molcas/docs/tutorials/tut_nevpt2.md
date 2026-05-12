<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_nevpt2.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.9. NEVPT2 — \\(n\\)-Electron Valence State Second-Order Perturbation Theory

[previous](<tut_caspt2.html> "3.3.8. CASPT2 — A Many Body Perturbation Program") | [next](<tut_rassi.html> "3.3.10. RASSI — A RAS State Interaction Program") | [index](<../genindex.html> "General Index")

# 3.3.9. NEVPT2 — \\(n\\)-Electron Valence State Second-Order Perturbation Theory¶

NEVPT2 is a second-order perturbation theory with a CAS (or a CAS-like) reference wavefunction originally developed by Angeli et al. [[21](<../references.html#id346> "C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu. J. Chem. Phys., 114 \(2001\) 10252."), [22](<../references.html#id347> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. Chem. Phys. Lett., 350 \(2001\) 297-305."), [23](<../references.html#id348> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. J. Chem. Phys., 117 \(2002\) 9138-9153."), [24](<../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")] In contrast to CASPT2, it uses a Dyall Hamiltonian [[25](<../references.html#id350> "K. G. Dyall. J. Chem. Phys., 102 \(1995\) 4909-4918.")] as the zeroth-order Hamiltonian and is therefore inherently free of intruder states and parameters such as the IPEA shift. NEVPT2 exists in two formulations – the strongly- (SC-) and the partially-contracted NEVPT2 (PC-NEVPT2), which differ in the basis of the first-order wavefunction expansion.

The implementation in the NEVPT2 program is based on the original NEVPT2 implementation by Angeli et al. [[23](<../references.html#id348> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. J. Chem. Phys., 117 \(2002\) 9138-9153."), [24](<../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")], with the implementation of the QCMaquis DMRG reference wave function and Cholesky decomposition for the two-electron integrals [[26](<../references.html#id351> "L. Freitag, S. Knecht, C. Angeli, M. Reiher. J. Chem. Theory Comput., 13 \(2017\) 451-459.")]. For excited states both single-state and multi-state calculations with the QD-NEVPT2 approach [[24](<../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")] are supported.

## 3.3.9.1. Running a NEVPT2 calculation¶

Prior to running a NEVPT2 calculation, one must obtain a reference wavefunction with the RASSCF or DMRGSCF program and perform an integral transformation with the MOTRA program.

Currently, the implementation supports **only** QCMaquis DMRG reference wavefunctions (support for CASSCF reference wavefunctions will be added in the near future). It is nevertheless possible to run NEVPT2 with a CASSCF reference wavefunction by performing a DMRG-CI calculation with a sufficiently large \\(m\\) value using the CASSCF converged orbitals. For example, an \\(m\\) value of 2000 recovers the exact CASCI energy up to \\(5\times{}10^{-8}\\) a.u. for active spaces of up to 14 orbitals.

Below we show an example workflow of a NEVPT2 calculation. The input below is a calculation of the lowest singlet state of methylene with an active space of 6 electrons in 6 orbitals:
    
    
    &GATEWAY
      coord
      3
      CH2 Triplet coordinates in angstrom
      C      0.000000  0.000000  0.000000
      H      0.000000  0.000000  1.077500
      H      0.784304  0.000000 -0.738832
      basis=cc-pVTZ
      Group=Nosym
      RICD
      CDTH=1.0E-7
    &SEWARD
    &DMRGSCF
      ActiveSpaceOptimizer=QCMaquis
      DMRGSettings
        max_bond_dimension=128
        nsweeps=5
      EndDMRGSettings
      OOptimizationSettings
        Spin=3
        Inactive=1
        Ras2=6
        NActEl=6,0,0
        NEVPT2Prep
      EndOOptimizationSettings
    &MOTRA
      Frozen=0
      CTOnly
      Kpq
      HDF5
    &NEVPT2
    

First, one performs a DMRG-SCF calculation with the keyword NEVPT2Prep, which enables the evaluation of the four-particle reduced density matrices (4-RDMs) (and, in case of multiple states, also transition three-particle density matrices (t-3RDMs)) required by NEVPT2.

Second, one must perform an integral transformation with the MOTRA module. If no Cholesky decomposition or RICD is used in the calculation, the only mandatory keyword is HDF5, which enables the write-out of the transformed integrals in the HDF5 format required by the NEVPT2 module. If Cholesky decomposition is used, one additionally needs to add the keys CTOnly and Kpq. Cholesky decomposition is strongly recommended, as the integral transformation without Cholesky is several times slower and not supported in parallel.

Note that running with the Cholesky decomposed integrals currently does not support symmetry, and the support for frozen orbitals in MOTRA with Cholesky is untested, hence also the keyword Frozen=0 is recommended.

Finally, one calls the NEVPT2 module with &NEVPT2. It has no mandatory options, but options described in the Users Guide can be specified.

## 3.3.9.2. Distributed RDM evaluation¶

The computational cost of the RDM evaluation grows as \\(N^8\\) with the number of active orbitals, therefore the RDM evaluation for active spaces larger than 11-12 orbitals becomes prohibitively expensive. Therefore NEVPT2 distribution provides an (experimental) python utility jobmanager.py for distributed massively parallel 4-RDM calculations. With distributed 4-RDM calculations, active spaces of up to 22 orbitals can be employed in DMRG-NEVPT2 calculations without any approximation to the 4-RDM.

jobmanager.py splits the evaluation of the 4-RDM \\(G_{ijklmnop}\\) into four-index subblocks with indices \\(i,j,k,l\\). Due to permutational symmetry and the properties of the creation and annihilation operators, \\(i \ge j \ge k \ge l\\) and no more than two indexes are equal (pairwise equality \\(i=j\\) and \\(k=l\\) is allowed). The script prepares input files and, if requested, submits a separate job for each subblock, and merges the subblocks into the full matrix once the jobs are finished. The script is expected to be run on a head node of a distrubuted computing system with a batch system: [LSF](<https://www.ibm.com/support/knowledgecenter/en/SSETD4/product_welcome_platform_lsf.html>) has been tested, but any batch system which supports the [DRMAA](<http://www.drmaa.org/>) library, such as Slurm or PBS, should work. If no support for DRMAA is found, the script still may be used to prepare the input files for each subblock, which then may be submitted manually. Note that the DMRG-SCF/NEVPT2 calculation need not be performed on the same system as the 4-RDM evaluation.

### 3.3.9.2.1. How to run NEVPT2 calculations with distributed 4-RDM evaluation¶

Prerequisites:

  * Python \\(\ge\\) 2.7.9 (3.x is also supported)

  * (optional) DRMAA library compatible with your batch submission system, (e.g. [LSF-DRMAA](<https://github.com/IBMSpectrumComputing/lsf-drmaa>))

  * [Python DRMAA](<https://github.com/drmaa-python/drmaa-python>)

  * (optional) GNU Parallel




If your system administrator has not set up DRMAA and Python DRMAA, you might need to download and install these libraries yourself. After the installation, the environment variable DRMAA_LIBRARY_PATH must be set to the path to libdrmaa.so and, if Python does not find the DRMAA Python binding, also PYTHONPATH to the path of the Python DRMAA library.

Workflow:

  * Run DMRGSCF and MOTRA calculations as shown above, but **omit** calling the NEVPT2 program. The NEVPT2Prep keyword in the DMRGSCF section creates QCMaquis input templates and the MPS checkpoint files required for a later 4-RDM and/or t-3RDM evaluation.

  * Copy the $MOLCAS/Tools/distributed-4rdm/prepare_rdm_template.sh script to the OpenMolcas scratch directory and run it. The script will create subdirectories named 4rdm-scratch.<state> for each state. If you wish to perform the 4-RDM evaluation on a different machine (e.g. a cluster), copy the subdirectory for each state to that machine. If you do not wish to evaluate the 4-RDM for all states, pass the list of desired states as parameters to the prepare_rdm_template.sh script. For example, ./prepare_rdm_template.sh 0 1 2 will create the scratch directories for states from 0 to 2 (note that QCMaquis starts counting states with 0).

  * **If you have installed and working DRMAA setup:** For each state, change to the 4rdm-scratch.<state> subdirectory and run
        
        nohup jobmanager.py &
        

(Login to the machine where you evaluate the 4-RDM before if you wish to run the evaluation on a different machine.) This will create a subdirectory for each batch job (corresponding for each four-index 4-RDM subblock) and submit the jobs. The script will stay in the background until all the jobs have completed. The script also accepts the following job-specific options:

    * -t HH:MM:SS: set the maximum walltime per job. Default is 24h.

    * -n NCPU: run each job in an SMP parallelised fashion and set the number of CPU cores per job. Default is 1 core. For large active spaces, it is recommended to use several cores (e.g. 16 or 24, or as much as is available on a single node on your cluster).

  * If you **do not** have DRMAA installed and working, run the jobmanager.py script with the -n option:
        
        jobmanager.py -n
        

This will create subfolders for each 4-RDM block and prepare all the necessary input scripts, but will not submit them to the batch system. Now you may manually submit the scripts from the subfolders parts/part-*.

  * If you ran the distributed 4-RDM calculation on a different machine, copy the 4rdm-scratch.<state> back to OpenMolcas $WorkDir.

  * Create an input file with the input to the NEVPT2 program and run it. The keyword DISTributedRDM followed by the path to 4rdm-scratch.<state> folders (in our case, $WorkDir) is **mandatory**.




### 3.3.9.2.2. Troubleshooting¶

The jobmanager.py script is experimental, and also batch jobs in queuing systems are prone to crash, therefore we provide a mechanism to identify and restart the crashed batch jobs. The NEVPT2 program will check if the 4-RDM calculation has been finished correctly. If some 4-RDM values are missing, the NEVPT2 program will stop with an error. In this case several options are available:

  * **If DRMAA has been used:** if the jobmanager.py finishes without errors, it will produce two files, successlist and faillist with the list of successful and failed batch jobs, respectively. In this case, the failed jobs may be restarted using the restart mode of jobmanager.py, which is invoked with
        
        nohup jobmanager.py -r successlist faillist &
        

If the jobmanager.py finishes with an error, the successlist and faillist will be either nonexistent or empty. Note that this does NOT necessarily mean that the jobs have failed: in our tests, certain configurations of the queuing system may lead to the crash of the jobmanager.py script after the successful completion of the jobs.

  * **If DRMAA has not been used and the script was run with the -n switch** : in this case the user is advised to check manually the subfolders for each 4-RDM subblock for the existence of $Project.results_state.X.h5 files. The files should exist and the command
        
        h5dump $Project.results_$state.X.h5 | grep fourpt
        

should not yield an empty result – otherwise the corresponding calculation should be rerun.

  * Finally, if NEVPT2 is started with the DISTributedRDM keyword, it will check the number of evaluated 4-RDM elements. If the number of evaluated elements is different from its expected value, the program will exit with an error.




### 3.3.9.2.3. Transition 3-RDM distributed calculations¶

jobmanager.py also supports distributed calculations of t-3RDMs (required for multi-state QD-NEVPT2). The split evaluation is similar to that of the 4-RDMs, and the workflow above can be followed with the following differences:

  * The t-3RDM evaluation requires two states instead of one. Run the prepare_rdm_template.sh script with the -3 parameter.

  * Launch the jobmanager.py script with the -3 parameter.




### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tut.html>)
    * [3.1. Quickstart Guide for Molcas](<nutshell.html>)
    * [3.2. Problem Based Tutorials](<pbtutorials.html>)
    * [3.3. Program Based Tutorials](<tutorials.html>)
      * [3.3.1. Molcas Flowchart](<flowchart.html>)
      * [3.3.2. Environment and EMIL Commands](<tut_emil.html>)
      * [3.3.3. GATEWAY — Definition of geometry, basis sets, and symmetry](<tut_gateway.html>)
      * [3.3.4. SEWARD — An Integral Generation Program](<tut_seward.html>)
      * [3.3.5. SCF — A Self-Consistent Field program and Kohn–Sham DFT](<tut_scf.html>)
      * [3.3.6. MBPT2 — A Second-Order Many-Body PT RHF Program](<tut_mbpt2.html>)
      * [3.3.7. RASSCF — A Multi Configurational Self-Consistent Field Program](<tut_rasscf.html>)
      * [3.3.8. CASPT2 — A Many Body Perturbation Program](<tut_caspt2.html>)
      * 3.3.9. NEVPT2 — \\(n\\)-Electron Valence State Second-Order Perturbation Theory
      * [3.3.10. RASSI — A RAS State Interaction Program](<tut_rassi.html>)
      * [3.3.11. CASVB — A non-orthogonal MCSCF program](<tut_casvb.html>)
      * [3.3.12. MOTRA — An Integral Transformation Program](<tut_motra.html>)
      * [3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program](<tut_guga.html>)
      * [3.3.14. MRCI — A Configuration Interaction Program](<tut_mrci.html>)
      * [3.3.15. CPF — A Coupled-Pair Functional Program](<tut_cpf.html>)
      * [3.3.16. CCSDT — A Set of Coupled-Cluster Programs](<tut_ccsdt.html>)
      * [3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization](<tut_optimiza.html>)
      * [3.3.18. MCKINLEY — A Program for Integral Second Derivatives](<tut_mckinley.html>)
      * [3.3.19. MCLR — A Program for Linear Response Calculations](<tut_mclr.html>)
      * [3.3.20. GENANO — A Program to Generate ANO Basis Sets](<tut_genano.html>)
      * [3.3.21. FFPT — A Finite Field Perturbation Program](<tut_ffpt.html>)
      * [3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules](<tut_vibrot.html>)
      * [3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program](<tut_single_aniso.html>)
      * [3.3.24. POLY_ANISO — Semi-_ab initio_ Electronic Structure and Magnetism of Polynuclear Complexes Program](<tut_poly_aniso.html>)
      * [3.3.25. GRID_IT — A Program for Orbital Visualization](<tut_grid_it.html>)
      * [3.3.26. Writing MOLDEN input](<tut_molden.html>)
      * [3.3.27. Tools for selection of the active space](<tut_expbas.html>)
      * [3.3.28. Most frequent error messages found in Molcas](<tut_errors.html>)
      * [3.3.29. Some practical hints](<tut_hints.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut_caspt2.html> "3.3.8. CASPT2 — A Many Body Perturbation Program") | [next](<tut_rassi.html> "3.3.10. RASSI — A RAS State Interaction Program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_nevpt2.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
