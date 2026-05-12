<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/environment.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

4.1.2. Commands and environment variables

[previous](<env-overview.html> "4.1.1. Overview") | [next](<emil.html> "4.1.3. General input structure. EMIL commands") | [index](<../genindex.html> "General Index")

# 4.1.2. Commands and environment variables¶

This section will describe the usage of Molcas in an UNIX environment.

Production jobs using Molcas in an UNIX environment can be performed as batch jobs. This requires the creation of a shell script that contains a few simple commands. Further you need to create input for each program module that you intend to use. This section describes the necessary steps you have to take in order to make a successful job using Molcas. Input examples for a typical Molcas run can be found in doc/samples/problem_based_tutorials/ directory. Also you can use some input examples in Test/input subdirectory.

## 4.1.2.1. Commands¶

There is a command supplied with the Molcas package, named molcas, that the user issue to perform a given task. A sequence of such commands will perform the calculation requested by the user.

molcas
    

This command tells which Molcas installation will be used, and gives some help about usage of Molcas command

molcas input-file
    

This command executes a command in the Molcas system.

molcas help PRGM
    

This command gives the list of available keywords for program PRGM.

molcas help PRGM KEYWORD
    

This command gives description of a KEYWORD.

molcas help ENVIRONMENT
    

This command gives a list of Molcas specific environment variables.

molcas help BASIS ELEMENT
    

This command gives a list of basis sets available for an ELEMENT.

The following is an example of running Molcas by using a single input file:
    
    
    molcas $Project.input
    

An alternative way of running Molcas as a sequence of separate calls:
    
    
    molcas $Project.seward.input    # Execute seward
    molcas $Project.scf.input       # Execute scf
    

By default, the output will go directly to the screen. It can be redirected by using flag -f, e.g. molcas -f water.inp will store the output in water.log and water.err files.

The default behavior of Molcas execution can be altered by setting environment variables.

## 4.1.2.2. Project name and working directory¶

When running a project, Molcas uses the variable Project giving a project name, and a scratch directory defined by the variable WorkDir. This serves the purpose of maintaining structure of the files and facilitating automatic file mapping.

There are several ways to set up these variables. By default, the name of the Project constructed from the name of the input file, by removing the last suffix, e.g. for example for an input name Water.SCF.input the Project name will be Water.SCF. Alternatively, user can set environment variable Project, or MOLCAS_PROJECT.

Scratch directory can be set by environment variable MOLCAS_WORKDIR. If it is set to value `PWD`, current directory will be used. Otherwise, it can be set to a directory name. In this case scratch area will be located in a subdirectory $MOLCAS_WORKDIR/$Project. It is also possible to overwrite the value of scratch area, by setting environment variable WorkDir.

  * Project=…; export Project

  * WorkDir=…; export WorkDir




Molcas modules communicates between each other via files, located in the WorkDir. The description of internal filenames and file mapping can be found at Appendix.

## 4.1.2.3. Input¶

When you have decided which program modules you need to use to perform your calculation, you need to construct input for each of these. There is no particular structure enforced on the input files, but it is recommended that you follow:

  * $Project.<prgm-name>.input




which is the name of the input files assumed in the sample shell script.

## 4.1.2.4. Preparing a job¶

When you prepare a job for batch processing, you have to create a shell script. It is recommended that you use the sample shell script supplied with Molcas as a starting point when building your own shell script. The following steps are taken in the shell script:

  1. Define and export the Molcas variables

     * Project (or use MOLCAS_PROJECT)

     * WorkDir (or MOLCAS_WORKDIR)

  2. Issue a sequence of Molcas commands.

  3. Remove the scratch directory and all files in it.




The following is an example of a shell script.
    
    
    Project=HF; export Project                               # Define the project id
    WorkDir=/temp/$LOGNAME/$Project.$RANDOM; export WorkDir  # Define scratch directory
    molcas $Project.input                                    # Run molcas with input file, which
                                                             # contains inputs for several modules
    rm -r $WorkDir                                           # Clean up
    

The file $ThisDir/$Project.input contains the ordered sequence of Molcas inputs and the EMIL interpreter will call the appropriate programs. See [Section 4.1.3](<emil.html#ug-sec-emil>) for an explanation of the additional tools available in the EMIL interpreter.

The following is an example of a shell script to be submitted for batch execution.
    
    
    Project=HF; export Project                               # Define the project id
    WorkDir=/temp/$LOGNAME/$Project.$RANDOM; export WorkDir  # Define scratch directory
    molcas $Project.seward.input                             # Execute seward
    molcas $Project.scf.input                                # Execute scf
    rm -r $WorkDir                                           # Clean up
    

An alternative way to control the usage of the WorkDir is to use flags in molcas command:

-new
    

clean WorkDir before the usage

-clean
    

clean WorkDir after the usage

Note, that if you configured your working environment by using setuprc script, the only command you have to place into the shell script is:
    
    
    molcas $Project.input
    

## 4.1.2.5. System variables¶

Molcas contains a set of system variables that the user can set to modify the default behaviour of Molcas. Two of them (Project and WorkDir) must be set in order to make Molcas work at all. There are defaults for these but you are advised not to use the defaults.

There are several ways of using Molcas environment variables:

  * These variables can be exported in your shell script
        
        export MOLCAS_MEM=512
        molcas input
        

  * These variables can be included into Molcas input:
        
        * begin of the input file
        >>> export MOLCAS_MEM=512
        
          . . .
        

  * variables can be included directly into molcas command in the form:
        
        molcas MOLCAS_MEM=512 input
        




The simplest way to set up default environment for Molcas is to use script setuprc, which can be run as command molcas setuprc. This interactive script creates a resource file molcasrc, located either in $MOLCAS or $HOME/.Molcas directory. The priority of these settings is: user defined settings (e.g. in molcas command), user resource file, Molcas resource file.

Two flags in Molcas command are related to resource files:

-env
    

Display current Molcas environment e.g. molcas -env input will print information about environment variables, used during execution of the input file.

-ign
    

Ignore resource files e.g. molcas -ign input will process input file without settings, which are stored in $MOLCAS/molcasrc and in $HOME/molcasrc files.

The most important environment variables, used in Molcas:

Project
    

This variable can be set in order to overwrite the default name of the project you are running. The default (and recommended) value of the project name is the name of the input file (without the file extension).

WorkDir
    

This variable can be used to specify directly the directory where all files that Molcas creates are placed. See MOLCAS_WORKDIR for more options.

CurrDir
    

This variable corresponds to the location of the input, and it is used as a default location for all output files, generated by Molcas modules.

MOLCAS
    

This variable indicates the location of Molcas. The default version of Molcas to be used is specified at file .Molcas/molcas, located at user HOME directory.

MOLCAS_NPROCS
    

This variable should be used to run Molcas code in parallel. It defines the number of computational units (cores or nodes) which will be used.

MOLCAS_MEM
    

This environment variable controls the size (soft limit) of the work array utilized in the programs that offer dynamic memory. It is specified in Megabytes, i.e.

MOLCAS_MEM=256; export MOLCAS_MEM

will assign 256MB for the working arrays. It is also possible to use Gb (Tb) to specify memory in Gb or Tb.

  * MOLCAS_MEM is undefined — The default amount of memory (1024MB), will be allocated for the work arrays.

  * MOLCAS_MEM is defined but nonzero — This amount of memory will be allocated.




See also MOLCAS_MAXMEM.

The complete list of Molcas-related environment variables:

MOLCAS_COLOR
    

By default molcas uses markup characters in the output. To overwrite, set the key to NO.

MOLCAS_NPROCS
    

See above

MOLCAS_THREADS
    

This variable should be used to run Molcas code with multithreaded capabilities. It defines the number of threads that will be used for the multithreaded portions of the code (mostly linear algrebra library calls).

MOLCAS_DEBUGGER
    

This variable can be set to the name of debugger (or another code) which will be used on top of molcas executables. The option is useful for tracing an error in the code

MOLCAS_DISK
    

The value of this variable is used to split large files into a set of smaller datasets, as many as are needed (max. 20 subsets). It is specified in Megabytes, for instance, MOLCAS_DISK=1000; export MOLCAS_DISK, and the following rules apply:

  * MOLCAS_DISK is undefined — The program modules will ignore this option and the file size limit will be defined by your hardware (2 GBytes for 32-bit machines).

  * MOLCAS_DISK=0 (zero) — The programs will assume a file size limit of 2 GBytes (200GBytes on 64-bit machines).

  * MOLCAS_DISK is defined but nonzero — The files will be limited to this value (approximately) in size.



MOLCAS_ECHO_INPUT
    

An environment variable to control echoing of the input. To suppress print level, set MOLCAS_ECHO_INPUT to `NO`.

MOLCAS_FIM
    

Activates the Files In Memory I/O layer. See [Section 2.6.1.5](<../installation.guide/maintain.html#mt-sec-fim>) for more details. _Note that this setting is available only in_ Molcas _compiled without Global Arrays._

Warning

This feature is not available in OpenMolcas.

MOLCAS_INPORB_VERSION
    

Selects the version used for writing orbital files ($Project.ScfOrb, $Project.RasOrb, etc.). The value should be a version number such as `1.0` or `2.2`. If the version is not known, the default (usually latest) version will be used.

MOLCAS_KEEP_WORKDIR
    

If set to NO Molcas will remove scratch area after a calculation. This setting can be overwritten by running molcas with flag -clean. Note that this does not work in a parallel environment.

MOLCAS_LICENSE
    

An environment which specifies the directory with Molcas license file license.dat. The default value of this variable is directory .Molcas/ in user home directory.

MOLCAS_LINK
    

An environment variable to control information about linking of files. By default (MOLCAS_LINK is not set) only essential information about linking will be printed. To increase/decrease the print level, set MOLCAS_LINK to `Yes`/`No`.

MOLCAS_MAXITER
    

An environment variable to control maximum number of iterations in DO WHILE loop.

MOLCAS_MAXMEM
    

An environment variable to set up a hard limit for allocated memory (in Mb). If is not specified, then it takes value of MOLCAS_MEM. Otherwise, the (MOLCAS_MAXMEM-MOLCAS_MEM) amount of RAM will be primarily used for keeping files in memory (FiM), or allocating Distributed Global Arrays. _Note that this setting is available only in |molcas| compiled without Global Arrays._

MOLCAS_MEM
    

See above.

MOLCAS_MOLDEN
    

If MOLCAS_MOLDEN set to `ON` a Molden style input file will be generated regardless of the number of orbitals.

MOLCAS_NEW_DEFAULTS
    

If set to `YES` (case insensitive), some new default values will be activated:

  * RICD will be enabled by default in GATEWAY, it can be disabled with NOCD.

  * The default IPEA shift in CASPT2 is set to 0.0, other values can be specified normally with the IPEA keyword.



MOLCAS_NEW_WORKDIR
    

If set to YES Molcas will never reuse files in scratch area. This setting can be overwritten by running molcas with flag -old: molcas -old input. Note that this does not work in a parallel environment.

MOLCAS_OUTPUT
    

This variable can alter the default directory for extra output files, such as orbitals files, molden files, etc. If set, Molcas will save output files to the specified directory. The directory name can be set in the form of absolute PATH, or relative PATH (related to the submit directory). A special value `WORKDIR` will keep all output files in WorkDir. A special value `NAME` will create a subdirectory with a name of Project. If the variable is not set, all output files will be copied or moved to the current directory. Default value can be forced by MOLCAS_OUTPUT=PWD.

MOLCAS_PRINT
    

MOLCAS_PRINT variable controls the level of output. The value could be numerical or mnemonic: SILENT (0), TERSE (1), NORMAL (2), VERBOSE (3), DEBUG (4) and INSANE (5).

MOLCAS_PROJECT
    

If set to value NAME, Molcas will use the prefix of the input file as a project name. Otherwise, it set a project name for the calculation. If set to the value NAMEPID, the Project name still will be constructed from the name of input file, however, the name of scratch area will be random.

MOLCAS_PROPERTIES
    

If MOLCAS_PROPERTIES is set to `LONG` properties with the individual MO contributions will be listed.

MOLCAS_RANDOM_SEED
    

Set to an integer to provide a fixed seed for operations that use a random number.

MOLCAS_REDUCE_PRT
    

If set to NO, print level in DO WHILE loop is not reduced.

MOLCAS_REDUCE_NG_PRT
    

If set to NO, print level in NUMERICAL_GRADIENT loop is not reduced.

MOLCAS_SAVE
    

This variable can alter the default filenames for output files. If not set (default), all files will overwrite old files. If set to `INCR` all output files will get an incremental filenames. If set to `ORIG` — an existent file will be copied with an extension `.orig`

MOLCAS_TIME
    

If set, switch on timing information for each module

MOLCAS_TIMELIM
    

Set up a timelimit for each module (in seconds). By default, the maximum execution time is set to unlimited. _Note that this setting is available only in MOLCAS compiled without Global Arrays._

MOLCAS_TRAP
    

If set to OFF Molcas modules will continue to be executed, even if a non-zero return code was produced.

MOLCAS_VALIDATE
    

If set to YES, the input for each module will be validated against the documented syntax, and the calculation will stop if it does not pass. If set to CHECK, the input will be validated, but the calculation will continue, although the program itself may stop. If set to FIRST, the whole input file will be validated prior to the calculation.

MOLCAS_WORKDIR
    

A parent directory for all scratch areas. It can be set to an absolute PATH (recommended), to a relative PATH, or to a special value PWD (to use current directory for scratch files)

User can customize his installation by adding MOLCAS environment variable into molcasrc file.

Another way of customizing Molcas is to use prologue and epilogue scripts. If user created a file prologue in $HOME/.Molcas directory it will be executed (as ./prologue) before Molcas calculation starts. epilogue in $HOME/.Molcas directory will be executed at the end of calculation. Files module.prologue and module.epilogue contains commands executing before and after each executable molcas module. These files may use internal Molcas variables, such as $Project, $WorkDir, $MOLCAS_MEM, etc. Note that prologue/epilogue scripts should be executable. For debug purposes, the location of prologue and epilogue files can be set by $MOLCAS_LOGUE_DIR variable.

Example:

prologue:
    
    
    echo Calculation of $Project input will start at `date`
    

module.prologue:
    
    
    echo Running module $MOLCAS_CURRENT_PROGRAM at $WorkDir
    

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<ug.html>)
    * [4.1. The Molcas environment](<env-main.html>)
      * [4.1.1. Overview](<env-overview.html>)
      * 4.1.2. Commands and environment variables
      * [4.1.3. General input structure. EMIL commands](<emil.html>)
    * [4.2. Programs](<programs.html>)
    * [4.3. GUI](<tools.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<env-overview.html> "4.1.1. Overview") | [next](<emil.html> "4.1.3. General input structure. EMIL commands") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/users.guide/environment.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
