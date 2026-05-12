<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/emil.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

4.1.3. General input structure. EMIL commands

[previous](<environment.html> "4.1.2. Commands and environment variables") | [next](<programs.html> "4.2. Programs") | [index](<../genindex.html> "General Index")

# 4.1.3. General input structure. EMIL commands¶

This is a general guide to the input structure of the programs in the Molcas program system. All programs conform to the same conventions except where explicitly stated otherwise.

The programs are driven by keywords, which are either used without further information, or followed by additional specifications on the line(s) following the keyword, and is normally numeric in nature. _All numerical inputs are read in free format, note that in general_ Molcas _will not be able to process lines longer than 120 characters._ The keywords can be given in mixed case (both upper and lower case are allowed). In the input stream you can insert comment lines anywhere, except between a keyword and the following additional specifications, with a comment line identified by an asterisk (*) in the first position on the line.

Most codes look at the first 4 characters of the keyword and ignores the rest. The entries in the lists of keywords below follow the standard that the significant characters are in upper case and larger than the nonsignificant characters. This do not imply that the keywords have to be typed in upper case; they can be typed freely in mixed case.

All inputs begin with a name of the program followed by the keywords:
    
    
    &PROGRAM
    * here follows the keywords
    

where PROGRAM is the name of the Molcas module. The input listing is finished when a new program name, precede by the symbol &, is found (or the end of file or an EMIL command).

The following is an example of a list of keywords common to most of the programs:

TITLe
    

This keyword starts the reading of title line. The following line is treated as title line.

The programs only decode the first four characters of a keyword (except otherwise specifically indicated). For clarity it is however recommended to write the full keyword name. The keywords can be typed freely in upper, lower or mixed case.

An example for an input file used to run the SCF program follows:
    
    
    &SCF
    Title
     Water molecule. Experimental equilibrium geometry
    * The symmetries are: a1, b2, b1 and a2.
    Occupied
    3 1 1 0
    * The ivo keyword prepares virtual orbitals for MCSCF.
    Ivo
    

Interpretation of Molcas input is performed by molcas.exe. The internal language used by molcas.exe is EMIL (Extended Molcas Input Language). It includes two different types of input commands:

  * Sections with Molcas input.

  * EMIL commands (a line started with > character)




## 4.1.3.1. Molcas input¶

EMIL allows to write Molcas input in a more compact way: user can omit &END, as well as a compulsory (in previous versions of Molcas) keyword End of input. As soon as a new module (or EMIL command) is requested in a user input, the input for the module is terminated.

Also, it is possible to separate lines by `;` sign, or by `=` sign (to create a pair `keyword=value`). In some rare occasions signs `;` and `=` are used in the input for a Molcas module. In order to keep these symbols unchanged, user can mark a part of an input, containing these symbols, by EMIL commands `>> verbatim` and `>> end verbatim`.

It means that the input:
    
    
    &SCF &END
    CHARGE
     1
    End of input
    &ALASKA &END
    End of input
    &SLAPAF &END
    End of input
    

could be written as:
    
    
    &SCF; CHARGE=1
    &ALASKA; &SLAPAF
    

User can comment parts of input, by using `*` at the beginning of line, or use C-style comments (`/* ... */`) to comment several lines.

In a rare occasion user might want to execute a UNIX command from the input. It is important to understand that not all UNIX commands can be understood and interpreted by EMIL. Also, EMIL should know where to execute a command – only at the master node, or for all parallel tasks. In the past, EMIL supports the usage of commands started from an exclamation mark, or with command UNIX. To avoid confusions, the serial execution of a command is now related to SHELL, and the parallel execution to EXEC.

## 4.1.3.2. EMIL commands¶

EMIL commands can be written in a short form:
    
    
    > KEY [VALUE]
    

or in a nice form:
    
    
    >>>>>>>>>>  KEY  [VALUE]  <<<<<<<<<
    

EMIL commands are not case sensitive, but the variables used in commands must be written in upper case. Also, it is important to place spaces in between elements (words) in the commands.

Here is a list of EMIL commands:

>> EXPORT A=B
    

a command to set environment variable A to value B

>> EXIT
    

a command to terminate execution. An optional value for this command is the return code (default value is 0)

>> INCLUDE file
    

a command to include a file into the input A compulsory value for this command is the filename.

>> FILE file
    

A compulsory value for this command is the filename. A command to inline a file in the input file. The file will be extracted into WorkDir before the start of the calculation. The end of file should be marked as EOF command. Note that the file is only created in the master process WorkDir, if the slaves need access to it, you’ll need to use the COPY command (see below). All files specified with FILE are created at the beginning of the calculation, regardless of their placement in the input.

>> EOF
    

A command to close inlined file.

>> SHELL
    

a command to execute a unix command in serial.

>> EXEC
    

a command to execute a unix command in parallel.

>> LINK
    

a command to make a link between two files, located in WorkDir. The command is similar to `!ln -s FILE1 FILE2` but in parallel environment it is executed in all WorkDirs. The command assumes that FILE1 does exist, and FILE2 does not at the moment. >>LINK -FORCE allows to link a file which does not exist. User should avoid the usage of LINK commands in the input.

>> COPY
    

a command to make a copy. The command is similar to `!cp -f /path/to/FILE1 FILE2` but can be used also in a parallel environment, in which case it will take the source file and distribute to the work directories of all processes. The destination must be located in the work directory. Note that EMIL command does not allow to use masks in the command. If FILE1 does not exist, the command returns an error code.

>> CLONE
    

a command to make a clone copy of a file, doing a local copy on all slaves if parallel. It is mostly used internally, e.g. to distribute an input file to all WorkDirs.

>> COLLECT
    

A command to copy one file to another, collecting files on slaves and put them on the master if parallel. It is mostly used internally, e.g. to collect output files.

>> SAVE
    

A command to copy one file to another, only on the master if parallel

>> RM
    

a command to delete a file. The command is similar to `!rm FILE` but can be used also in parallel environment. Note that EMIL command does not allow to use masks in the command. An attempt to remove non existent file leads to an error. It is possible to use -FORCE flag to allow deleting of non-existent file.

>> EVAL A=B
    

evaluate a numerical value

Keywords to organize loops in input, and execute modules conditionally:

>> DO WHILE
    

a command to start a loop. The loop should be terminated by SLAPAF or LOOP module, followed by ENDDO command

>> DO GEO
    

a command to start a special loop for geometry optimization with constrained internal coordinates. The loop should be terminated by ENDDO command. (See documentation for GEO for more details.)

>> FOREACH A in (B, C, D)
    

a command to loop when the value of A is in the comma or space separated list. The list also can be written in the format `From .. To`. Note that variable in the loop must be uppercased.

>> ENDDO
    

a command to finish the loop. If last module (before ENDDO command) returns 1 — the loop will be executed again (if number of iterations is less than MAXITER). If the return code is equal to 0 the loop will be terminated.

>> IF ( ITER = N )
    

a command to make conditional execution of modules/commands on iteration N (N possibly could be a space separated list)

>> IF ( ITER NE N )
    

a command to skip execution of modules/commands on iteration N

>> IF ( ITER != N )
    

same as above

>> IF ( $VAR = N )
    

a command to make conditional execution if $VAR value equals to N (if statement terminated by ENDIF command)

>> IF ( $VAR = N ) GOTO JUMP
    

a command to make conditional goto to a label JUMP

>> IF ( -FILE file )
    

test for existence of a file

>> LABEL JUMP
    

a command to define a label. Note! Only forward jumps are allowed.

>> ENDIF
    

terminate IF block. Note nested if’s are not allowed.

EMIL interpreter automatically stops calculation if a module returns a returncode higher than 0 or 1. To force the interpretor to continue calculation even if a returncode equal to 16 (which is a return code for non-convergent calculation) one should set environment variable MOLCAS_TRAP=`OFF`.

SLAPAF returns a special return code in the case of converged (non converged) geometry. So, to organize a structure calculation one should place the call to SLAPAF as a last statement of loop block. The summary of geometry optimization convergence located in a file $Project.structure. The programs following a geometry optimization will automatically assume the optimized geometry and wave function. Any new SEWARD calculation after an optimization (minimum or transition state) will disregard the input coordinates and will take the geometry optimized at previous step.

It is also possible to use a special dummy program LOOP to organize infinite loops, or loops terminated by the counter (set by MOLCAS_MAXITER)

Keyword SET is obsolete and should be changed to EXPORT.

Verbatim input.

If an input for a module must contain special symbols, such as `;` or `=`, user can mark a corresponding part of the input by EMIL command VERBATIM

>> VERBATIM <<
    

start verbatim input

>> END VERBATIM <<
    

finish verbatim input

Below are different input examples.

The first example shows the procedure to perform first a CASSCF geometry optimization of the water molecule, then a numerical hessian calculation on the optimized geometry, and later to make a CASPT2 calculation on the optimized geometry and wave function. Observe that the position of the SLAPAF inputs controls the data required for the optimizations.
    
    
    *
    *    Start Structure calculation
    *
    >>EXPORT MOLCAS_MAXITER=50
     &GATEWAY
    coord
    $MOLCAS/Coord/Water.xyz
    BASIS = ANO-S
    
    >>>>>>>>>>>>> Do while <<<<<<<<<<<<
     &SEWARD
    >>>>>>>> IF ( ITER = 1 ) <<<<<<<<<<
     &SCF
    >>>>>>> ENDIF <<<<<<<<<<<<<<<<<<<<
    
     &RASSCF
    Title
     H2O ANO(321/21).
    Nactel   = 6  0  0
    Spin     = 1
    Inactive = 1  0  0  0
    Ras2     = 3  1  0  2
    
     &ALASKA; &SLAPAF
    
    >>>>>>>>>>>>> ENDDO <<<<<<<<<<<<<<
    
     &CASPT2
    Maxit = 20
    Lroot = 1
     &GRID_IT
    

Another example demonstrate a possibility to use loops. SCF module will be called twice — first time with BLYP functional, second time with B3LYP functional.
    
    
    *------------------------------------------------------
     &GATEWAY
    coord
    $MOLCAS/Coord/C2H6.xyz
    basis
    ANO-S-VDZ
    group
    y xz
    *------------------------------------------------------
     &SEWARD
    Title
    Ethane DFT test job
    *------------------------------------------------------
    >>foreach DFT in (BLYP, B3LYP )
     &SCF ; KSDFT = $DFT
    >>enddo
    *------------------------------------------------------
    

The next examples calculates HF energy for the several structures:
    
    
    * modify coordinates in place
    >>foreach DIST in (1.0, 2.0, 20.0)
     &GATEWAY
    Coord
    2
    hydrogen molecule
    H 0 0 0
    H $DIST 0 0
    BASIS= ANO-S-MB
    GROUP= C1
     &SEWARD
     &SCF
    UHF
    SCRAMBLE=0.3
    >>enddo
    
    * incremental change of coordinates
    >>export DIST=1.0
    >>foreach L in ( 1 .. 3 )
    >>eval DIST=$DIST+0.1
     &GATEWAY
    Coord
    2
    hydrogen molecule
    H 0 0 0
    H $DIST 0 0
    BASIS= ANO-S-MB
    GROUP= C1
     &SEWARD
     &SCF
    >>enddo
    
    * different coordinate files
    >> FILE H2001.xyz
    2
    
    H  0.300000000  0.000000000  0.000000000
    H -0.300000000  0.000000000  0.000000000
    
    >> FILE H2002.xyz
    2
    
    H  0.350000000  0.000000000  0.000000000
    H -0.350000000  0.000000000  0.000000000
    
    >> FILE H2003.xyz
    2
    
    H  0.400000000  0.000000000  0.000000000
    H -0.400000000  0.000000000  0.000000000
    
    >>foreach COO in ( 000, 001, 002)
     &GATEWAY
    Coord = H2$COO.xyz
    BASIS= ANO-S-MB
    GROUP= C1
     &SEWARD
     &SCF
    >>enddo
    

## 4.1.3.3. Use of shell parameters in input¶

The Molcas package allows the user to specify parts or variables in the the input file with shell variables, which subsequently are dynamically defined during execution time. **Note:** the shell variable names must be in upper case. Find below a simple example where a part of the \\(\ce{H2}\\) potential curve is computed. First, the script used to run the calculation:
    
    
    #! /bin/sh
    #
    Home=`pwd` ;                     export Home
    Project=H2 ;                     export Project
    WorkDir=/tmp/$Project ;          export WorkDir
    #
    # Create workdir and cd to it
    #
    rm -fr $WorkDir
    mkdir $WorkDir
    #
    # Loop over distances
    #
    for R in 0.5 0.6 0.7 0.8 0.9 1.0
    do
       export R
       molcas $Home/$Project.input > $Home/$Project-$R-log 2> $Home/$Project-$R-err
    done
    #
    # Cleanup WorkDir
    #
    rm -fr $WorkDir
    

In this sh shell script we have arranged the call to the Molcas package inside a loop over the various values of the distances. This value is held by the variable $R which is exported every iterations. Below is the input file used, note that the third cartesian coordinate is the variable $R.
    
    
    &SEWARD
    Symmetry
     x y z
    Basis set
    H.sto-3g....
    H   0.000   0.000   $R
    End of basis
    End of input
    
    &SCF
    

## 4.1.3.4. Customization of molcas input¶

Warning

This feature is not available in OpenMolcas.

EMIL interpretor supports templates (aliases) for a group of program calls or/and keywords. The definition of these templates can be located in file alias located at Molcas root directory, or at .Molcas/ directory. The definition should be written in the following format: `@name { sequence of EMIL commands }`. In order to use the alias, the input should contain `@name`.

For example, user can define
    
    
    @DFTgeometry {
    >> DO WHILE
     &SEWARD
     &SCF; KSDFT=B3LYP;
     &SLAPAF
    >>ENDDO
    }
    

and so, an input for geometry optimization can be written in the following form:
    
    
    &GATEWAY; Coord=Water.xyz; Basis = ANO-L-MB;
    @DFTgeometry
    

It is also possible to use parameters. In the alias file, possible parameters have names: `$1`, `$2`, etc. up to 5 parameters. In the user input an alias should be followed by parenthesis with comma separated list of values.

Modifying the previous example:
    
    
    @DFTgeometry {
    >> DO WHILE
     &SEWARD
     &SCF; CHARGE=$1; KSDFT=$2;
     &SLAPAF
    >>ENDDO
    }
    

Input file now looks like:
    
    
    @DFTgeometry(0,B3LYP)
    

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<ug.html>)
    * [4.1. The Molcas environment](<env-main.html>)
      * [4.1.1. Overview](<env-overview.html>)
      * [4.1.2. Commands and environment variables](<environment.html>)
      * 4.1.3. General input structure. EMIL commands
    * [4.2. Programs](<programs.html>)
    * [4.3. GUI](<tools.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<environment.html> "4.1.2. Commands and environment variables") | [next](<programs.html> "4.2. Programs") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/users.guide/emil.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
