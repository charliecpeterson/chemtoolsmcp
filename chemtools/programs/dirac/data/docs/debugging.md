orphan  

# Localizing bugs in DIRAC run using debugger

The DIRAC program system is continuously evolving. From time to time,
however, it gets - unintentionally - spoiled with various programming
bugs. Here we would like to share knowledge on how to localize
crashes/bugs in the code.

We believe that this short tutorial is useful not only for DIRAC
developers, but also for ordinary users, who likewise could send us
precise localization and specification of the bug.

## Introduction

Let us assume that we have serially compiled DIRAC code. Make sure that
DIRAC is compiled with "-g" flag for all Fortran, C and C++ compilers.
Anyway, this flag is the the default part in all combination of compiler
flags.

If you intend to run many debugging runs, please choose *--type=Debug*
in the setup flags. This means omitting optimization flags by the CMake
buildup system and speed up of (repeating) DIRAC compilation. Note that
some bugs (crashes) might "disappear" when the optimization is
completely turned off, because not all compilers have bug-free
optimization procedure.

Ensure that proper debugger, installed on your machine, is called with
the "pam" script. For example, for the GNU compilers set this is the
debugger "gdb". Some commercial compilers are providing their own
debuggers: Intel OneAPI includes the gdb-oneapi.

## Debugging with the "pam" script

Launching DIRAC with the "pam --debug ..." is suitable for one-shoot
debugging. Pam script calls the debugger, which afterwards shows up its
command line. Aftewards the user can insert breakpoints, stops into the
code and repeatedly run 'dirac.x' inside the debugger.

### Stack printing

There is the possibility to print out the stack content.

Any 'QUIT' crash of the code causes a SegFault termination you can
investigate with the debugger - you can 'visit' individual procedures
within the stack sequence and examine their variables.

For example, beeing the "gdb" debugger session:

    > run
    (Dirac runs and crashes somewhere)
    > backtrace
    (show the stack)
    > up {n}
    (go to the n-the previous function in the call stack; see help up)
    > print <VARIABLE>
    (whatever you want to inspect).

Oposite way can be achiewed using 'down' command.

## Debugging in the plain mode with the GNU/gdb debugger

We give a simple demonstrative session of an thorough and effective
debugging. Most recent version of the widely used the GNU/gdb debugger
is recommended.

Make a working directory with enough disk space. Provide a link of the
\$DIRAC/dirac.x executable file there. This is because you will repeat
modifications of DIRAC source files and subsequent compilation, so it is
easier to have linked executable rather than copying it.

Make sure that the 'DIRAC.INP', 'MOLECULE.INP', *gdb_demo* and other
necessary files (for example, the 'CHECKPOINT.h5' if you are restarting
SCF iterations) are in the working directory as well.

### Demonstration of GNU/gdb debugging commands

This *gdb_demo* file (create&copy from this web-page) contains some
commands (and macros) of the gdb debugger:

    ###  load the executable file with all symbols
    file dirac.x

    ### This settings is necessary for complex breakpoint conditions in fortran...
    #set language fortran
    set language c
    set language auto

    ### Write where I am...
    echo In directory:
    shell echo pwd
    echo \n Files in dir:
    shell ls -a

    ##################################################################
    ### Delete  DIRAC files in the scratch directory
    ### - needed for restarting and for further catching of bugs..
    ##################################################################
    shell /bin/rm -r DF*  AOPROPER BSSMAT
    echo \n After deletion remaining files in dir:
    shell ls -a

    ### clean the desk: delete all previous breakpoints
    d b

    ###################################
    # Example of macros
    ###################################
    define my_macro
    printf "in GMOTRA begin NZ=%d N2BBASXQ=%d SPINFR=%d\n",nz,n2bbasxq,spinfr
    end

    define my_macro1
    printf "in GMOTRA begin SPINFR2=%d\n",spinfr2
    end

    ### demo of a conditional breakpoint
    b gmotra_ if (nz==1 | (n2bbasxq==0 && spinfr==0))
    command
     my_macro
     my_macro1
     echo type continue or sipmly c...\n
    # c
    end

    ### example of the unconditional breakpoint
    b dirone.F:1399
    command
     if ( (nfsym.eq.1 && n2bbasxq.gt.10) | nz.eq.1 | spinfr.eq.0)
      printf "NFSYM=%d\n", NFSYM
      if (nfsym.eq.2)
        printf "NTMO(1)=%d NTMO(2)=%d\n",NTMO(1),NTMO(2)
      else
        printf "One symmetry...NTMO(1)=%d \n",NTMO(1)
      end
     else
      printf "... else branch...."
     end

     ##### while cycle demo #####
     printf" Few values of the WORK(KTMAT) array, KTMAT=%d... \n",KTMAT
     set $indx=0
     while ($indx.lt.5)
      printf "WORK(%d)=%lf\n",ktmat+$indx,(*WORK)(ktmat+$indx) 
      #  p ktmat+$indx
      set $indx=$indx+1
     end
    end

In the working space type *gdb dirac.x* to run the *dirac.x* executable
in the debugging mode. When the debugger's command line appears, type
*source gdb_demo* to load the debugger source file.

Each time when you modify and recompile DIRAC please retype again *file
dirac.x* in the gdb mode to reload the fresh executable.

If you modify the *gdb_demo* debugger source file apply again the gdb
command *source gdb_demo* to reload it.

More GNU/gdb know-how is given at [this
web-page](http://sources.redhat.com/gdb/current/onlinedocs/gdb.html).

## Other debuggers

To debug the *dirac.x* program compiled with the Intel suite we
recommended to employ own Intel debugger, *idb*, which is works with the
same set of commands as the GNU/gdb counterpart. The same holds for
Portland compilers. Combining different compilers with different
debuggers does not work well.

For parallel debugging we recommend special (commercial) software, which
is able to follow individual threads, like the *totalview debugger*.
Otherwise in a multithreaded *dirac.x* run each thread can be caught and
debugged with any serial debugger.
