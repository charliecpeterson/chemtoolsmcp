orphan  

# Static linking

Theoretically, it's easy to tell the *setup* script to prepare the
dirac.x executable as statically linked (with the "--static" flag,
thanks to the intervined python scripts and the universal CMake buildup
system.

However, neither all DIRAC buildups are ending up happily with static
dirac.x, nor - in cases of succesfull compilation - resulting static
executables are performing correctly.

Therefore we show some example configurations giving correct, partially
correct, totally wrong and non-compilable executables. We hope that this
info shall help developers to identify possible cracks in the generation
of their own static executables.

Developers and users are invited to extend this collection of static
buildup experiences.

Report about successful static DIRAC buildups and deployments is in
reference Ilias2014.

## Generating static OpenMPI

If you want to have static dirac.x running in parallel way, users have
to prepare own static version of the OpenMPI wrappers. Download the
installation package first, unpack it and follow installation
instructions.

You get static OpenMPI suite by adding specific configuration flags. For
example, with GNU compilers:

    ./configure --prefix=<path> --without-memory-manager CXX=g++ CC=gcc F77=gfortran FC=gfortran LDFLAGS=--static --disable-shared --enable-static

and with Intel compilers:

    ./configure --prefix=<space> --without-memory-manager CXX=icpc CC=icc F77=ifort FC=ifort LDFLAGS=--static --disable-shared --enable-static

finally type

    make all install

You could to check your obtained OpemMPI wrappers for static linking:

    ldd mpif90
    not a dynamic executable

### Correct static executables

- miro.utcpd.sk:

<!-- -->

    Linux-2.6.30-1-amd64, ownmath, i32lp64, GNU Fortran/gcc (Debian 4.4.5-10) 4.4.5; -static -fpic, nooptim
    Linux-2.6.30-1-amd64, ownmath, i32lp64, GNU Fortran/gcc (Debian 4.6.2-9); -static -fpic, nooptim

- frpd2.utcpd.sk:
  - Linux-2.6.32-5-amd64; ownmath; i32lp64, GNU Fortran/gcc (Debian
    4.6.1-4); -static -fpic, nooptim
  - Linux-2.6.32-5-amd64, ATLAS library; i32lp64, GNU Fortran/gcc
    (Debian 4.6.1-4); optimized gives few warnings at the linking step:

<!-- -->

    /usr/lib/gcc/x86_64-linux-gnu/4.6/../../../x86_64-linux-gnu/libpthread.a(sem_open.o): In function `sem_open':
    /home/aurel32/eglibc/eglibc-2.13/nptl/sem_open.c:333: warning: the use of `mktemp' is dangerous, better use `mkstemp' or `mkdtemp'

This is due to the " -Wl,--whole-archive -lpthread
-Wl,--no-whole-archive " linking command containing the pthread-library,
but this must be, because placing the sole "-lpthread" keyword elsewhere
generates wrong executable giving this error:

    Program received signal 11 (SIGSEGV): Segmentation fault.
    Backtrace for this error:
    + function __restore_rt (0x15D0620)
      from file sigaction.c

- grid3.ui.savba.sk:

<!-- -->

    Linux-2.6.18-274.3.1.el5; ownath; x86_64, ilp64, GNU Fortran (GCC) 4.4.4 20100726 (Red Hat 4.4.4-13)';  -g -static -fpic; opt
    Note: this gfortran shows error in test program for NAMELIST reading, however, DIRAC runs OK...
    NB: gfortran44 gives crashes for these OpenRSP tests: openrsp_pv,openrsp_jones,openrsp_cme.
    For "pure" runs you have to have higher version of GNU compilers.

- miro_ilias_desktop (home PC Desktop, with most recent Ubuntu)

  > - Linux-3.0.0-14-generic; x86_64; i32lp64; GNU Fortran/gcc
  >   (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1;
  >   /usr/lib/libblas.a+/usr/lib/liblapack.a; -g -static -fpic -O0 (NB:
  >   as superuser remove xerbla.o from /usr/lib/liblapack.a -mixing
  >   with /usr/lib/liblas.a)
  > - Linux-3.0.0-14-generic;x86_64; i32lp64; GNU Fortran/gcc
  >   (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1; ATLAS-static; -g -static
  >   -fpic -O0 (fixed bug below)

- opteron.tau.ac.il:

<!-- -->

    ./setup  --type=debug  -D VERBOSE_OUTPUT=ON --build=build_static  --static --internal-math  --int64
    /opt/intel/fce/10.1.015/bin/ifort (ifort (IFORT) 10.1 20080312); /usr/bin/gcc - gcc (GCC) 4.1.2 20080704 (Red Hat 4.1.2-51)

- 194.160.34.207, Miro's old PC Desktop with updated Ubuntu
  - Linux-3.0.0-16-generic; GNU Fortran (Ubuntu/Linaro 4.6.1-9ubuntu3)
    4.6.1 - both serial and parallel OpenMPI dirac.x are working

Based on these achievements we may say that GNU compilers produce proper
static executable mostly not linked against GNU external static
libraries (libblas-dev,liblapack-dev,libatlas). Likewise the Intel
compiler - ifort - gives correct static executable.

### Partially wrong executables

One can state that static executables are partially wrong when large
portion of tests is crashing. For instance, failures are happening in
input readings of DFTINPUT, CI/RELCCSD.

- gbb4.grid.umb.sk:

<!-- -->

    Linux-2.6.26-2-xen-amd64;ilp64,MKL ilp64 static math, ifort (IFORT) 12.0.2 20110112, gcc (Debian 4.3.2-1.1) 4.3.2
    export MATH_ROOT='/home/chemia/ucitelia/milias/bin/intel_stuff/intel/mkl'
    setup --static -D VERBOSE_OUTPUT=ON --int64

- pd.uniza.sk:

<!-- -->

    Linux-2.6.32-5-amd64, i32lp64, ownmath; GNU Fortran/gcc (Debian 4.4.5-8) 4.4.5
    setup --static -D VERBOSE_OUTPUT=ON  --internal-math (--debug)

- grafix.fpv.umb.sk:
- Linux-2.6.32-28-generic; x86_64; i32lp64; GNU Fortran/Gcc (Ubuntu
  4.4.3-4ubuntu5) 4.4.3; ATLAS-static:

<!-- -->

    setup --static -D VERBOSE_OUTPUT=ON  --debug

- Linux-2.6.32-28-generic; x86_64; ilp64; GNU Fortran/Gcc (Ubuntu
  4.4.3-4ubuntu5) 4.4.3; ownmath

<!-- -->

    setup -D VERBOSE_OUTPUT=ON --debug --static --int64 --internal-math

Note that the Gfortran error in NAMELIST reading is described
<http://comments.gmane.org/gmane.comp.gcc.bugs/289633> . This ill Gcc
installation is probably causing mentioned tests failures.

### Wrong executables

Here we describe cases of static dirac.x executables crashing already at
the start of the run.

- miro_ilias_desktop:

Linux-3.0.0-14-generic;x86_64; i32lp64; GNU Fortran/gcc (Ubuntu/Linaro
4.6.1-9ubuntu3) 4.6.1; ATLAS-static; -g -static -fpic -O0, giving :

    Program received signal 11 (SIGSEGV): Segmentation fault.
    Backtrace for this error:
    + function __restore_rt (0x15C4220)
    from file sigaction.c
    Segmentation fault
    (see also http://gcc.gnu.org/bugzilla/show_bug.cgi?format=multiple&id=42477)

- 194.160.135.47:

Linux-2.6.30-1-amd64, i32lp64, GNU Fortran/gcc (Debian 4.4.5-10) 4.4.5;
-static -fpic, nooptim does not compile with static external
/usr/lib/libblas.a+/usr/lib/liblapack.a, giving :

    /usr/lib/liblapack.a(xerbla.o):function xerbla_: error: undefined reference to '_gfortran_transfer_character_write'
    /usr/lib/liblapack.a(xerbla.o):function xerbla_: error: undefined reference to '_gfortran_transfer_integer_write'
    /usr/bin/ld: error: /usr/lib/libblas.a(xerbla.o): multiple definition of 'xerbla\_'
    /usr/bin/ld: /usr/lib/liblapack.a(xerbla.o): previous definition here

The upgrade is compilable, but does not work with static ATLAS (as
frpd2): Linux-2.6.30-1-amd64, ATLAS BLAS+LAPACK; i32lp64, GNU
Fortran/gcc (Debian 4.6.2-9); -static -fpic, nooptim

### Executables not compilable

We give some cases when one can not obtain the static dirac.x executable
due to error at linking stage.

Linking Fortran executable dirac.x gives for parallel, OpenMPI, for
example :

    /home/ilias/bin/openmpi_ilp64/lib/libopen-pal.a(dlopen.o): In function 'vm_open':
    dlopen.c:(.text+0x148): warning: Using 'dlopen' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking
    /home/ilias/bin/openmpi_ilp64/lib/libopen-rte.a(plm_rsh_module.o): In function `setup_launch':
    plm_rsh_module.c:(.text+0x821): warning: Using 'getpwuid' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking
    /home/ilias/bin/openmpi_ilp64/lib/libmpi.a(btl_tcp_component.o): In function `mca_btl_tcp_component_create_listen':
    btl_tcp_component.c:(.text+0x104): warning: Using 'getaddrinfo' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking
    [100%] Built target dirac.x

- opteron.tau.ac.il

<!-- -->

    export MATH_ROOT='/opt/intel/mkl/10.0.3.020/lib/em64t' , static buildup does not work with MKL libs, giving 

    /opt/intel/fce/10.1.015/bin/ifort -static -Wl,-E -w -assume byterecl -DVART -g -traceback -static-libgcc -static-intel -i8 -O0 CMakeFiles/dirac.x.dir/main/main.F90.o  -o dirac.x lib/libdirac.a lib/libxcfun.a  -Wl,--start-g   /opt/intel/mkl/10.0.3.020/lib/em64t/libmkl_lapack.a    /opt/intel/mkl/10.0.3.020/lib/em64t/libmkl_core.a   /opt/intel/mkl/10.0.3.020/lib/em64t/libmkls_ilp64.a  /opt/intel/mkl/10.0.3.020/lib/em64t/libmkl_sequential.a /opt/intel/mkl/10.0.3.020/lib/em64t/libmkl_em64t.a   /opt/intel/mkl/10.0.3.020/lib/emibguide.a  /usr/lib64/libpthread.a /usr/lib64/libm.a -Wl,--end-group
       ld: cannot find libmkl_intel_lp64.a

- grid3.ui.savba.sk

> - SL5 Linux, 2.6.18-348.1.1.el5, x86_64; i32lp64 using system static
>   blas+lapack:
>
> /usr/bin/gfortran44 -static -Wl,-E -g -fcray-pointer -fbacktrace
> -DVAR_GFORTRAN -DVAR_MFDS -fno-range-check -static -O0
> CMakeFiles/dirac.x.dir/src/main/main.F90.o -o dirac.x lib/libdirac.a
> lib/libxcfun.a -lstdc++ /usr/lib64/liblapack.a /usr/lib64/libblas.a

    /usr/lib64/liblapack.a(ilaenv.o): In function `ilaenv_':
    (.text+0x25d): undefined reference to `_gfortran_copy_string'
    /usr/lib64/liblapack.a(ilaenv.o): In function `ilaenv_':
    (.text+0x2ea): undefined reference to `_gfortran_copy_string'
    /usr/lib64/liblapack.a(ilaenv.o): In function `ilaenv_':
    (.text+0x305): undefined reference to `_gfortran_copy_string'
    /usr/lib64/liblapack.a(ilaenv.o): In function `ilaenv_':
    (.text+0x31b): undefined reference to `_gfortran_copy_string'
    /usr/lib64/liblapack.a(zlargv.o): In function `zlargv_':
    (.text+0x7d2): undefined reference to `_gfortran_pow_r8_i4'
    /usr/lib64/liblapack.a(zlartg.o): In function `zlartg_':
    (.text+0x4d9): undefined reference to `_gfortran_pow_r8_i4'
    /usr/lib64/liblapack.a(dlamch.o): In function `dlamc2_':
    (.text+0x44c): undefined reference to `_gfortran_pow_r8_i4'
    /usr/lib64/liblapack.a(dlamch.o): In function `dlamc2_':
    (.text+0x594): undefined reference to `_gfortran_pow_r8_i4'
    /usr/lib64/liblapack.a(dlamch.o): In function `dlamch_':
    (.text+0xe15): undefined reference to `_gfortran_pow_r8_i4'
    /usr/lib64/liblapack.a(dlamch.o):(.text+0xe39): more undefined references to `_gfortran_pow_r8_i4' follow
    collect2: ld returned 1 exit status

This problem is fixed by abandoning /usr//lib64/liblapack.a completely,
what is accomplished by using "--lapack=off" in the setup flags.
