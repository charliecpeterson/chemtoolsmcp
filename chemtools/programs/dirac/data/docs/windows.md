orphan  

# DIRAC on Windows

We offer native support for Microsoft Windows 7/8 and for Cygwin.

## Dependencies

1.  The CMake system (<http://www.cmake.org>). Add path to CMake
    executables into your PATH environment variable. Example of path to
    add: *"C:\cmake-3.0.2-win32-x86\bin"*

2.  Fortran, C and C++ compilers. We recommend to use compilers from
    <http://mingw-w64.sourceforge.net/> for both 32/64 bit Windows. You
    can download 32-bit version from [this
    URL](http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/)
    And 64-bit version from [this
    page](http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/)
    In both cases we recommend to download archive file instead of
    installer and then just unpack it somewhere. Add path to compilers
    into your PATH environment variable. Example of path to add:
    *"C:\mingw64\bin"*

3.  The Python interpreter of version at least 2.7
    (<http://www.python.org>). Install 32-bit version of Python on
    32-bit Windows and 64-bit Python on 64-bit Windows. Into PATH
    variable add paths to both Python and its scripts. Example of paths
    to add: *"C:\Python27;C:\Python27\Scripts"*

4.  Zlib library (<http://www.zlib.net/>)

    For 32-bit Windows (if you place zlib libraries at the same place as
    we do then CMake should find them automatically):

    \[a\] download zlib in archive from
    <http://zlib.net/zlib128-dll.zip>

    \[b\] unpack downloaded archive (Extract all)

    \[c\] rename unpacked folder to *"zlib"* (note: that folder must
    contain folders: *"include"*, *"lib"*, *"test"*, file *"zlib1.dll"*
    and some other files)

    \[d\] move it to *"C:\Program Files"*

    \[e\] run in command line these commands: :

        cd C:\Program Files\zlib
        dlltool -D zlib1.dll -d lib/zlib.def -l lib/libzdll.a 

    \[f\] add *"C:\Program Files\zlib"* into PATH

    \[g\] if CMake will have problems to find zlib libraries you can use
    this (available since CMake 2.8.7): :

        python setup -DZLIB_ROOT=[path to zlib folder]

    For 64-bit Windows (if you place zlib libraries at the same place as
    we do then CMake should find them automatically):

    \[a\] download zlib in archive from
    [here](http://sourceforge.net/projects/mingw-w64/files/External%20binary%20packages%20%28Win64%20hosted%29/Binaries%20%2864-bit%29/zlib-1.2.5-bin-x64.zip/download)

    \[b\] unpack downloaded archive (Extract all)

    \[c\] in unpacked folder find folder named *"zlib"* and move it to
    *"C:\Program Files"* (note: that folder must contain folders:
    *"bin"*, *"include"* and *"lib"*)

    \[d\] add *"C:\Program Files\zlib"* into PATH

    \[e\] if CMake will have problems to find zlib libraries you can use
    this (available since CMake 2.8.7): :

        python setup -DZLIB_ROOT=[path to zlib folder]

5.  Boost (<http://www.boost.org/>). Download Boost in archive file from
    <http://sourceforge.net/projects/boost/files/boost/> Then follow
    this instructions (if you place Boost libraries at the same place as
    we do then CMake should find them automatically):

    \[a\] unpack downloaded archive (Extract all)

    \[b\] in unpacked folder (for example:
    *"C:\boost_1_56_0\boost_1_56_0"*) run this command in command line:
    :

        bootstrap.bat mingw

    \[c\] in the same directory run: :

        b2 --with-system --with-filesystem --with-test --with-chrono --with-timer toolset=gcc install

    \[d\] in *"C:\Boost"* you will find *"include"* and *"lib"*
    directories

    \[e\] if CMake will have problems to find your Boost libraries yo
    can use: :

        python setup -DBOOST_INCLUDEDIR=[path]\include -DBOOST_LIBRARYDIR=[path]\lib

6.  To speed-up your calculations you can download OpenBLAS (it includes
    BLAS and LAPACK implementation) library from
    <http://www.openblas.net/>. There are available prebuilt binaries
    for 32/64 bit Windows for Intel NEHALEM architecture with threading
    capability. You can specify number of threads with
    *"OPENBLAS_NUM_THREADS"* environment variable. On 64-bit Windows you
    can choose from *"Win64-int32"* and *"Win64-int64"* versions. Use
    *"Win64-int32"* version when building without *"--int64"* setup
    option and *"Win64-int64"* version when building with *"--int64"*
    setup option. Add path to *"libopenblas.dll"* file into PATH.
    Example of path to add:
    *"C:\libraries\OpenBLAS\OpenBLAS-v0.2.14-Win64-int64\bin"*.

> [!NOTE]
> To use OpenBLAS libraries you have to set them explicitly with
> *"--blas"* and *"--lapack"* setup options. See Build examples below.

> [!NOTE]
> For detailed info about how to use threads and for detailed install
> guide go to <https://github.com/xianyi/OpenBLAS/wiki>.

> [!NOTE]
> If you need you can compile your own OpenBLAS binaries, for example
> for different CPU architecture. See
> <https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio> -
> do not be afraid it is built by MinGW in MSYS!

7.  How to install software needed to create documentation see
    `windows_developers`

## Important notes

1.  You can modify or create PATH environment variable in *"Control
    Panel"* --\> *"User Accounts"* and click *"Change my environment
    variables"*. Please add or change variables only in the part for
    User variables not in the part for System variables. When you want
    to add several paths into PATH environment variable then you can use
    *";"* as a separator.
2.  Please avoid to have spaces in your paths (also in name of your
    user's home directory). The zlib library in *"C:\Program
    Files\zlib"* is an exception from this rule.
3.  Avoid to have path to program *"sh.exe"* in your PATH. See also
    <http://www.cmake.org/Wiki/CMake_MinGW_Compiler_Issues>. Examples of
    not good paths: *"C:\MinGW\msys\1.0\bin"*, *"C:\Git\bin"* or
    *"C:\cygwin\bin"*. These paths are causing conflict with CMake
    programs!
4.  Upon installing Python for all users on the Windows computing server
    it happens that registers are not under HKEY_CURRENT_USER. One has
    to export the "Python" section under HKEY_LOCAL_MACHINE, modify the
    exported file and import it again. See [this
    post](http://stackoverflow.com/a/8712435/3101910).
5.  To check which dynamic libraries are used by your program you can
    use *"Process Explorer"* utility from
    <https://technet.microsoft.com/en-us/sysinternals/bb896653.aspx>.
    This utility can also be used to show number of threads used by your
    program, its I/O operations and so on. Yo should care about your
    program is loading correct *"dll"* libraries during execution. For
    example: you want to build Dirac with *"--int64"* option and you
    specify to use *"Win64-int64"* version of OpenBLAS with *"--blas"*
    option - this is correct. But in your PATH *"Win64-int32"* version
    of OpenBLAS is found before *"Win64-int64"* version and
    *"Win64-int32"* version is used - this is not correct and Dirac
    program fails.

## Windows PowerShell

If you are using Windows PowerShell instead of command line
(*"cmd.exe"*) you should realise that commands look little bit
different: :

    bootstrap.bat mingw

looks in PowerShell like: :

    .\bootstrap.bat mingw

If you have problems to build Dirac in PowerShell try to use command
line (type *"cmd.exe"* in PowerShell).

## Installation and running

Configuration, installation and own execution of DIRAC is possible from
the Windows (DOS) command line or Windows PowerShell. Windows operating
systems are generating executables with the ".exe" suffix, like
dirac.x.exe.

To execute Python scripts please type "python" command before script
name as Windows can not determine scripting language.

Finally, instead of "make" type its MinGW64 version, what is
"mingw32-make".

## Build examples

For specifying one or more parameters for Python scripts (either in the
*".diracrc"* configuration file, or directly by typing in the command
line), you should envelope them into inverted commas, "...". In
*".diracrc"* you can use backlash in Windows paths. For example:

    --basis="C:\Dirac\my-dirac\basis;C:\Dirac\my-dirac\basis_dalton;C:\Dirac\my-dirac\basis_ecp"
    --scratch="D:\dirac\tmp"

> [!NOTE]
> If you want to use *".diracrc"* configuration file it must be placed
> in your user's home directory, for example:
> *"C:\Users\User\\diracrc"*.

If you want to specify *"setup"* options (*"--fc"*, *"--cc"*, *"--cxx"*,
*"--blas"*, *"--lapack"* and others) you have to use slash sign in paths
like in Linux, if you want to specify -D options you have to use
backslash sign in paths. See example below.

> [!NOTE]
> If your compilers are in PATH you do not need to specify full path to
> *"--fc"*, *"--cc"*, *"--cxx"*.

> [!NOTE]
> To show all setup options run in command line: python setup --help.

Configure DIRAC:

    C:\Dirac\my-dirac> python setup --generator="MinGW Makefiles" --int64 --blas="C:/libraries/OpenBLAS/OpenBLAS-v0.2.14-Win64-int64/bin/libopenblas.dll" --lapack="C:/libraries/OpenBLAS/OpenBLAS-v0.2.14-Win64-int64/bin/libopenblas.dll" -D ZLIB_ROOT="C:\libraries\zlib" -D BOOST_INCLUDEDIR="C:\libraries\Boost\include" -D BOOST_LIBRARYDIR="C:\libraries\Boost\lib"

Compile the executable (in the default *build* directory):

    cd build
    C:\Dirac\my-dirac\build> mingw32-make -j 4

Run your selected test set together with uploading results onto the
CDash web:

    C:\Dirac\my-dirac\build> ctest -D ExperimentalTest -L short -D ExperimentalSubmit -j 4

## Intel (not available yet)

We tried to test Dirac with *Intel Parallel Studio XE 2015 Update 3
Cluster Edition*
(<https://software.intel.com/en-us/intel-parallel-studio-xe>) which
contains Intel C/C++ and Fortran compilers, Intel MKL and Intel MPI
libraries in virtual machine with *Microsoft Windows Server 2012 R2*
running on AMD Athlon II 64-bit processor.

Our installation steps:

1.  Install *Visual Studio Professional 2013 with Update 3*
    (<https://www.visualstudio.com/>)
2.  Install *Intel Parallel Studio XE 2015 Update 3 Cluster Edition*
3.  Initialize the Cluster edition tools:

<!-- -->

    cd C:\Program Files (x86)\Intel\Parallel Studio XE 2015
    psxevars.bat intel64

4.  Add *"C:\Intel\Composer XE 2015\bin\intel64_mic;C:\Program Files
    (x86)\Intel\Composer XE\bin\intel64"* into PATH environment
    variable.
5.  Make own *cmd.exe* shortcut, open its properties and put into
    *"Target"* field (then you can use Intel compilers from this custom
    command line):

<!-- -->

    %comspec% /k ""C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64 && "C:\Program Files (x86)\Intel\Composer XE 2015\bin\compilervars.bat" intel64 vs2013"

Intel compiler names:

C/C++ compiler: *icl* Fortran compiler: *ifort* CMake Generator: NMake
Makefiles MPI compilers for C/C++ and Fortran (*.bat* wrappers):
*mpicc*, *mpicxx*, *mpif90* MPI execution program: *mpiexec* (when first
time run after Windows start/user login it asks for user password)

We have used this setup command: :

    python setup --cc=icl --cxx=icl --fc=ifort --type=debug --int64 --generator="NMake Makefiles" -D ENABLE_PCMSOLVER=OFF

and then this command for building: :

    cd build
    nmake

## Cygwin

Regarding the Microsoft Windows platform, DIRAC is able to run under the
Cygwin (<https://cygwin.com/>) intermediating environment, which has
almost all necessary Linux substitutes. Thus it is considered as the
Windows non-native installation, because the Cygwin layer ensures full
Linux workplace on the top of the ground Windows operating system.

In order to build Dirac on Cygwin you have to install these packages:

1.  compilers: *gcc-core*, *gcc-fortran*, *gcc-g++* (also these we need
    to get all header files: *gcc-ada*, *gcc-objc*, *gcc-objc++*)
2.  *make* and *cmake* (it includes *ctest*..)
3.  *python* interpreter
4.  *zlib0*, *zlib-devel*
5.  *git* version control system
6.  optionally you can install *OpenBLAS* (*libopenblas*) - recommended
    for 32 and 64 integer builds or *BLAS* and *LAPACK* (together in
    *liblapack0* package) - but these are only 32-bit versions

> [!NOTE]
> *OpenBLAS* library is placed in *"/usr/bin/cygblas-0.dll"*.

> [!NOTE]
> You have to explicitly specify to use *OpenBLAS* library with
> *"--blas"* setup option.

> [!NOTE]
> With 64 bit integers enabled use builtin *LAPACK* with *OpenBLAS*
> (*OpenBLAS* on Cygwin does not contain *LAPACK*).

> [!NOTE]
> To run *OpenBLAS* in parallel export OPENBLAS_NUM_THREADS environment
> variable.

> [!NOTE]
> *BLAS* and *LAPACK* libraries from *liblapack0* package are placed in
> *"/usr/lib/lapack"* (*"cygblas-0.dll"* and *"cyglapack-0.dll"*).

7.  optionally you can install *Open MPI*

> [!NOTE]
> If you want to use Open MPI you have to run Cygserver as an
> administrator, see
> <https://cygwin.com/cygwin-ug-net/using-cygserver.html>.

8.  to be able to build documentation (*make html*, *make doxygen* -
    problematic, *make slides*) install also: *pkg-config*,
    *ghostscript*, *libfreetype-devel*, *libpng-devel*, *python-gtk2.0*,
    *libgtk2.0-devel*, *doxygen* (why so many see
    <http://blogs.bu.edu/mhirsch/2014/06/matplotlib-in-cygwin-64-bit/>)
    and get and install *pip* from
    <https://pip.pypa.io/en/latest/installing.html> and in Cygwin
    terminal run:

<!-- -->

    pip install sphinx python-dateutil pyparsing sphinxcontrib-bibtex sphinx_rtd_theme numpy matplotlib hieroglyph

> [!NOTE]
> Previous pip install command usually does not work if you have
> installed *libopenblas*. You should remove it, install *liblapack0*
> and try again.

Always check if you are using Cygwin programs and libraries (placed in
for example: *"/usr/bin"*, *"/usr/include"*, *"usr/lib"*,
*"/usr/share"*) instead of programs/libraries from Windows (those are
placed in *"/cygdrive/.."*) for example by this commands: :

    which gcc g++ gfortran make cmake ctest ar ranlib python git doxygen
    whereis zlib cygblas-0.dll

Installation of packages can be done by running Cygwin installer (there
is nothing like *sudo apt-get*). You can use Cygwin installer also to
update installed packages. Sometimes new version of installer is
released and then you must install and update your packages with its new
version but there is no need to completely reinstall your Cygwin
installation.

> [!NOTE]
> Ask your administrator to install Cygwin packages, *"pip"* and other
> software into Cygwin environment.

> [!NOTE]
> You can have also your own Cygwin installation and then you do not
> need to ask anybody.

## Cygwin Troubleshooting

When using Cygwin programs and environment variables from Windows are
also available in Cygwin. There can be problems for example with *Boost*
and *zlib* libraries from Windows which can be detected by CMake when
run from Cygwin. Those libraries can make configuration or build fail.
In order to avoid this temporarily move folders with those libraries to
some another place which is not investigated by CMake or change their
names (then if you want to build from Windows you can return everything
back or see `windows_users` how to learn setup where to look for *zlib*
and *Boost*). Also you should check environment variables.

Example how to point to correct *Boost* and *zlib* libraries: :

    python setup -DBOOST_INCLUDEDIR=[install_prefix]/include -DBOOST_LIBRARYDIR=[install_prefix]/lib -DZLIB_ROOT=/usr/lib

> [!NOTE]
> Boost libraries which can be installed from Cygwin installation are
> not recognized by Dirac setup.

You can try to compile your own (download from
<http://sourceforge.net/projects/boost/files/boost/>): :

    bootstrap.sh gcc --prefix=[install_prefix]
    b2 --with-system --with-filesystem --with-test --with-chrono --with-timer toolset=gcc install

To more clearly see (in case when problems occur) what happens it is
possible to run Dirac configuration directly by CMake when you create a
build directory in Dirac top directory then move into it and run: :

    cmake ..

If there is need to remove Cygwin in an environment with multiple users
then all users must delete their own home directories and then
administrator can delete whole Cygwin installation/folder.

Sometimes you can get errors like *os.fork()* is not available. In that
case try to rebase Cygwin: <http://cygwin.wikia.com/wiki/Rebaseall>.
