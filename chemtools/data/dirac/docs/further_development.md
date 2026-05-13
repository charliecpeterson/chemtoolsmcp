orphan  

# Past and current works on DIRAC development

This web-page lists various projects for the development of the DIRAC
software. It involved the collection of finished, on-going and proposed
bachelor and master thesis projects for (mainly) IT-students, plus
general wishlist.

IT-students at universities around the world have potential to
contribute to the DIRAC infrastructure development. For them, one has
specify suitable, "IT-only" projects around DIRAC, without the need of
the deep knowledge of relativistic quantum chemical methods.

The DIRAC developers are kindly invited to share possible project
proposals here. Each proposed theme should have short desciption
(annotation), author (supervisor) and assigned student if any.

## Thesis list

### Comparison of CPU and GPU mathematical libraries

The aim of this thesis was to compare the speed of some numerical
procedures from CUBLAS and CULA libraries with their Fotran analogues
BLAS and LAPACK libraries. We wrote a short working program (hosted on
github now) in Fortran and C which compares the time complexity for
various input parameters (matrix, vector sizes etc). The usefullness of
this student project is in possible application of CUBLAS / CULA library
functions in the entire DIRAC program suite. (Finished in summer 2012,
student D.Kuzma, supervised by Miro Ilias.)

### CMake platform universal system for the software buildup

First, learn the CMake buildup system on simple testing programs written
in C,C++,Fortran languages. Later, try to program more complex CMake
buildup receipes, including macros. Assigned student I.Hrasko. Miro
Ilias was supervising, Rado Bast was the referee. This bachelor project
was finished in summer 2013.

### Documentation generator Sphinx and its usage for DIRAC

The Sphinx documentation system, based upon Python language, has great
potential for creating and easy maintaining the on-line documentation.
This is clearly demonstrated on the DIRAC documentation. However, there
is place for various improvements in the documentation framework. The
assigned IT-student (Juraj Bubniak, supervised by Miro Ilias) polished
the bibliography (for instance, better formatted table of references
would fit), math equations (export in various formats), and introduces
other goodies. Juraj succesfully defended his thesis in summer 2015.

### Adpatation of DIRAC for Microsoft Windows and high-performace computing

We need to adapt DIRAC thorougly for the MS Windows operating system and
enable high-performace computing also on this (commertial) operating
system. Assigned and currently working student is Ivan Hrasko, who
gained rich experience with CMake in his bachelor project (Miro Ilias
supervising). This master project was finished in summer 2015.

### IT-aspects of grid-computations with DIRAC

Complex adaptation of DIRAC for computing in the grid environment.
Create set of universal, low-level scripts (bash), handling the DIRAC
suite calculations on unknown grid computing elements. Supervisor Miro
Ilias is currently looking for an IT-student.

### Testing various compilers and mathematical libraries for DIRAC

It is known that selection of compilers, their flags and mathematical
libraries has influence on the code's time performance. For DIRAC, the
aim is to employ variours compilers (GNU, Intel, Portland), available
libraries (MKL, ACML, ATLAS, netlib) for the time performance
measurements. We hope to select set of compiler flags and suitable
libraries giving the best execution timing. Supervisor Miro Ilias got an
IT-student for bachelor thesis, hopefully he will start the work from
2016.

### Git and web-based project development platforms in the service of DIRAC

The Git version control system is very popular for the code development.
On top of it, various web-based project development platforms (like
assembla, bitbucket, deveo, gitorious etc) are heplful for managing git
repositories and controling the program development. The aim is to
describe (in the form of manual) the Git system and available project
development web-servers. We investigate various features and show how
all these goodies can be applied for the collaborative DIRAC
development. This project is suitable for bachelor thesis.

### Doxygen code documentation for DIRAC

The *Doxygen* is popular and free code documentation generator. It works
for many programming languages, including C, C++ and Fortran. Our aim is
to adapt DIRAC code description comments for the *Doxygen*, and,
eventually, the *Doxygen* source code as well in order to get properly
generated documentation of all Fortran, C and C++ DIRAC source files.
This theme is suitable for one or more IT-students who are wanted.

## Other ideas

### New input reader with documenting of keywords

Let us write/use an keyword input reader which requires keywords to have
some documentation. Otherwise the code won't even compile. Then the
"make html" command would go through the source code and generate the
RST files which would exactly reflect the state of the code (unless of
course somebody changes the keyword without changing the text 2 lines
below but we could introduce some punishment for it).
