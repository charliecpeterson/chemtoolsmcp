orphan  

# Good Fortran90 programming practices

We present some proper Fortran90 programming practices. By adopting them
by programmers one could be working with readable and clean code.

## Module names and file names

Use the same name for the module and the file. For instance, if your
module is called foobar, call the file foobar.F90. This makes it easier
to find the source code linked by "use foobar".

It is possible to group several modules into one file. Do it when your
modules share some common characteritics. Modules grouping in files
prevents flood of individual module source files slowing down the
repository connections.

## Example module

<div class="literalinclude" language="fortran">

nice_module.F90

</div>

## Include files and common blocks in the F90 era

Please do not introduce new common blocks. When you write new code and
use F90, there is no need to introduce new common blocks (and include
files) - use modules instead, they are much safer. Common blocks make
the code harder to maintain.

Sometimes you have to use already present include files in F90. To make this work:  
- only use "!" to start comments in include files (all F77 compilers
  understand this)
- continuation use a & in column 73 of the first line and a & (nothing
  else!) in column 6 of the following line.

This will work with both free form and fixed form when the file is included)  
- declare all variables explicitly, so that they can be used with the
  *IMPLICIT NONE* declaration.

There is a Python-script *reformat_includes.py* in the utils directory,
that reformats include files according to these rules.

## I/O control statements

To ease the localization of input/output-type bugs during the code run
please provide main I/O statements (OPEN, READ) with the IOSTAT
parameter evaluation or with other error-trapping programing scheme.

Examples: :

    !... control reading if gaunt=true
    READ(LUCMD,*,IOSTAT=IOS) ILLINT,ISLINT,ISSINT,IGTINT
    IF (IOS.NE.0) THEN
      WRITE(LUPRI,'(/,2X,A)') 'Error in SCF INTFLG reading !'
    & //'4 parameters needed for Gaunt term !'
      CALL FLSHFO(LUPRI)
      CALL QUIT( 'DHFINP: Error in INTFLG reading for GAUNT=true!')
    ENDIF

or (assuming that LU is correct unit number: :

    WRITE(LU,desired_format,IOSTAT=IOS) SOMETHING
    IF (IOS.NE.0) THEN
    ! Yell, that some variables in "SOMETHING" are out of desired format-bounds ;
    ! this could indicate run-time error (getting variable(s) hurt before)
    ! might activate program stop (CALL QUIT)....
    ENDIF

## F90 INCLUDE keyword

Do not use it. Use \#include preprocessor statements only.

## SIXLTR SBRTNE NAMING OBSOLT

Because it is so hard to read and understand. There is no need to to
stick to 6 letter names anymore. The limit is 31 characters. Use
meaningful names so that other people (and you in a couple of months)
will understand your code.

## Dynamical strings

The programmer may utilize short dynamical strings carrying descriptive
labels, and attach this data type to his own data structures.

The module is :

    use mystring

Methods at hand are:

1)  creating the string

<!-- -->

    type(string) :: this_string
    call make_string(this_string,'some description text')

2)  printing the string on the LUPRI channel

<!-- -->

    call print_string(this_string)

3)  copying one string into another

<!-- -->

    call copy_string(this_string, another_string)

4)  canceling (deallocating) the string

<!-- -->

    call cancel_string(this_string)

For more details about the dynamical string functionalities check
directly the `dynamic_string.F90 <../../../src/gp/dynamic_string.F90>`
source file.

## Code debugging with LINEFILE

C-debugging symbols are defined by the preprocessor on each line.
Therefore it is useful to insert printouts of file name and line number:
:

    print *, 'I am in file ',__FILE__, ' on line ', __LINE__

Now, of course we don't want to write that everywhere. What one can do
is to use a macro: :

    #define MYFUNCTION(x) myfunction(x,__FILE__,__LINE__)

This allows '''myfunction''' to print good error messages, because it
knows the value of \_\_FILE\_\_ and \_\_LINE\_\_. But the macro has to
be defined somewhere, so you need an '''#include "mymem.h"''' or
something.

Here is an example that works, just for using \_\_FILE\_\_ and
\_\_LINE\_\_ statements: :

    subroutine mysub(someargument,file,line)
      character(*) file
      integer someargument, line
      print *, 'mysub(',someargument,')'
      print *, 'I was called in file "',file,'" at line ',line
    end subroutine mysub

    #define mysub_wrapped(arg) mysub(arg,__FILE__,__LINE__)

    program test
      integer x
      x = 12
      call mysub_wrapped(x)
    end program test

For large scale employment one has to put the '''#define''' in an
include file, or better, in a Fortran90 module.
