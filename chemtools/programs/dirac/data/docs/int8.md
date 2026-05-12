orphan  

# Working with integer\*8 variables

## Introduction

What happens when a subroutine expecting an integer\*4 argument is
called with a integer\*8 argument?

The answer depends on the hardware and software platform, and
unfortunately errors can often be difficult to detect. To understand
this, look at the memory layout of the number 12345 stored in integer\*4
and integer\*8 variables, on an regular Intel CPU (x86):

    39 30  0  0 
    39 30  0  0  0  0  0  0

The first line shows the four bytes of the integer\*4, in the order they
are stored in memory on this particular machine. The second line shows
the eight bytes of the integer\*8 representing the same number. As you
can see the start of the integer\*8 is the same as the integer\*4 (on
this machine!).

Similarly, the number -1 is stored as

    ff ff ff ff 
    ff ff ff ff ff ff ff ff 

Again the first four bytes are the same.

## Testing program

To understand what can go wrong take the FORTRAN77 example

    subroutine f(x)
    integer*4 x
    print *, x
    x = 12345
    end

    program p
    integer*8 y
    y = -1
    call f(y)
    print *,y
    end

Since Fortran arguments are passed by reference, the subroutine f will
get a pointer to the variable y. From the example above we know that the
first four bytes of an integer\*8 "-1" are the same as those of a
integer\*4 "-1".

Therefore the number "-1" will be printed from f(), after which things
go bad. This is what happens:

First we set y = -1:

    y = ff ff ff ff  ff ff ff ff

Then f(y) is called. The variable x will reference "y", but will only
use the first four bytes:

    x = ff ff ff ff (ff ff ff ff)

This x is printed as "-1". The we set x = 12345, note that only the
first four bytes are modified!

    x = 39 30 00 00 (ff ff ff ff)

When f() returns y will have the value

    y = 39 30 00 00  ff ff ff ff

Which will be printed as -4294954951.

This is just one example of how things can go wrong, and why things can
often work by "chance".

Note that this illustrated situation is very platform dependent; on a
big-endian CPU the above example would not work at all. Always make sure
to use the correct integer type in arguments, and link to correct
versions of external libraries.

## Notes

- The above example would work "correctly" if y was set to 0 on entry.
  If you find that you need to initialize intent(out) variables to 0 you
  may have an integer size mismatch.
- Passing integer arrays of the wrong size gives even more obvious bugs.
- Problems also appear if you pass integer\*4 variables to a subroutine
  expecting integer\*8.

## Integer\*4/8 type safe programs

FORTRAN90 compilers, which are checking also agreement of data types,
should catch most of their incompatibilities in the code.

Though this is not the case of the Fortran77 example given above, one
can rewrite this small piece of the code into a proper 'Fortran90
checking form' by utilizing module.

Program "test_f90.F90" (don't forget big F in the suffix when applying
preprocessor statements!) shall have this form:

    module pass_integer
    contains
    subroutine f(x)
    #if defined (F_INT4)
    integer(kind=4), intent(inout):: x! F90 compiler gives error!
    #else
    integer(kind=8), intent(inout):: x! F90 compiler passed
    #endif
    print *, x
    x = 12345
    end subroutine f
    end module pass_integer
    program p
    use pass_integer
    integer (kind=8):: y
    y = -1
    call f(y)
    print *,y
    end program p

Typing

    gfortran -DF_INT4 test_f90.F90 

gives error of "Type/rank mismatch in argument 'x'" while

    gfortran test_f90.F90

compiles with the proper integer kind.
