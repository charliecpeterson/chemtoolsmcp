orphan  

# Programming rules

Here we provide some guidelines for DIRAC programmers.

## Use of languages

DIRAC is for historical reasons largely written in Fortran 77 (F77), but
in the newer parts of the code you may also see parts in Fortran 90
(F90), C and C++. Fortran 90 programming style with modules and other
usefull features is required.

Do not introduce a new language before consulting the authors forum. New
languages might require compilers that may be proprietary and
consequently not be available to all authors and users.

Likewise do not introduce compiler-dependent extensions, though they are
inviting and so convenient. Do not use them. You will regret it later
when you have to modify the code for other compilers.

### Fortran 90

We require a F90 compiler by default. Therefore all files with .F, .F90,
.f, and .f90 suffixes can contain F90 syntax. In .F and .f you can use
F77 syntax, not in .F90 and .f90 files. In .F and .f files you can use
F90 syntax however respecting the F77 line limits. Files with .F and
.F90 suffixes are preprocessed, files with .f and .f90 are not
preprocessed. If you introduce new source code file, always use .F90
suffix.

F90 is case insensitive and allows both uppercase and lowercase. It is
not necessary to program with the caps lock. Use uppercase and lowercase
in a way that makes your code more readable.

"C" at the beginning of empty lines is unnecessary.

## Do not read keywords with table/goto lookups

Perhaps 98% of the input reading is done using tables and gotos.

Please do not introduce new keyword sections like that for at least
three reasons:

1.  It is a nightmare to read and maintain.
2.  There is no type checking so the user gets useless error messages or
    unexpected results.
3.  New scheme minimizes conflicts when merging branches.

Rather have a look in src/input. This is the modern way to do it.

## Never allocate memory with MEMGET/MEMREL

Large parts of the DIRAC code use MEMGET/MEMREL. Please do not allocate
memory using MEMGET/MEMREL. We are trying to get rid of MEMGET/MEMREL.

Memory allocation should be done either using Andre's alloc/dealloc or
using the native allocate/deallocate routines. Alloc/dealloc are
frontends to the native allocate/deallocate. The advantage of
alloc/dealloc is that the memory allocation gets tracked and shows up in
the high water mark. Do not use other home-cooked memory allocation
schemes unless you have absolutely no other choice.

Here is an example for alloc/dealloc:

    use memory_allocator

    real(8), allocatable :: one_dim_array(:)
    integer, allocatable :: two_dim_array(:, :)

    call alloc(one_dim_array, 1000)
    call alloc(two_dim_array, 1000, 500)

    ...

    call dealloc(one_dim_array)
    call dealloc(two_dim_array)

Here is an example for allocate/deallocate:

    real(8), allocatable :: one_dim_array(:)
    integer, allocatable :: two_dim_array(:, :)

    allocate(one_dim_array(1000))
    allocate(two_dim_array(1000, 500))

    ...

    deallocate(one_dim_array)
    deallocate(two_dim_array)

When allocating memory please consider the following:  
- The memory is not a bottomless pit. Try to be economic.
- Do not allocate and deallocate inside of a loop that is traversed a
  billion times. Allocate/deallocate outside.
- If you allocate arrays, clean up after yourself and deallocate data
  that is not needed anymore as soon as possible to keep the peak
  allocation as low as possible. Ideally at the end of the run we should
  have zero allocations. We are very far away from that.

## Never use implicit.h

Plating the *implicit.h* into the code is bad programing practice and I
strongly recommend to always use the *implicit none* statement.

Reasons: 1. With *implicit.h* it is easy to use variables that are
undefined. Example: :

    call daxpy(size(array1), D1, array1, 1, array2, 2)

if one forgets to define the parameter D1, the code will compile and D1
can be any number so the result is undefined. With "implicit none" this
will not compile.

Another example is a typo: istead of :

    NDIM = 12  ! correct code

you could type :

    NDIN = 12  ! typo

With *implicit none* such mistake won't compile, with the *implicit.h*
the code will compile and behave unexpectedly. In this case also the
know compiler flag "-Wuninitialized" won't help!

In the code we had (and unfortunatelly still have( many bugs like this
and some of us have spent several long afternoons chasing them. The
insertions of "implicit.h" is too dangerous and also experienced
programmers make typos.

2\. The insertion of *implicit none* is nice for other people reading
your code. They can see which variables are local, which are global
(from common blocks). Without implicit none you have no idea whether
this is a local or a global variable if you don't know the common blocks
by heart.

3\. The command *implicit none* makes it easier to identify include
statements that are really used and includes that are just included but
not used. With *implicit.h* you can comment out includes and the code
may still compile. Contrary, the *implicit none* will warn you. If you
remove an include file using *implicit none* and the code compiles, then
this include was useless.

The time that you gain with *implicit.h*, you will spend it in debugging
bugs that implicit none would have detected. *Implicit none* makes it
easier for other people to reuse your routines. With *implicit none*
they know exactly which variables are local to the routine and which are
passed via header files or modules. With *implicit.h* they are
completely in the blind.

## Integer kinds

Please always use *KIND free* declarations in your codes:

    integer :: variable

instead of *KIND restricted* integer variables :

    integer(KIND=4) :: variable
    integer(KIND=8) :: variable

This is because we need flexibility for integer variables: compilation
of the DIRAC suite goes also with predefined integer\*8 variables
(through -i8 Fortranflag), for what all integer variables in the code
must be declared without KIND restriction.

Radovan: I don't agree. I think we should do the exact oposite. We would
avoid a lot of trouble if we would use well defined kinds.

## Interface Fortran and C using iso_c_binding

This is robust and portable. The iso_c_binding is supported by all
modern compilers. Matching types at configure time with
fortran_interface.h is painful and not robust.

## If you replace a code remove the old code

Do not keep old code "just to be sure". We have the well preserved Git
history. Cut dead wood.

Don't check in commented out lines of code. It will make it harder for
other developers to modify or improve your code because they cannot know
whether the commented code is important or not. If you really have to
leave commented code you should document exactly why you feel that the
old code has to stay as a comment inside the source file.

## Never commit new functionality without tests

Functionality without tests will break sooner or later and nobody will
notice for long time.

Also with tests you make it easier to understand what your code does and
you make it easier for other people to improve your code. If there are
no tests, then others will be afraid to modify untested code.

## Make sure that new code compiles with and without MPI

Before you push new code to the main line development, test whether it
compiles with and without MPI.
