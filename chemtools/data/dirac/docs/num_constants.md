orphan  

# How numerical constants are handled in DIRAC

In Dirac numerical constants are handled in a few different ways.

For example the code pieces ::  
double precision :: x,y y = 2\*x

with an integer "two", ::  
double precision :: x,y y = 2.0D0\*x

with a double precision "two" and ::  
double precision :: x,y double precision, parameter :: DTWO = 2 y =
DTWO\*x

all results in the same value being assigned to the variable y. The two
first examples often result in exactly the same machine code, since the
compiler know how to convert the integer or double precision number
"two" to the most efficient form. In the third example the number is
first stored in a (parameter) variable, and can then be used. Besides
obvious uses for constants such as pi, this is perhaps most important
when the constant is passed as an argument to a subroutine.

Since Fortran does not know what type of argument the subroutine expects
it cannot convert the argument given by the programmer to the correct
type. Therefore it is very important that the argument is passed with
the correct type by the programmer. By making, for example, a double
precision number DTWO the type is guaranteed to have this representation
when passed to the subroutine. The same effect can be achieved by
writing i.e. 2.0D0 (if a double is expected), or 2 (if an integer is
expected). This does not mean that it is somehow "wrong" to write y =
2\*x in a line that is not a subroutine call, even if x and y are
floating point numbers. In this case the compiler knows what to do, and
you don't have to write y = 2.0D0\*x. In the same way you can write y =
x/3, and get correct results. On certain older cpus it was slightly
faster to execute :

    onethird = 1.0D0/3.0D0
    y = onethird*x

than a simple divide by 3 (if repeated maybe times). On modern cpus the
difference is most likely minimal. Using this style might cost one bit
of precision, but in most cases it is best to write code in the most
readable way possible and how to do this has to be decided on a case by
case basis.

Finally most compilers are now smart enough to evaluate constants such
as sqrt(2.0D0) at compile time, so if you only use this constant once
there is no point in declaring this value as a parameter. However, here
you have to be careful to pass the correct type to sqrt(). The call
sqrt(2.0) will return a single precision approximation to the square
root of two.

See \[<http://www.ibiblio.org/pub/languages/fortran/ch2-2.html>\] for
more information.
