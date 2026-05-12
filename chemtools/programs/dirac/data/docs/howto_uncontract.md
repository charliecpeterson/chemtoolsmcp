orphan  

# How to uncontract basis sets

It is boring to uncontract basis sets manually. There is a keyword for
it:

    **DIRAC
    [...]
    **INTEGRALS
    *READIN
    .UNCONTRACT

Note that there is the limit for maximum number of exponents in one
shell, which specified in the "include/cbirea.h" file (as MAXPRD = 35).
