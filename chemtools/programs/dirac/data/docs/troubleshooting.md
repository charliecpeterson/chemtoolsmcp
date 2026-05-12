orphan  

# Troubleshooting

## Can I move the DFCOEF file from one machine to another and restart from it?

In general no, unless it is the same architecture.

What you can do instead is to export the DFCOEF in a machine independent
format using the .PCMOUT keyword. This creates a file called DFPCMO.
Copy DFPCMO to the other machine, copy it to the scratch directory using
the pam script, and DIRAC will correctly restart from it.

## I tried to restart but get the error "Incompatible number of basis functions" - what does it mean?

This probably means that you have changed the basis set. Presently DIRAC
cannot restart from DFCOEF or DFPCMO if the basis set changes (more
correctly if matrix dimensions change).
