orphan  

# MP2 sample inputs

Valence + subvalence <sub>2</sub> calculation. The 19 active occupied
Kramers pairs are split in two and the calculation is done in three
batches (1..10+1..10, 11..19+1..10, and 11..19+11..19 for I+J). All
orbitals with energy above 100 hartree (a.u.) are neglected in the
virtual space. No SS integrals are included. Note that the orbitals in
the active space need not be consecutive.

    .NELECT
    80 78
    .....
    *MP2CAL
    .OCCUP
    24,32..40
    24,32..39
    .VIRTUAL
    all
    all
    .VIRTHR
    100.0
    .INTFLG
    1 1 0
    .IJTSK
    10
