orphan  

# Long-range WFT/short-range DFT

Long-range wave function theory (lrWFT) combined with short-range
density functional theory (srDFT) is based on a separation of the
two-electron interaction

``` math
\frac{1}{r_{12}}=\frac{erf\left(\mu r_{12}\right)}{r_{12}}
- \frac{1-erf\left(\mu r_{12}\right)}{r_{12}}
```

here using the error function (erf). The first term, the long-range
interaction, is assigned to wave function theory, whereas the second
term, the short-range interaction, is assigned to DFT. Contrary to
conventional DFT the fictitious reference (Kohn-Sham) system is no
longer non-interacting and can not be solved exactly, so that the
hierarchy of approximate wave function methods must be invoked.
MP2-srDFT has been implemented in DIRAC by Ossama Kullie and Trond Saue
Kullie2011.

## Specification of functionals

### srLDA

Short-range LDA is specified as:

    **HAMILTONIAN
    .DFT
    CAM p:alpha=0.0 p:beta=1.0 p:mu=0.4 x:slater=1.0 c:srvwn5=1.0

### srGGA

Two flavours of short-range PBE has been implemented in DIRAC. The
short-range PBE of Hirao and co-workers is specified as:

    **HAMILTONIAN
    .DFT
    CAM p:alpha=0.0 p:beta=1.0 p:mu=1.0 x:PBEX=1.0 c:srPBEC=1.0

The short-range PBE of Scuseria and co-workers is specified as:

    **HAMILTONIAN
    .HFXATT
    1.0D0
    .HFXMU
    1.0D0
    .DFT
    GGAKEY HJSX=1.0 srPBEC=1.0

## MP2-srDFT

MP2-srDFT is activated by specifying the short-range DFT functional as
above and complementing with:

    **WAVE FUNCTIONS
    .SCF
    .MP2

Further specifications can be given in the `*SCF` and `*MP2CAL`
subsections.
