orphan  

# Density-functionals available in DIRAC using the `HAMILTONIAN_.DFT` keyword

Below is a list of density functionals available with the keyword
`HAMILTONIAN_.DFT` in DIRAC:

## *Exchange functionals*

- LDA exchange Dirac1930
- Becke 1988 Becke1988
- The correction term to LDA proposed by Becke Becke1988
- Perdew--Wang 1986 Perdew1986

## *Correlation functionals*

- The Vosko--Wilk--Nusair LDA correlation functional (VWN5) Vosko1980
- The Lee--Yang--Parr functional Lee1988
- Perdew 1986 Perdew1986

## *Predefined combinations*

### LDA functionals

The explicit definitions in terms of GGAKEY is also given.

LDA  
GGAKEY Slater=1.0 VWN5=1.0

SVWN5  
GGAKEY Slater=1.0 VWN5=1.0

SVWN3  
GGAKEY Slater=1.0 VWN3=1.0

Xalpha  
...

### GGA functionals

The explicit definitions in terms of GGAKEY is also given.

BLYP  
GGAKEY Slater=1.0 Becke=1.0 LYP=1.0

PBE  
GGAKEY PBEx=1.0 PBEc=1.0

RPBE  
GGAKEY RPBEX=1.0 PBEC=1.0

BP86  
GGAKEY Slater=1.0 Becke=1.0 P86c=1.0 PZ81=1.0

BPW91  
GGAKEY Slater=1.0 Becke=1.0 PW91c=1.0

PP86  
GGAKEY PW86x=1.0 P86c=1.0

KT1  
GGAKEY Slater=1.0 VWN5=1.0 KT=-0.006

KT2  
GGAKEY Slater=1.07173 VWN5=0.576727 KT=-0.006

KT3  
GGAKEY Slater=1.092 OPTX=-0.925452 LYP=0.864409 KT=-0.004

OLYP  
GGAKEY Slater=1.05151 OPTX=-1.43169 LYP=1.0

### Hybrid functionals

The explicit definitions in terms of GGAKEY is also given.

B3LYP  
GGAKEY Slater=0.8 Becke=0.72 HF=0.2 VWN5=0.19 LYP=0.81

B3LYP-G  
GGAKEY Slater=0.8 Becke=0.72 HF=0.2 VWN3=0.19 LYP=0.81

B3P86  
GGAKEY Slater=0.8 Becke=0.72 VWN5=0.19 P86c=0.81

B3P86-G  
GGAKEY Slater=0.8 Becke=0.72 VWN3=0.19 P86c=0.81

PBE0  
GGAKEY PBEx=0.75 PBEC=1.0 HF=0.25 HF

PBE38  
GGAKEY PBEx=0.625 PBEx HF=0.375 PBEC=1.0

### Long-range corrected functionals

The explicit definitions in terms of CAM is also given.

CAMB3LYP  
CAM p:alpha=0.19 p:beta=0.46 p:mu=0.33 x:slater=1 x:becke=1 c:lyp=0.81
c:vwn5=0.19
