orphan  

# CAM functional

In DIRAC, you can calculate with the CAMB3LYP functional using:

    .DFT
     CAMB3LYP

It is also possible to change the default parameters. The following line
will reproduce the above functional:

    .DFT
     CAM p:alpha=0.19 p:beta=0.46 p:mu=0.33 x:slater=1 x:becke=1 c:lyp=0.81 c:vwn5=0.19

but it will give you complete freedom over the parameters. As always,
please verify your results carefully when changing these parameters.

## Approaching the B3LYP limit

CAM uses the following partitioning of the two-electron interaction:

``` math
\frac{1}{r_{12}} = \frac{1 - [\alpha + \beta erf (\mu r_{12})}{r_{12}}
                 + \frac{    [\alpha + \beta erf (\mu r_{12})}{r_{12}}
```

For testing purposes we can try to approach the B3LYP limit using the
CAM code, in order to check that the alpha/beta limits work well.

In B3LYP, "HF" admixture is 0.2 so in CAM this can be obtained with
alpha=0.2 and beta=0.0.

So this is B3LYP:

    .DFT
     GGAKEY Slater=0.8 Becke=0.72 HF=0.2 LYP=0.81 VWN=0.19

Now naively we could try to obtain B3LYP results like this (**this is
wrong**):

    .DFT
     CAM p:alpha=0.2 p:beta=0.0 p:mu=0.0 x:slater=0.8 x:becke=0.72 c:lyp=0.81 c:vwn5=0.19

**This does not work in DIRAC** and the correct B3LYP expressed using
CAM in DIRAC can be obtained like this:

    .DFT
     CAM p:alpha=0.2 p:beta=0.0 p:mu=0.0 x:slater=1.0 x:becke=0.9 c:lyp=0.81 c:vwn5=0.19

This is because x:slater and x:becke get scaled inside fun-cam.c by
(1-alpha). The x:becke=0.9 can be surprising if you read the paper by
Yanai et al. \[Chem. Phys. Lett. 393 (2004) 51\] and follow their
examples.
