orphan  

# Davidson size-consistency corrections for truncated CI wave functions

## Background

A serious deficiency of truncated CI expansions, including the most
popular ones, namely (multi-reference; MR)CI-SD is the lack of
size-consistency; i.e., the energy of the system does not scale properly
with the system size. Besides corrections to the CI equations which will
improve the energy and the wave function, *a posteriori* corrections
(most often referred to as *Davidson-type* corrections or *+Q*) to the
energy have become quite popular. A comprehensive overview on this topic
has been given recently in a [review paper on multiconfigurational and
multirefernce methods](http://dx.doi.org/10.1021/cr200137a). Starting
with Dirac14 we have implemented a number of different flavors of the
latter *a posteriori* corrections to the energy for the modules KRCI
with LUCIAREL or with LUCITA. The nomenclature and detailed equations
for each correction is explained in great detail in the above review
article (see chapter 2.1.3.1).

## Example: Water

Following the example for water in the [review
article](http://dx.doi.org/10.1021/cr200137a) we need to run two
calculations. Although it's possible to do it in one step we will split
the task in two.

### Step 1: Get reference wave function and configuration(s)

We first start with a multiconfigurational SCF calculation for water
with a CAS(4,4) space using a cc-pVDZ basis set. This will yield a
suitable reference wave function for the subsequent CI. The MCSCF module
will in addition automatically create a file named **refvec.luci** which
contains the reference configurations needed later for the *+Q*
corrections. Our CASSCF input `scf-casscf.inp` is

<div class="literalinclude">

scf-casscf.inp

</div>

The molecular structure is located in `h2o.xyz`

<div class="literalinclude">

h2o.xyz

</div>

Suppose that we have run the CASSCF calculation using:

    pam --inp=scf-casscf.inp  --mol=h2o.xyz --outkrmc --get="refvec.luci"

where we keep the MCSCF coefficients (in KRMCSCF) and the reference
vectors stored in refvec.luci we are now in position to proceed to Step
2.

### Step 2: Run the MRCISD calculation and compute the +Q corrections

In the second step we now run a MRCISD calculation using the reference
wave function coefficients from the previous step as starting point and
compute the *+Q* corrections to the MRCISD energy. Using the same
molecular input structure we run the MRCISD+Q calculation using the
input file `krci-q.inp`

<div class="literalinclude">

krci-q.inp

</div>

with the following start command:

    pam --inp=krci-q.inp  --mol=h2o.xyz --inkrmc --put="refvec.luci"

We will find a summary of different *+Q* corrections printed at the end
of the output:

<div class="literalinclude">

summary_q.txt

</div>

where a detailed account of the general *performance* of the different
*+Q* corrections (and which other QC programs have implemented which
*+Q* correction by default) is given in chapter 2.1.3.1 of the above
given review article. The equations references given in the output match
the equation labels given in chapter 2.1.3.1. We strongly recommend the
interested user to carefully read this chapter 2.1.3.1.
