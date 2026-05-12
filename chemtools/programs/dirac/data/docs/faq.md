orphan  

# I am about to install DIRAC on a 64-bit machine, should I compile using 64-bit integers or 32-bit integers?

DIRAC runs smoothly (with few exceptions; see below) on 64-bit platforms
using either 32-bit or 64-bit integers.

It is easier to install for 32-bit integers because MPI and math
libraries are typically built for 32-bit integers. Therefore DIRAC is by
default built for 32-bit integers.

If you decide to build for 32-bit integers, you will **not** be able to:

- Safely allocate more than 16 GB of memory **per cpu-core**
- Run LUCITA
- Run really, really large Coupled-Cluster calculations

If you want to be able to run the above calculations, you have to build
for 64-bit integers.

If you decide to build for 64-bit integers, you **have to** make sure
that the math library you link to can handle 64-bit integers.

Otherwise DIRAC will stop.
