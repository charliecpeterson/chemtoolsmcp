orphan  

# Known problems

## Modules not running in parallel

- RELADC is not parallelized

## Modules not running with 32 bit integers

- LUCITA
- VERY large Coupled-Cluster calculations

## gfortran on Apple MX (X= 1,2,3,..)

- The default optimization level 3 gives wrong results for ExaCorr runs
  with the TALSH library. See issue \#155 for updates on this.
