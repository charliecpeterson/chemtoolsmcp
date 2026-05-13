orphan  

# DIRAC hybrid parallelization

Here we present the simple performance study of the hybrid (or mixed)
Open MPI - OpenMP parallelization.

The **Open MPI** is code's explicit parallelization, while the
**OpenMP** is implicit parallelization of the linked mathematical
library - MKL or OpenBLAS.

## Machine

We employed the Kronos cluster (sitting at the GSI.de).

The SLURM system choosed the working node:

- Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
- 40 CPU node with 126 GB of the total memory (checked via free -g -t)

## System

We use DIRAC benchmark Coupled-Cluster test and two different parallel
DIRAC installations. Both consist of the Open MPI framework, the Intel
compiler enhanced with highest optimization flag, *-xHost*, and
internally threaded mathematical library - the (open) OpenBLAS library
and the commercial MKL library.

- Open MPI 2.0.1; Intel-17 with MKL-integer8 internally-threaded(OpenMP)
  library

<!-- -->

    $PAM1  --mpi=$nOpenMPI  --gb=4.60  --ag=5.00  --noarch --inp=$DIRAC/test/benchmark_cc/cc.inp --mol=$DIRAC/test/benchmark_cc/C2H4Cl2_sta_c1.mol --suffix=i17mkl-mpi$nOpenMPI-omp$MKL_NUM_THREADS-tmp_out

- Open MPI 2.0.1; Intel-17 with OpenBLAS-integer8
  internally-threaded(OpenMP) library

<!-- -->

    $PAM2  --mpi=$nOpenMPI  --gb=4.60  --ag=5.00  --noarch --inp=$DIRAC/test/benchmark_cc/cc.inp --mol=$DIRAC/test/benchmark_cc/C2H4Cl2_sta_c1.mol --suffix=i17oblas-mpi$nOpenMPI-omp$OPENBLAS_NUM_THREADS-tmp_out

The variables --gb=MEM1 and --ag=MEM2 are to be set carefully with
respect to the total node memory and number of OpenMPI-threads. For
instance, for 24 OpenMPI threads MEM2 is max. 120/24=5GB; MEM1 is to be
lower, 4.60GB. The higher number of threads, the lower assigned memory
per thread. We have to be carefull to set the suitable value of MEM2 to
make the memory demanding job pass.

## Results

Wall times depending on number of OpenMPI (mpi) and OpenMP (omp) threads
are shown in the following Table:

> <table>
> <thead>
> <tr>
> <th>mpi</th>
> <th>omp</th>
> <th>OpenBLAS</th>
> <th>MKL</th>
> </tr>
> </thead>
> <tbody>
> <tr>
> <td>4</td>
> <td>1</td>
> <td>38min15s</td>
> <td>37min4s</td>
> </tr>
> <tr>
> <td>4</td>
> <td>10</td>
> <td>33min18s</td>
> <td>29min1s</td>
> </tr>
> <tr>
> <td>8</td>
> <td>1</td>
> <td>30min32s</td>
> <td>29min13s</td>
> </tr>
> <tr>
> <td>8</td>
> <td>5</td>
> <td>28min15s</td>
> <td>23min36s</td>
> </tr>
> <tr>
> <td>16</td>
> <td>1</td>
> <td><ul>
> <li></li>
> </ul></td>
> <td>48min59s</td>
> </tr>
> </tbody>
> </table>

## Discussion

One can see (in Table `mytableCC`) that better performance is obtained
with the Intel17-MKL compilation settings.

For this testing system, the fastest calculation is with 8 OpenMPI
threads, where for each mpi-thread there are 5 MKL omp threads by
keeping total number of CPUs,8x5=40.

Higher number of threads may be causing communication overhead and thus
performance slowdown, as we see for 16 mpi-thread.

Note that we have not splitted this compact run into separate steps
(SCF, MOLTRA, RELCCSD), where each step would have with its specific
setting for the hybrid-parallel run.
