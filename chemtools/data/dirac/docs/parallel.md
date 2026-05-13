orphan  

<div class="index">

\*PARALLEL

</div>

# \*PARALLEL

Parallelization directives.

<div class="index">

.PRINT

</div>

## .PRINT

Print level for the parallel calculation. Default:

    .PRINT
     0

A print level of at least 2 is needed in order to be able to evaluate
the parallelization efficiency. A complete timing for all nodes will be
given if the print level is 4 or higher.

<div class="index">

.NTASK

</div>

## .NTASK

Number of tasks to send to each node when distributing the calculation
of two-electron integrals. Default:

    .NTASK
     1

A task is defined as a shell of atomic integrals, a shell being an input
block. One may therefore increase the number of shells given to each
node in order to reduce the amount of communication. However, the
program uses dynamical allocation of work to each node, and thus this
option should be used with some care, as too large tasks may cause the
dynamical load balancing to fail, giving an overall decrease in
efficiency. The parallelization is also very coarse grained, so that the
amount of communication seldom represents any significant problem.
