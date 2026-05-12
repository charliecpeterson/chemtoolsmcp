orphan  

# Profiling how to

This is a quick and dirty guide to profiling. You are **strongly**
recommended to read the profiling software documentation to get deeper
insight in the commands suggested in the following.

## Intel VTune Amplifier XE

This software is a nice tool for profiling and has a GUI working also on
Linux. Moreover, you will not need to compile your program with
profiling flags on. The only bad thing is that it is commercial
software. Profiling takes two steps: 1. run your program under VTune; 2.
analyze the collected data.

Assuming your program is called `foobar`, and further assuming that you
only want to collect the *hotspots*, for step 1 you will launch:

    amplxe-cl -collect hotspot -r /results/directory ./foobar

the results of the sampling will be put in `/results/directory`. If the
results directory is not specified VTune will put everything in a
subdirectory of the current directory. For step 2:

    amplxe-cl -report hotspots -r /results/directory > report_name

this will produce a hotspots report called `report_name`.

## gprof

`gprof` is free software. It works a bit differently from VTune since we
need to explictly compile our code with the `-pg` flag. This step is
needed in order to link correctly the profiling library to the
executable that will be produced. Once the program is correctly compiled
you just need to execute it without specifying `gprof` as launcher as is
the case for VTune:

    ./foobar [--flag1 --flag2 ...] [input1 input2 ...]

your program will automatically create a file, called `gmon.out`
containing the profiling information collected. This file will by
default sit in the current working directory. After data collection you
need to generate a human-readable profile summary. We run then the
`gprof` command:

    gprof options [executable-file [profile-data-files ...]] [>outfile]

this will produce a so-called flat profile from the raw data collected.
A more detailed introduction to `gprof` can be found here:
[gprof](http://www.cs.utah.edu/dept/old/texinfo/as/gprof.html#SEC2).

## Profiling DIRAC with Intel VTune Amplifier XE

The situation is a bit different if you want to profile DIRAC, because
we usually launch the executable through a script. If you are using the
`wrapper.py` script, you will specify as launcher:

    --launcher="amplxe-cl -collect hotspot -r /results/directory"

the rest of the procedure remains the same. What if we want to profile
during an MPI run? We modify the launcher as follows:

    --launcher="mpirun -np 12 amplxe-cl -collect hotspots -follow-child -mrte-mode=auto -target-duration-type=medium -no-allow-multiple-runs -no-analyze-system -data-limit=100 -slow-frames-threshold=40 -fast-frames-threshold=100 -r /results/directory"

Some more words of comment (shamelessly copied from
`amplxe-cl -help collect`):

- `follow-child`, collects data on processes launched by the target
  process;
- `mrte-mode`, selects the profiling mode;
- `target-duration-type`, estimates the application duration time. This
  value affects the size of collected data;
- `no-allow-multiple-runs`, disables multiple runs to achieve more
  precise results for hardware event-based collections;
- `no-analyze-system`, disables analyzing all processes running on the
  system;
- `data-limit`, limits the amount of raw data to be collected;
- `slow-frames-threshold`, specifies a threshold to separate slow and
  good frames. It must be smaller than the threshold for fast frames;
- `fast-frames-threshold`, specifies the threshold to separate good and
  fast frames.

## Profiling DIRAC with gprof

In contrast with VTune using `gprof` to profile DIRAC is rather
straightforward. The only thing you will need to do is to link the
profiling library. Thus if using the `setup` script you will type:

    ./setup --fc=... --cc=... --cxx=... --profiling --release

we recommend to build in release mode if you want to have your collected
profiling data to be really significant. Once you managed to compile the
sources correctly, you just run DIRAC as you're used to using `pam` or
`wrapper.py`. The program itself will produce in your current working
directory the `gmon.out` file and you will translate it to
human-readable form with `gprof` as explained before.

One word of caution for MPI runs. To avoid all the processes trying to
write to the same `gmon.out` file you should export the
`GMON_OUT_PREFIX` environment variable:

    export GMON_OUT_PREFIX=foobarmon

and pass it to `mpirun`:

    mpirun -x GMON_OUT_PREFIX -np <np> ./foobar

In this way you will have a series of files named `GMON_OUT_PREFIX.pid`,
post-collection analysis works exactly as for non-parallel runs. One
warning from the `mpirun` manual: "\*Users are advised to set variables
in the environment, and then use -x to export (not define) them.\*"
