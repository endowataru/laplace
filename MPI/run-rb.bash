#!/bin/bash

set -eu

module unload intel/18.1.163 intel-mpi/2018.1.163
module load pgi mvapich2/mxm/2.2/pgi cuda9/9.1.85

mpicc -acc -Minfo=accel laplace_mpi.c

qsub -q h-short -l select=4:mpiprocs=1 \
    -W group_list=$1 -l walltime=00:10:00 \
    -N challenge \
    job-rb.bash

