#!/bin/bash

module unload pgi mpi/pgi_openmpi intel
module load pgi mpi/pgi_openmpi

#mpirun -n 4 ./a.out
mpirun -n 4 -genv MV2_ENABLE_AFFINITY=0 ./a.out 2>&1 | tee $PBS_JOBNAME.$PBS_JOBID.log
