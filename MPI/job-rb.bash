#!/bin/bash

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module unload intel/18.1.163 intel-mpi/2018.1.163
module load pgi mvapich2/mxm/2.2/pgi cuda9/9.1.85

mpirun -n 4 ./a.out 2>&1 | tee $PBS_JOBNAME.$PBS_JOBID.log

