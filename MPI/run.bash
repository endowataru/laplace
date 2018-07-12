#!/bin/bash

module unload pgi mpi/pgi_openmpi intel
module load pgi mpi/pgi_openmpi

mpicc -O3 -mp -acc -Minfo=accel -Mipa=fast,inline laplace_mpi.c

sbatch \
-N $2 -p GPU --ntasks-per-node 2 \
--gres=gpu:p100:2 -t 5 -A ac560tp \
--reservation=$1 \
-J challenge \
job.bash

