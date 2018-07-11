#!/bin/bash

mpicc -acc -Minfo=accel laplace_mpi.c

sbatch \
-N 1 -p GPU-shared --ntasks-per-node 1 \
--gres=gpu:p100:1 -t 5 -A ac560tp \
--reservation=$1 \
-J challenge \
job.bash

