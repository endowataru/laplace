#!/bin/bash

module unload pgi mpi/pgi_openmpi intel
module load pgi mpi/pgi_openmpi

mpirun -n 4 ./a.out

