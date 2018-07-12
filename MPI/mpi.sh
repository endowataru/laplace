#!/bin/sh
#SBATCH -N 4
#SBATCH -p RM
#SBATCH --res=ihssrm
#SBATCH --account ac560tp
#SBATCH -t 5

mpirun -np 4 ./a.out
