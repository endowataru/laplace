#!/bin/bash

mpicc -acc -Minfo=accel laplace_mpi.c

sbatch gpu.job

