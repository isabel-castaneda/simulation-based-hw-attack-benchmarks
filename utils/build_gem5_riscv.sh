#!/bin/bash
#SBATCH --job-name=build_gem5
#SBATCH --output=build_gem5.out
#SBATCH --error=build_gem5.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

cd gem5

scons build/RISCV/gem5.opt -j$SLURM_CPUS_PER_TASK
