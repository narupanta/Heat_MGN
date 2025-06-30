#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=groundtruth_gen
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere


singularity exec --nv /home/y0113799/container/lpbf.sif python -u groundtruth_gen.py