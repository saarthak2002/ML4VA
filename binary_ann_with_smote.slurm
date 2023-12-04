#!/bin/bash
#SBATCH -p bii-gpu # partition/queue
#SBATCH --nodes=1		# number of compute nodes
#SBATCH --ntasks=1		# number of program instances
#SBATCH --cpus-per-task=8       # use 1 cpu core
#SBATCH --time=24:00:00		# max time before job cancels
#SBATCH --mem=350GB                   # memory
#SBATCH --gres=gpu:v100:4    # GPUs

module purge
module load anaconda/2020.11-py3.8
module load singularity tensorflow/2.10.0
singularity run --nv $CONTAINERDIR/tensorflow-2.10.0.sif binary_ann_with_smote.py