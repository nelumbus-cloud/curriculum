#!/bin/bash
#SBATCH --job-name=test_mail
#SBATCH --partition=research-gpu
#SBATCH --time=00:05:00
#SBATCH --mem=1G

#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

echo "Hello from Slurm"
sleep 30
