#!/bin/bash
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --job-name=embc_tgfnn_cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=10GB
#SBATCH --time=10:00:00
#SBATCH --account=
#SBATCH --partition=standard

module load python3.9
source env/bin/activate
python -W ignore main_cv.py -i $1 -n $2 --n_folds $3 --n_folds_hyper_tuning $4 --search_iters $5