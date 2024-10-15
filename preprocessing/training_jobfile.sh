#!/bin/bash -l                 
#
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
                                 
#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
            
module load python/3.8-anaconda cuda/11.8.0 cudnn/8.8.0.121-11.8

cd ${HOME}/cellpose_src
conda activate cellpose_git

python -m cellpose --train --dir ./dataset/calculated/manual_dataset/train --pretrained_model ./dataset/calculated/closest_vertex_all_degree_120/train/models/cellpose_confocal_calculated_all_deg_cyto1719325560.4115765 --chan 2 --chan2 1 --weight_decay 0.0001 --n_epochs 50 --mask_filter _masks --verbose --use_gpu --batch_size 16
