#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --nodes=1
#SBATCH --tasks=40
#SBATCH --cpus-per-task=1
#SBATCH --error=/srv/home/bguo23/cs759/git_homework/CS759_final_proj/code/p1.err
#SBATCH --output=/srv/home/bguo23/cs759/git_homework/CS759_final_proj/code/example.out
#SBATCH --gres=gpu:1
cd $SLURM_SUBMIT_DIR
module load cuda;
./lfe_stack_gpu sac_list/34901_list_1000 res/example.pws.sac res/example.lin.sac 
