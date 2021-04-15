#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J OcrDb              # The job name
#SBATCH -o ./log/ret-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ./log/ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Needed resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu-hw          # Submit to 'nv-gpu' and 'nv-gpu-hw' Partitiion
#SBATCH -t 0-12:30:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:8                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --constraint="Volta|RTX8000" # Request GPU Type: Volta(V100 or V100S) or RTX8000

###
### The system will alloc 8 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K        # Request K cores
###

#SBATCH --qos=gpu-normal               # Request QOS Type

#- Operstions
echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh

##- tools
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424
module load slurm-tools

##- language


##- cuda

module load cuda-cudnn/10.1-7.6.5

##- virtualenv
# source xxxxx/activate
conda activate open-mmlab

#- Log information

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(nvcc -V)
nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit
echo "Use GPU ${CUDA_VISIBLE_DEVICES}$"
#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script

#- Job step
# sleep 3600
cd /lustre/S/wangxianzhuo/wxz/ocr/mmocr
./tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py work_dir 8
#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
