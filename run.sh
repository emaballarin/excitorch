#!/usr/bin/bash -li
#SBATCH --job-name=excitorch
#SBATCH --mail-user=emanuele@ballarin.cc
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=DGX
#SBATCH --time=1-00:00:01           # DD-HH:MM:SS
#SBATCH --nodes=1                   # Nodes
#SBATCH --ntasks-per-node=1         # GPUs per node
#SBATCH --cpus-per-task=16          # Cores per node / GPUs per node
#SBATCH --mem=128G                  # 4 * Cores per node
#SBATCH --gres=gpu:1                # GPUs per node
################################################################################

sleep 3

export HOME="/u/dssc/s223459/"
export CODEHOME="$HOME/Downloads/excitorch/src"
export MYPYTHON="$HOME/pixies/minilit/.pixi/envs/default/bin/python"

cd $CODEHOME || exit

echo " "
echo "hostname=""$(hostname)"
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "START TIME ""$(date +'%Y_%m_%d-%H_%M_%S')"
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
srun "$MYPYTHON" -O "$CODEHOME/run.py"
echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "STOP TIME ""$(date +'%Y_%m_%d-%H_%M_%S')"
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
