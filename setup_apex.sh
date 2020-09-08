# load conda into environment
eval "$(conda shell.bash hook)"

# set CUDA_HOME (to PyTorch cuda version)
# make directories for apex
mkdir -p ~/lib && cd ~/lib
git clone https://github.com/NVIDIA/apex
cd apex
module purge
module load 2019
module load pre2019
module load CUDA/10.1.243
module load cuDNN/7.6.3-CUDA-10.1.243
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.6.3-CUDA-10.1.243/lib64:$LD_LIBRARY_PATH
module load Anaconda3


CONDA_PREFIX=$(conda info --base)
source $CONDA_PREFIX/etc/profile.d/conda.sh
# install apex
conda activate thesisp375 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
