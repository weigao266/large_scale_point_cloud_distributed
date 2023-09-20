echo "[FPT INFO] Creating $1..."
conda create -n $1 python=3.9 -y
conda activate "$1"
echo "[FPT INFO] Done."

python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

conda install openblas=0.3.4 -c conda-forge 
conda install ninja==1.11.0 -c conda-forge
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
sudo python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas


echo "[FPT INFO] Installing other dependencies..."
conda install openblas-devel -c anaconda -y
conda install -c anaconda pandas scipy h5py scikit-learn -y
conda install -c conda-forge plyfile pytorch-lightning wandb wrapt gin-config rich einops -y
python -m pip install torchmetrics==0.8.2
conda install -c open3d-admin -c conda-forge open3d -y
python -m pip install lightning-bolts
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
echo "[FPT INFO] Done."


echo "[FPT INFO] Installing cuda_ops..."
cd src/cuda_ops
python -m pip install .
cd ../..
echo "[FPT INFO] Done."

TORCH="$(python -c "import torch; print(torch.__version__)")"
ME="$(python -c "import MinkowskiEngine as ME; print(ME.__version__)")"

echo "[FPT INFO] Finished the installation!"
echo "[FPT INFO] ========== Configurations =========="
echo "[FPT INFO] PyTorch version: $TORCH"
echo "[FPT INFO] MinkowskiEngine version: $ME"
echo "[FPT INFO] ===================================="
