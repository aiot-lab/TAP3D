
# 0. you should be in a python environment with python=3.9

# 1. prepare compilation tools
python -m pip install --upgrade pip wheel ninja
python -m pip install "setuptools<81"

# 2. install Pytorch that match the version CUDA 12.4
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. ensure PyTorch is correctly installed
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# 4. install requirements. disable build isolation
MAX_JOBS=4 python -m pip install \
  --no-build-isolation \
  -r requirements.txt

# 5. install PyTorch3D. disable build isolation
MAX_JOBS=4 python -m pip install \
  --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"
