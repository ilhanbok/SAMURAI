conda create -n "venv_SAMURAI"
conda activate venv_SAMURAI
conda install pip
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install -e .
cp ../sam2_video_predictor.py .
cd checkpoints
./download_ckpts.sh
cd ../..
pip install -r requirements.txt
