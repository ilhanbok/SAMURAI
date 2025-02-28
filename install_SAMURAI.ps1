git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
cp ../sam2_video_predictor.py .
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..
pip install -r requirements.txt
git clone https://github.com/neurolabusc/niivue-images.git
cd scripts
