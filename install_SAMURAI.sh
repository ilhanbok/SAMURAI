#!/bin/bash

git clone https://github.com/facebookresearch/sam2.git \
&& pip install -r requirements.txt \
&& cd sam2 \
&& git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 \
&& cp ../sam2_video_predictor.py ./sam2/ \
&& pip install -e . \
&& cd checkpoints \
&& ./download_ckpts.sh \
&& cd ../.. \
&& git clone https://github.com/neurolabusc/niivue-images.git \
&& cd scripts
