# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)

    ## Test code below (1/28/2025)
    # 1. Set the model to training mode
    # model.train()

    # 2. Forward pass: Calculate the output of the model
    #output = model(input_data) # currently not needed for testing

    # 3. Calculate the loss

    #old model output
    # import nibabel as nib
    # M_out = nib.load('/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Segmentation_Results/03_1097_M.nii.gz').get_fdata()
    # delta = 1

    def downsample(image):
        m1 = nn.Conv2d(
            in_channels=1152,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        m2 = nn.Conv2d(
            in_channels=576,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        m3 = nn.Conv2d(
            in_channels=288,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        m4 = nn.Conv2d(
            in_channels=144,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        ri = torch.zeros((1,1,256,256))
        ri[0,0,:,:] = torch.tensor(image).float()
        print(m1(ri))
        image2 = m4(m3(m2(m1(image))))
        return image2


    # import pickle
    # import torch.nn as nn
    # from torch.autograd import Variable
    # for frame in range(126,190):
    #     print('Frame: {0}'.format(frame))
    #     with open('07_1064_Bifrontal_pkl_{0}.pkl'.format(frame),'rb') as f:
    #         M = pickle.load(f)
    #     with open('07_1064_Bifrontal_pkl_Manual_{0}.pkl'.format(frame),'rb') as f:
    #         NN = pickle.load(f)
    #         print(NN.shape)
    #     print(M)
    #     delta = 1
    #     loss = (NN.cpu() - M.cpu() + delta - 1)
    #     plt.imshow(loss.numpy())
    #     plt.colorbar()
    #     plt.title('Loss matrix for 07_1064')
    #     plt.show()
    #     loss = Variable(np.multiply(loss,loss),requires_grad=True)
    #
    #     # 4. Backpropagation: Calculate gradients
    #     loss.sum().backward()
    #
    #     LEARNING_RATE = 0.9 #(-1 to 1)
    #     # 5. Update the model's weights and biases using an optimizer (e.g., SGD)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    #     optimizer.step()
    #     import matplotlib.pyplot as plt
    #     # 6. Reset gradients for the next iteration
    #     optimizer.zero_grad()
    #     # plt.imshow(loss,cmap='jet')
    #     # plt.colorbar()
    #     # plt.show()
    #
    #     input('Looks good?')
    #     ## carry on, facebook

    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


import numpy as np

def sigmoid(x, slope = 1):
  return 1 / (1 + torch.exp(-slope*x))

def map_weights(tensor, thresh = 0.01, scale = 5):

    print(np.where(np.abs(tensor)<thresh))
    # Apply weights to the tensor
    tensor[np.invert(np.abs(tensor)<thresh)] = tensor[np.invert(np.abs(tensor)<thresh)]*scale
    tensor[np.abs(tensor)<thresh] = tensor[np.abs(tensor)<thresh]/scale

    # Apply sigmoid upscaling
    tensor = tensor*sigmoid(tensor,2)
    return tensor


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

        sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.0.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.0.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.1.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.1.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.weight"])

        sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.0.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.0.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.1.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.1.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.2.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.1.layers.2.weight"])

        sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.0.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.0.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.1.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.1.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.2.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.2.layers.2.weight"])

        sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.0.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.0.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.1.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.1.weight"])
        #sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.2.weight"] = map_weights(sd["sam_mask_decoder.output_hypernetworks_mlps.3.layers.2.weight"])
