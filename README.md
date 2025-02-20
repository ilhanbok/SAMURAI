# SAMURAI (**S**egment **A**nything **M**odel **U**tilized for **R**adiomic **A**nalysis of **I**mages)
<br>
<p align="center">
<img src="logo/Samurai_Helmet_Icon-2.png" width="200" center>


## Introduction

Our abstract:
> *Machine Learning Segmentation of Magnetic Resonance Images for Radiomics of Traumatic Brain Injury Hematomata*, Neurotrauma 2025 (Philadelphia, PA)

Adaptation of SAM2 for segmentation of brain hematomas.

**NOTE:** This software is not intended for clinical use.

## Installation

```bash
git clone https://github.com/ilhanbok/SAMURAI.git
cd SAMURAI
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install -e .
mv ../sam2_video_predictor.py .
cd checkpoints
./download_ckpts.sh
cd ../../scripts
pip install -r requirements.txt
```
Example open-access nifti files are here:
https://github.com/neurolabusc/niivue-images.git

```bash
cd ..
git clone https://github.com/neurolabusc/niivue-images.git
cd scripts
```

## Usage

There are two modes of operation: (i) Manual (GUI-based) selection and (ii) threshold-based lesion identification.

For DICOM files use a conversion tool such as [dcm2niix](https://github.com/rordenlab/dcm2niix) to create input NIFTI file.

To use mode (i);
- Run `python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py <nii file name>`
- Click on the points within the frame which you believe to be either intra-lesional (fg:1) or extra-lesional (bg:0).
- Follow the commands given on-screen.
- The segmented lesion will be saved to a `.nii.gz` file with the lesion ID as part of the name.

Video tutorial:<br>
https://www.youtube.com/watch?v=8VA3K332Exo
To use mode (ii):
- Usage: `python thresholdSAM_01062025.py NIFTI_PATH output_csv(.csv) mode(QSM or RAW) threshold(e.g.400) flip(lesions are dark?) blur(gaussian blur to hide artifacts?) skullstrip(strip skull or not?)`
- There will be a loading bar and a GUI dictating whether you want to `accept` each lesion or not. If you decline, that lesion will be skipped and the program will iterate through all additional lesions

## About/Contact Information

Patrick Belton (belton@neurosurgery.wisc.edu)<br>
Ilhan Bok (ibok@wisc.edu)

Department of Neurological Surgery<br>
School of Medicine and Public Health<br>
University of Wisconsin - Madison
