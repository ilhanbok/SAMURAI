# SAMURAI (**S**egment **A**nything **M**odel **U**tilized for **R**adiomic **A**nalysis of **I**mages)
<br>
<p align="center">
<img src="logo/Samurai_Helmet_Icon-2.png" width="200" center>

## Introduction

This repository provides easy access to a hematoma segmentation software wrapper for [Meta (Facebook Research) SAM 2](https://github.com/facebookresearch/sam2/tree/main). SAMURAI has been developed within the Department of Neurological Surgery at the University of Wisconsin - Madison School of Medicine and Public Health in conjunction with the Transforming Research and Clinical Knowledge in Traumatic Brain Injury (TRACK-TBI) consortium.

Please cite the following if you are using SAMURAI:

     Machine learning-assisted segmentation of traumatic brain
     injury-induced hematomas: a TRACK-TBI study,
     Neurotrauma 2025 (Philadelphia, PA)

SAMURAI offers significant advantages compared to other methods of segmentation:
- SAMURAI has been tested and validated using data from the TRACK-TBI consortium on multiple MRI images of traumatic hematomas including multiple susceptibilty-weighted (SWI, SWAN, GRE) sequences. SAMURAI is built upon [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2/tree/main) and may work with other MRI sequences as well (try it out!).
- SAMURAI comes pre-trained and operates well with user input through an interactive guided user-friendly GUI and terminal prompt system.
- SAMURAI works with MR images unlike [BLAST-CT](https://github.com/biomedia-mira/blast-ct/tree/master)

**NOTE:** This software is not intended for clinical use.

## System requirements

### Minimum requirements
```
16 GB RAM
300 GB Disk Space
10 Cores
macOS or Windows 11
```

### Recommended requirements
```
36 GB RAM
1 TB Disk Space
14 Cores
macOS
```

## Installation

For macOS:
```bash
git clone https://github.com/ilhanbok/SAMURAI.git && cd SAMURAI && ./install_SAMURAI.sh
```

For Windows:<br><br>
First install Anaconda: https://www.anaconda.com/download<br>
Then run the following in Anaconda Powershell:
```bash
git clone https://github.com/ilhanbok/SAMURAI.git
cd SAMURAI
./install_SAMURAI.ps1
```

This should take about 5 minutes

Example open-access nifti files are here:
https://github.com/neurolabusc/niivue-images.git

For both:
```bash
git clone https://github.com/neurolabusc/niivue-images.git
cd scripts
```

Optional: Install [HD-BET](https://github.com/MIC-DKFZ/HD-BET) for skull stripping
```
pip install hd-bet
```

## Usage

There are two modes of operation: (i) Manual (GUI-based) selection and (ii) threshold-based lesion identification.

For DICOM files use a conversion tool such as [dcm2niix](https://github.com/rordenlab/dcm2niix) to create input NIFTI file.

Before execution, ensure all pip and anaconda environments are deactivated.

First run:
`source ../venvs/venv_SAMURAI/bin/activate`<br>

To use mode (i);
- Run:
`python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py ../niivue-images/chris_t1.nii.gz` or your nifti file of choice<br>
- Click on the points within the frame which you believe to be either intra-lesional (fg:1) or extra-lesional (bg:0).
- Enter the number of intra- and extra-lesional points. e.g. `4 2`
- Segmentation will proceed automatically
- The final mask will be displayed and can be edited as desired
- The segmented landmark will be saved to `scripts/output_SAMURAI.nii.gz`

Video tutorial:<br>
https://www.youtube.com/watch?v=8VA3K332Exo

To use mode (ii):
- Usage: `python thresholdSAM_01062025.py NIFTI_PATH output_csv(.csv) mode(QSM or RAW) threshold(e.g.400) flip(lesions are dark? [0/1]) blur(gaussian blur σ to hide artifacts? [0/1]) skullstrip(strip skull or not? [0/1])`
- For example: `python thresholdSAM_01062025.py ../niivue-images/chris_t1.nii.gz chris_t1 RAW 100 1 0 0`
- After iterating over possible lesions, you have the option to inspect and Accept (`Si`) each lesion or not.
- As for mode (i), there will be a pop-up point selection window to add additional foreground / background points before a final confirmation dialog
- Each lesion has a unique identification number which will be appended to its output NIFTI file. These files can be merged and viewed as a final .nii mask.

## About/Contact Information

Patrick Belton (belton@neurosurgery.wisc.edu)<br>
Ilhan Bok (ibok@wisc.edu)

Department of Neurological Surgery<br>
School of Medicine and Public Health<br>
University of Wisconsin - Madison
