import os
import numpy as np
import nibabel as nib

def merge_nifti(directory, output_file):
    """
    Merges all NIfTI files in a directory by performing a voxel-wise logical OR operation.

    Parameters:
        directory (str): Path to the directory containing NIfTI files.
        output_file (str): Path to save the merged NIfTI file.

    Returns:
        None
    """
    # Get all NIfTI files in the directory
    nifti_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nii") or f.endswith(".nii.gz")]

    if not nifti_files:
        raise ValueError("No NIfTI files found in the directory.")

    # Load the NIfTI files and perform logical OR on their data
    images = [nib.load(f) for f in nifti_files]
    data_arrays = [img.get_fdata().astype(bool) for img in images]

    # Ensure all images have the same shape
    shapes = [data.shape for data in data_arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"All input NIfTI files must have the same shape. Found shapes: {shapes}")

    # Perform voxel-wise logical OR
    merged_data = np.logical_or.reduce(data_arrays)

    # Use the affine of the first image (assuming consistent spatial orientation)
    affine = images[0].affine

    # Save the merged data as a new NIfTI file
    merged_img = nib.Nifti1Image(merged_data.astype(np.uint8), affine)
    nib.save(merged_img, output_file)

    print(f"Merged NIfTI file saved as {output_file}")

import nibabel as nib
import numpy as np
import os
import csv

def compute_nifti_volume(nifti_path):
    # Load the NIfTI file
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Get voxel dimensions from header
    voxel_dims = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)

    # Count nonzero voxels (assumed to be part of the mask)
    num_voxels = np.count_nonzero(data)

    # Compute total volume
    total_volume = num_voxels * voxel_volume

    return total_volume

def process_folder(folder_path, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Volume (mm³)"])
        volumes = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                file_path = os.path.join(folder_path, filename)
                volume = compute_nifti_volume(file_path)
                volumes.append(volume)
                writer.writerow([filename, volume])
                print(f"Processed {filename}: {volume:.2f} mm³")
        print('Mean: {0}'.format(np.mean(volumes)))
        print('Std: {0}'.format(np.std(volumes)))

# Compute Cohen's IRV between two masks
import nibabel as nib
import numpy as np
import os

def print_IRVDC(mask1,mask2):
    # Load NIfTI file and mask
    mask1_img = nib.load(mask1)
    mask2_img = nib.load(mask2)

    # Get data arrays
    mask1_data = (mask1_img.get_fdata() > 0).astype(bool)
    mask2_data = (mask2_img.get_fdata() > 0).astype(bool)

    # As a sanity check, try setting one of the two arrays to all zeros
    #mask2_data = np.zeros(mask2_data.shape).astype(bool)

    TP = np.sum(mask1_data*mask2_data)
    TN = np.sum(~mask1_data*~mask2_data)
    FN = np.sum(~mask1_data*mask2_data)
    FP = np.sum(mask1_data*~mask2_data)
    print(TP)
    CohenIRV = 2*(TP*TN  -  FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
    print('Cohen Inter-rater reliability: {0}'.format(CohenIRV))

    DSC = 2*(TP)/(np.sum(mask1_data)+np.sum(mask2_data))
    print('Dice Score Coefficient: {0}'.format(DSC))
    return CohenIRV

import nibabel as nib
import numpy as np

def duplicate(input_file, slice, slice_send):
    """
    Sets the specified slices of a NIfTI image to zero.

    Parameters:
        input_file (str): Path to the input NIfTI file.
        output_file (str): Path to save the modified NIfTI file.
        slice_range (tuple): Range of slices to set to zero (inclusive).

    Returns:
        None
    """
    # Load the NIfTI file
    nii = nib.load(input_file)
    data = nii.get_fdata()

    data[:, :, slice_send-1] = data[:, :, slice-1]

    # Save the modified data as a new NIfTI file
    modified_nii = nib.Nifti1Image(data, nii.affine)
    nib.save(modified_nii, output_file)

import nibabel as nib
import numpy as np
import sys

def zero_slices(input_file, output_file, slice_range):
    """
    Sets the specified slices of a NIfTI image to zero.

    Parameters:
        input_file (str): Path to the input NIfTI file.
        output_file (str): Path to save the modified NIfTI file.
        slice_range (tuple): Range of slices to set to zero (inclusive).

    Returns:
        None
    """
    # Load the NIfTI file
    nii = nib.load(input_file)
    data = nii.get_fdata()

    # Get the slice range and set the specified slices to zero
    start_slice, end_slice = slice_range
    data[:, :, start_slice:end_slice+1] = 0

    # Save the modified data as a new NIfTI file
    modified_nii = nib.Nifti1Image(data, nii.affine)
    nib.save(modified_nii, output_file)

import numpy as np
if __name__ == "__main__":

    args = sys.argv
    type = args[1]
    args = [sys.argv[0]]
    for i in range(2,len(sys.argv)):
        args.append(sys.argv[i])
    sys.argv = args
    # Example usage
    #if __name__ == "__main__":
    #input_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology.nii.gz"
    #output_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology(1).nii.gz"
    #slice_range = (0, 179)  # Slices 125 to 0

    # Example usage
    if type == "MERGE":

        if len(sys.argv) != 3:
            print("""1. `MERGE`
- Prerequisites: Ensure that NIFTI files all belong to the same subject and
have been placed in the same folder `input_dir`
- Description: Combine multiple NIFTI files into a single NIFTI file.
- Usage:
`python Utilities.py MERGE input_dir output_file`
- Example:
`python Utilities.py MERGE combined_lesions subject_<m>.nii.gz`
- Output: A NIFTI file containing a logical union of all input files""")
        else:
            input_dir = sys.argv[1]
            output_file = sys.argv[2]

            merge_nifti(input_dir, output_file)
            print(f"Merged NIfTI file saved to {output_file}")


    if type == "ZERO":
        # Ensure correct number of arguments
        if len(sys.argv) != 5:
            print("""2. `ZERO`
- Prerequisites: Have the NIFTI file and slice indices ready. ITK-SNAP can
be used to inspect the file and determine slices to clear before script usage.
- Description:
Used to delete slices within a range for spurious segmentation output.
- Usage:
`python Utilities.py ZERO input_file output_file slice_start slice_finish`
- Example:
`python Utilities.py ZERO subject_<m>.nii.gz subject_<m>_zeroed.nii.gz 1 14`
- Output: A new NIFTI file with superfluous segmentation masks removed""")
        else:
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            slice_start = int(sys.argv[3])
            slice_finish = int(sys.argv[4])

            zero_slices(input_file, output_file, (slice_start, slice_finish))
            print(f"Modified NIfTI file saved to {output_file}")


    # Example usage
    if type == "DUPLICATE":
        # input_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology(affine).nii.gz"
        # output_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology(affine).nii.gz"
        # slice_to_propagate = 140
        # slice_to_send_to = 141

        if len(sys.argv) != 5:
            print("""3. `DUPLICATE`
- Prerequisites: Have the slice indices ready for input. ITK-SNAP can
be used to inspect the file and determine which slice to duplicate and which
layer to duplicate to before script usage.
- Description:
Used to duplicate a slice from one layer to another within an input NIFTI
- Usage:
`python Utilities.py DUPLICATE input_file output_file slice_source slice_destination`
- Example:`python Utilities.py DUPLICATE subject_<m>.nii.gz subject_<m>_cloned.nii.gz 13 14`
- Output: A new NIFTI file with requested changes made""")
        else:
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            slice_to_propagate = int(sys.argv[3])
            slice_to_send_to = int(sys.argv[4])

            duplicate(input_file, output_file, slice_to_propagate, slice_to_send_to)
            print(f"Modified NIfTI file saved to {output_file}")


    # mask1 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Segmentation_Results_Manual/006_1005_MANUAL_streak_corrected'
    # mask2 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Segmentation_Results/06_1005_Segmentation_01222025_Done.nii'
    # mask3 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/06_1005_MERGED_03032025.nii'
    #
    # print('MAN/SAM1')
    # print_IRVDC(mask1,mask2)
    # print('MAN/SAM2')
    # print_IRVDC(mask1,mask3)
    # print('SAM1/SAM2')
    # print_IRVDC(mask3,mask2)
    #
    # mask1 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/test_SH_SAMURAI/SAMURAI/scripts/07_1064_MANUAL_subdural+contusion.nii'
    # mask2 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Segmentation_Results/07_1064_Segmentation_Final_01302025.nii'
    # mask3 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/07_1064_MERGED_03042025.nii'
    #
    # print('MAN/SAM1')
    # print_IRVDC(mask1,mask2)
    # print('MAN/SAM2')
    # print_IRVDC(mask1,mask3)
    # print('SAM1/SAM2')
    # print_IRVDC(mask3,mask2)
    #
    # mask1 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/07_1064_SNAKE(cont)+SAMURAI(subd)_03042025.nii'
    # mask2 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/07_1064_MERGED_03042025.nii'
    # print('MAN1/MAN2')
    # print_IRVDC(mask1,mask2)

    # IRVs = [
    # '01_1062',
    # '04_1112',
    # '07_1064',
    # '07_1096',
    # '02_1200',
    # '03_1181',
    # '06_1005',
    # '07_1084',
    # '03_1037',
    # '07_1302',
    # '01_1021',
    # '03_1107',
    # '03_1125',
    # '03_1175_L',
    # '03_1175_R',
    # '03_1175_IPH',
    # '03_1183_IHF',
    # '03_1183_IPH',
    # '07_1391_BH',
    # '10_1137'
    # ]

    if type == "IRV":
        # Ensure correct number of arguments
        if len(sys.argv) != 3:
            print("""1. `IRV`
- Prerequisites: Ensure that each pair of files used for this script in each directory
have the same name
- Description:
Used to determine inter-rater variability (IRV) across a set of subjects.
- Usage:
`python Utilities.py IRV dir_1 dir_2`
- Example:
`python Utilities.py IRV IRV_Alice/ IRV_Bob/`
- Output: The inter-rater variability between each pair of subjects as well as
the average and standard deviation across the whole set of subjects. An IRV of IRV<0.5 is considered poor, 0.5<IRV<0.7 is good, and IRV>0.7 is excellent.""")
        else:
            folder_path1 = sys.argv[1]
            folder_path2 = sys.argv[2]

            IRVres = []

            SAMS = []
            MANUALS = []
            #folder_path1 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Bok_Nifti/Snake_assisted_manual/Manual/'
            #folder_path2 = '/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Bok_Nifti/SAM_assisted/SAM/'
            for filename in sorted(os.listdir(folder_path1)):
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    file_path = os.path.join(folder_path1, filename)
                    SAMS.append(file_path)
            for filename in sorted(os.listdir(folder_path2)):
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    file_path = os.path.join(folder_path2, filename)
                    MANUALS.append(file_path)

            print((MANUALS))
            print((SAMS))
            for i in range(len(MANUALS)):
                print('SAM/MANUAL')
                IRVres.append(print_IRVDC(SAMS[i],MANUALS[i]))

            print(np.mean(IRVres))
            print(np.std(IRVres))


    if type == "VOLUME":
        # Ensure correct number of arguments
        if len(sys.argv) != 3:
            print("""2. `VOLUME`
- Prerequisites: Ensure all files for analysis are placed in a single directory
- Description: Computes the volumes of NIFTI masks
- Usage:
`python Utilities.py VOLUME subject_folder output_csv`
- Example:
`python Utilities.py VOLUME find_volumes/ volumes.csv`
- Output: A list of estimated volumes for the subjects in the input directory""")
        else:
            subject_folder = sys.argv[1]
            output_csv = sys.argv[2]

            process_folder(subject_folder, output_csv)
