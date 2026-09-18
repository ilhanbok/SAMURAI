import os
import numpy as np
import nibabel as nib

valids = ['MERGE','DUPLICATE','ZERO','IRV','VOLUME']

def print_help_message():
    print(f'Your options are: {valids}\n')
    print(f'You can print the rest of the menu info by using: `python Utilities.py <option>')

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

def print_IRVDC(mask1, mask2):
    # Load NIfTI file and mask
    mask1_img = nib.load(mask1)
    mask2_img = nib.load(mask2)
 
    # Get data arrays
    mask1_data = (mask1_img.get_fdata() > 0).astype(bool)
    mask2_data = (mask2_img.get_fdata() > 0).astype(bool)
 
    if mask1_data.shape != mask2_data.shape:
        raise ValueError('Shape mismatch: {0} vs {1}'.format(
            mask1_data.shape, mask2_data.shape))
 
    # int() forces Python's arbitrary-precision integers. np.sum() on a bool
    # array returns the platform default int, which is int32 on Windows with
    # numpy < 2.0; TP*TN then overflows (TN alone is ~6e6 for a typical
    # volume) and wraps silently, which is what produced kappa = -122.
    TP = int(np.count_nonzero(mask1_data & mask2_data))
    TN = int(np.count_nonzero(~mask1_data & ~mask2_data))
    FN = int(np.count_nonzero(~mask1_data & mask2_data))
    FP = int(np.count_nonzero(mask1_data & ~mask2_data))
 
    denom = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    if denom == 0:
        # Both raters marked every voxel the same way (all empty or all full);
        # chance agreement is 1, so kappa is undefined.
        CohenIRV = float('nan')
        print('WARNING: masks are degenerate (all-empty or all-full), '
              'Cohen kappa undefined.')
    else:
        CohenIRV = 2 * (TP * TN - FN * FP) / denom
        if not -1.0 <= CohenIRV <= 1.0:
            # Cannot happen with exact arithmetic; catches any future
            # regression to a fixed-width accumulator.
            raise ArithmeticError(
                'Cohen kappa out of range ({0}) - counts overflowed. '
                'TP={1} TN={2} FP={3} FN={4}'.format(
                    CohenIRV, TP, TN, FP, FN))
    print('Cohen Inter-rater reliability: {0}'.format(CohenIRV))
 
    mask_total = int(np.count_nonzero(mask1_data)) + \
        int(np.count_nonzero(mask2_data))
    if mask_total == 0:
        DSC = float('nan')
        print('WARNING: both masks are empty, Dice Score Coefficient '
              'undefined.')
    else:
        DSC = 2 * TP / mask_total
    print('Dice Score Coefficient: {0}'.format(DSC))
    return CohenIRV, DSC

def index_by_prefix(folder_path, prefix_len=7):
    """
    Indexes the NIfTI files in a folder by the leading characters of their
    filenames, so that files can be paired across folders even when the rest
    of the name differs (e.g. 01_1009_MANUAL.nii vs 01_1009_SAM_final.nii.gz).

    Parameters:
        folder_path (str): Path to the directory containing NIfTI files.
        prefix_len (int): Number of leading characters used as the subject key.

    Returns:
        (dict, dict): {prefix: file_path} for the first file found for each
        prefix, and {prefix: [all_paths]} for any prefix seen more than once.
    """
    index = {}
    duplicates = {}

    for filename in sorted(os.listdir(folder_path)):
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            continue

        prefix = filename[:prefix_len]
        file_path = os.path.join(folder_path, filename)

        if prefix in index:
            duplicates.setdefault(prefix, [index[prefix]]).append(file_path)
        else:
            index[prefix] = file_path

    return index, duplicates


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

def zero_slices(input_file, output_file, slice_range, axis):
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
    match axis:
        case 0:
            data[:, :, start_slice:end_slice+1] = 0
        case 1:
            data[:, start_slice:end_slice+1, :] = 0
        case 2:
            data[start_slice:end_slice+1, :, :] = 0
        case _:
            print('Error with invalid axis.')

    # Save the modified data as a new NIfTI file
    modified_nii = nib.Nifti1Image(data, nii.affine)
    nib.save(modified_nii, output_file)

import numpy as np
if __name__ == "__main__":

    args = sys.argv
    if len(args) > 1:
        type = args[1].lower()
    else:
        print_help_message()
        print('PLEASE enter a valid option!')
        quit()
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
    match type:
      case "merge":

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


      case "zero":
        print('test')
        # Ensure correct number of arguments
        if len(sys.argv) < 5 or len(sys.argv) > 6:
            print("""2. `ZERO`
- Prerequisites: Have the NIFTI file and slice indices ready. ITK-SNAP can
be used to inspect the file and determine slices to clear before script usage.
- Description:
Used to delete slices within a range for spurious segmentation output.
- Usage:
`python Utilities.py ZERO input_file output_file slice_start slice_finish (axis)`
- Example:
`python Utilities.py ZERO subject_<m>.nii.gz subject_<m>_zeroed.nii.gz 1 14 0`
- Output: A new NIFTI file with superfluous segmentation masks removed""")
        else:
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            slice_start = int(sys.argv[3])
            slice_finish = int(sys.argv[4])
            if len(sys.argv) == 6:
                axis = int(sys.argv[5])
            else:
                axis = 0

            zero_slices(input_file, output_file, (slice_start, slice_finish), axis)
            print(f"Modified NIfTI file saved to {output_file}")


    # Example usage
      case "duplicate":
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

      case "irv":
        # Ensure correct number of arguments
        if len(sys.argv) not in (3, 4, 5):
            print("""1. `IRV`
- Prerequisites: Ensure that each pair of files to be compared shares the same
leading 7 characters (e.g. `01_1009`). The rest of each filename is ignored,
so `01_1009_MANUAL.nii` and `01_1009_SAM_final.nii.gz` will be paired.
- Description:
Used to determine inter-rater variability (IRV) across a set of subjects.
- Usage:
`python Utilities.py IRV dir_1 dir_2 (output_csv) (prefix_length)`
- Example:
`python Utilities.py IRV IRV_Alice/ IRV_Bob/`
`python Utilities.py IRV IRV_Alice/ IRV_Bob/ dice_scores.csv`
- Output: The inter-rater variability between each matched pair of subjects as
well as the average and standard deviation across the matched set, followed by
a list of any subjects that could not be paired. An IRV of IRV<0.5 is considered
poor, 0.5<IRV<0.7 is good, and IRV>0.7 is excellent.
A CSV of `Prefix,Dice Score Coefficient` is also written, one row per matched
pair, defaulting to IRV_DSC.csv in the current directory.""")
        else:
            folder_path1 = sys.argv[1]
            folder_path2 = sys.argv[2]
            output_csv = 'IRV_DSC.csv'
            prefix_len = 7

            if len(sys.argv) >= 4:
                # A bare integer here still means prefix length, so the older
                # `IRV dir_1 dir_2 9` form does not silently become a filename.
                if sys.argv[3].isdigit():
                    prefix_len = int(sys.argv[3])
                    print('Read \'{0}\' as the prefix length; writing the CSV to {1}.'.format(
                        sys.argv[3], output_csv))
                else:
                    output_csv = sys.argv[3]
            if len(sys.argv) == 5:
                prefix_len = int(sys.argv[4])

            index1, dupes1 = index_by_prefix(folder_path1, prefix_len)
            index2, dupes2 = index_by_prefix(folder_path2, prefix_len)

            # Warn about ambiguous prefixes rather than silently picking one
            for folder_path, dupes in ((folder_path1, dupes1), (folder_path2, dupes2)):
                for prefix in sorted(dupes):
                    names = [os.path.basename(p) for p in dupes[prefix]]
                    print('WARNING: prefix \'{0}\' matches {1} files in {2}: {3}. Using {4}.'.format(
                        prefix, len(names), folder_path, ', '.join(names), names[0]))

            matched = sorted(set(index1) & set(index2))
            unmatched1 = sorted(set(index1) - set(index2))
            unmatched2 = sorted(set(index2) - set(index1))

            IRVres = []
            DSCres = []
            for prefix in matched:
                print('\n{0}  |  {1}  vs  {2}'.format(
                    prefix,
                    os.path.basename(index1[prefix]),
                    os.path.basename(index2[prefix])))
                irv, dsc = print_IRVDC(index1[prefix], index2[prefix])
                IRVres.append(irv)
                DSCres.append(dsc)

            # One row per matched pair, in the same order as the printout
            with open(output_csv, 'w', newline='') as dscf:
                writer = csv.writer(dscf)
                writer.writerow(['Prefix', 'Dice Score Coefficient'])
                for prefix, dsc in zip(matched, DSCres):
                    writer.writerow([prefix, dsc])
            print('\nWrote {0} ({1} row(s))'.format(output_csv, len(matched)))

            print('\n===== SUMMARY =====')
            print('Matched pairs: {0}'.format(len(matched)))
            if IRVres:
                print('Mean IRV: {0}'.format(np.nanmean(IRVres)))
                print('Std IRV: {0}'.format(np.nanstd(IRVres)))
                print('Mean DSC: {0}'.format(np.nanmean(DSCres)))
                print('Std DSC: {0}'.format(np.nanstd(DSCres)))
            else:
                print('No pairs were matched - check that the leading {0} characters agree.'.format(prefix_len))

            if unmatched1 or unmatched2:
                print('\n===== UNMATCHED =====')
                if unmatched1:
                    print('Only in {0} ({1}):'.format(folder_path1, len(unmatched1)))
                    for prefix in unmatched1:
                        print('  {0}  ->  {1}'.format(prefix, os.path.basename(index1[prefix])))
                if unmatched2:
                    print('Only in {0} ({1}):'.format(folder_path2, len(unmatched2)))
                    for prefix in unmatched2:
                        print('  {0}  ->  {1}'.format(prefix, os.path.basename(index2[prefix])))
            else:
                print('All files were matched.')


      case "volume":
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
      case _:
          print_help_message()
          quit()