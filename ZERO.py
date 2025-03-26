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
    data[:, :, start_slice-1:end_slice] = 0

    # Save the modified data as a new NIfTI file
    modified_nii = nib.Nifti1Image(data, nii.affine)
    nib.save(modified_nii, output_file)

# Example usage
#if __name__ == "__main__":
#input_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology.nii.gz"
#output_file = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/Segmentation_smaller_lesions/07_1391_2wk_radiology(1).nii.gz"
#slice_range = (0, 179)  # Slices 125 to 0

if __name__ == "__main__":
    # Ensure correct number of arguments
    if len(sys.argv) != 5:
        print("""2. ZERO.py
        Description:
        Used to delete slices within a range for spurious segmentation output.

        Usage:
        python ZERO.py input_file output_file slice_start slice_finish
        Example:
        python ZERO.py subject_<m>.nii.gz subject_<m>_zeroed.nii.gz 0 14""")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        slice_start = int(sys.argv[3])
        slice_finish = int(sys.argv[4])

    zero_slices(input_file, output_file, (slice_start, slice_finish))
    print(f"Modified NIfTI file saved to {output_file}")
