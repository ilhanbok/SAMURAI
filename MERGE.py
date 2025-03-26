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

# Example usage
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("""1. MERGE.py
        Description: Combine multiple NIFTI files into a single NIFTI file.

        Usage:
        python MERGE.py input_dir output_file
        Example:
        python MERGE.py scripts/combined_lesions subject_<m>.nii.gz""")
    else:
        input_dir = sys.argv[1]
        output_file = sys.argv[2]

    merge_nifti(input_dir, output_file)
    print(f"Merged NIfTI file saved to {output_file}")
