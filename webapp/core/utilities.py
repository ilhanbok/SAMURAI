# Post-segmentation curation helpers - port of scripts/Utilities.py.
#
# Same five operations (MERGE, ZERO, DUPLICATE, IRV, VOLUME) with the same
# semantics; they return values instead of printing so the web layer can render
# them, and DUPLICATE takes the output path it was always meant to take (the
# CLI version references an undefined `output_file`).

import csv
import os

import numpy as np
import nibabel as nib

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def list_nifti(directory):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(NIFTI_SUFFIXES)
    )


def merge_nifti(directory, output_file):
    """Voxel-wise logical OR of every NIfTI in ``directory``."""
    nifti_files = list_nifti(directory)
    if not nifti_files:
        raise ValueError("No NIfTI files found in the directory.")

    images = [nib.load(f) for f in nifti_files]
    data_arrays = [img.get_fdata().astype(bool) for img in images]

    shapes = {data.shape for data in data_arrays}
    if len(shapes) > 1:
        raise ValueError(
            "All input NIfTI files must have the same shape. Found shapes: {0}".format(
                sorted(shapes)
            )
        )

    merged_data = np.logical_or.reduce(data_arrays)
    nib.save(nib.Nifti1Image(merged_data.astype(np.uint8), images[0].affine), output_file)
    return {
        "output": output_file,
        "inputs": [os.path.basename(f) for f in nifti_files],
        "voxels": int(np.count_nonzero(merged_data)),
    }


def zero_slices(input_file, output_file, slice_range, axis=0):
    """Blank a slice range.

    Axis numbering is Utilities.py's: 0 blanks along the third array dimension,
    1 the second, 2 the first.
    """
    nii = nib.load(input_file)
    data = nii.get_fdata()
    start_slice, end_slice = slice_range
    if axis == 0:
        data[:, :, start_slice:end_slice + 1] = 0
    elif axis == 1:
        data[:, start_slice:end_slice + 1, :] = 0
    elif axis == 2:
        data[start_slice:end_slice + 1, :, :] = 0
    else:
        raise ValueError("Error with invalid axis.")
    nib.save(nib.Nifti1Image(data, nii.affine), output_file)
    return {"output": output_file, "voxels": int(np.count_nonzero(data))}


def duplicate(input_file, output_file, slice_source, slice_destination):
    """Copy one slice onto another (1-based indices, as in Utilities.py)."""
    nii = nib.load(input_file)
    data = nii.get_fdata()
    data[:, :, slice_destination - 1] = data[:, :, slice_source - 1]
    nib.save(nib.Nifti1Image(data, nii.affine), output_file)
    return {"output": output_file, "voxels": int(np.count_nonzero(data))}


def compute_nifti_volume(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    return float(np.count_nonzero(data) * voxel_volume)


def process_folder(folder_path, output_csv):
    """Volumes of every mask in a folder, written to CSV."""
    rows = []
    volumes = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(NIFTI_SUFFIXES):
            continue
        volume = compute_nifti_volume(os.path.join(folder_path, filename))
        volumes.append(volume)
        rows.append({"Filename": filename, "Volume (mm^3)": volume})

    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Filename", "Volume (mm3)"])
        for row in rows:
            writer.writerow([row["Filename"], row["Volume (mm^3)"]])

    return {
        "output": output_csv,
        "rows": rows,
        "mean": float(np.mean(volumes)) if volumes else 0.0,
        "std": float(np.std(volumes)) if volumes else 0.0,
    }


def irv_pair(mask1, mask2):
    """Cohen inter-rater reliability and Dice score for one pair of masks."""
    mask1_data = (nib.load(mask1).get_fdata() > 0).astype(bool)
    mask2_data = (nib.load(mask2).get_fdata() > 0).astype(bool)

    TP = int(np.sum(mask1_data * mask2_data))
    TN = int(np.sum(~mask1_data * ~mask2_data))
    FN = int(np.sum(~mask1_data * mask2_data))
    FP = int(np.sum(mask1_data * ~mask2_data))

    denominator = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    cohen = 2 * (TP * TN - FN * FP) / denominator if denominator else float("nan")
    total = np.sum(mask1_data) + np.sum(mask2_data)
    dice = 2 * TP / total if total else float("nan")
    return {
        "mask1": os.path.basename(mask1),
        "mask2": os.path.basename(mask2),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "cohen_irv": float(cohen), "dice": float(dice),
    }


def irv_directories(dir_1, dir_2):
    """Pair files by sort order across two directories, as Utilities.py does."""
    first = list_nifti(dir_1)
    second = list_nifti(dir_2)
    if not first or not second:
        raise ValueError("Both directories must contain NIfTI files.")
    if len(first) != len(second):
        raise ValueError(
            "Directories hold {0} and {1} NIfTI files; each pair must correspond.".format(
                len(first), len(second)
            )
        )
    pairs = [irv_pair(a, b) for a, b in zip(first, second)]
    values = [p["cohen_irv"] for p in pairs]
    return {
        "pairs": pairs,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "interpretation": interpret_irv(float(np.mean(values))),
    }


def interpret_irv(value):
    if value != value:  # NaN
        return "undefined"
    if value < 0.5:
        return "poor"
    if value < 0.7:
        return "good"
    return "excellent"
