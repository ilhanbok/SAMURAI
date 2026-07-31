# Threshold-based lesion identification - mode (ii).
#
# Headless port of scripts/RAW_lesion_threshAndFilter_wrapper.py and
# scripts/QSM_lesion_threshAndFilter_wrapper.py, plus the lesion book-keeping
# that scripts/thresholdSAM_01062025.py and the *_NoShow wrapper perform
# between the threshold pass and SAM2.

import csv
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generate_binary_structure, label

# Volume gates from Adjunct_SAM_lesion_threshAndFilter_wrapper_01082025_NoShow.py
VOLUME_THRESHOLD_MIN = 10.0
VOLUME_THRESHOLD_MAX = 1e6

# Defaults per imaging mode, from the two desktop wrappers.
MODE_DEFAULTS = {
    "RAW": {"cutoff": 400.0, "zero_slices_below": 0},
    # The QSM wrapper hard-codes a 0.3 cutoff and blanks z < 50 before labelling.
    "QSM": {"cutoff": 0.3, "zero_slices_below": 50},
}


def sparsify(array, ratio, seed=42):
    """Keep a reproducible random fraction of the rows of ``array``.

    Verbatim behaviour from the desktop wrapper (arrays of 3 or fewer rows are
    returned untouched), used to turn a lesion's voxels into SAM2 fg prompts.
    """
    if len(array) <= 3:
        return array
    if not (0 < ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1 (exclusive).")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(array))
    retained_count = int(np.ceil(len(array) * ratio))
    retained = rng.choice(indices, size=retained_count, replace=False)
    return array[np.sort(retained)]


def skull_strip_image(nifti_path, output_path, fractional_intensity=0.1):
    """Run FSL's ``bet``, as the desktop wrappers do."""
    if shutil.which("bet") is None:
        raise RuntimeError(
            "Skull stripping needs FSL's 'bet' on PATH; it was not found. "
            "Untick skull-strip or install FSL."
        )
    subprocess.run(
        ["bet", nifti_path, output_path, "-m", "-f", str(fractional_intensity)], check=True
    )
    return output_path


def run_threshold_pipeline(
    nifti_path,
    workdir,
    output_csv="RAW",
    mode="RAW",
    cutoff=None,
    flip=False,
    sigma=0.0,
    skull_strip=False,
    zero_slices_below=None,
    save_intermediates=True,
    progress=None,
):
    """Threshold, label and catalogue candidate lesions.

    Returns a dict with the labelled array, the rows written to the CSV and the
    paths of every file produced. Intermediates are written into ``workdir``
    rather than beside the input, which is the only deviation from the desktop
    scripts (a browser upload has no meaningful source directory).
    """
    mode = (mode or "RAW").upper()
    defaults = MODE_DEFAULTS.get(mode, MODE_DEFAULTS["RAW"])
    cutoff = float(defaults["cutoff"] if cutoff is None else cutoff)
    if zero_slices_below is None:
        zero_slices_below = defaults["zero_slices_below"]

    def step(message):
        if progress is not None:
            progress(message)

    os.makedirs(workdir, exist_ok=True)
    outputs = {}

    step("Loading volume")
    source_path = nifti_path
    nifti_image = nib.load(source_path)
    image_data = nifti_image.get_fdata()

    if skull_strip:
        step("Skull stripping (bet)")
        stripped = os.path.join(workdir, "temp_sks.nii.gz")
        skull_strip_image(source_path, stripped)
        source_path = stripped
        nifti_image = nib.load(stripped)
        image_data = nifti_image.get_fdata()
        outputs["skull_stripped"] = stripped

    affine = nifti_image.affine

    if sigma and sigma > 0:
        step("Gaussian blur (sigma={0})".format(sigma))
        image_data = gaussian_filter(image_data, sigma)
        if save_intermediates:
            path = os.path.join(workdir, "{0}_gaub_sigma={1}.nii".format(output_csv, sigma))
            nib.save(nib.Nifti1Image(image_data, affine=affine), path)
            outputs["blurred"] = path

    if flip:
        # Dark lesions: invert and renormalise, then move the cutoff with it.
        step("Inverting contrast (dark lesions)")
        id_max = np.max(image_data)
        image_data = np.divide(np.subtract(id_max, image_data), id_max)
        cutoff = float(np.divide(np.subtract(id_max, cutoff), id_max))
        thresholded_img = image_data.copy()
    else:
        thresholded_img = image_data.copy()

    step("Thresholding at {0}".format(cutoff))
    thresholded_img[np.where(thresholded_img < cutoff)] = 0
    if zero_slices_below:
        thresholded_img[:, :, : int(zero_slices_below)] = 0

    if save_intermediates:
        path = os.path.join(workdir, "{0}_thresholded_at_{1}_gaub.nii".format(output_csv, cutoff))
        nib.save(nib.Nifti1Image(thresholded_img, affine=affine), path)
        outputs["thresholded"] = path

    step("Labelling connected components")
    structure = generate_binary_structure(3, 3)
    labeled_array, num_features = label(thresholded_img, structure=structure)

    if save_intermediates:
        path = os.path.join(workdir, "{0}_{1}_labeled.nii".format(output_csv, cutoff))
        nib.save(nib.Nifti1Image(labeled_array, affine=affine), path)
        outputs["labeled"] = path

    step("Measuring {0} candidate regions".format(num_features))
    # The desktop loop assigns each region its maximum intensity, one full-array
    # mask per label; ndimage.maximum computes the same values in one pass.
    mean_value_matrix = np.zeros_like(labeled_array, dtype=float)
    if num_features:
        region_max = np.asarray(
            ndimage.maximum(image_data, labeled_array, index=range(1, num_features + 1)),
            dtype=float,
        )
        region_max = np.nan_to_num(region_max)
        lut = np.concatenate([[0.0], region_max])
        mean_value_matrix = lut[labeled_array]
        if save_intermediates:
            path = os.path.join(workdir, "{0}_{1}_mean_values.nii".format(output_csv, cutoff))
            nib.save(nib.Nifti1Image(mean_value_matrix, affine=affine), path)
            outputs["mean_values"] = path

    thresholded_img[np.where(mean_value_matrix == 0)] = 0

    sx, sy, sz = nifti_image.header.get_zooms()[:3]
    voxel_volume = float(sx * sy * sz)
    thresholded_img[np.where(thresholded_img > 0)] = voxel_volume

    step("Computing centres of mass and volumes")
    # Row i is label i, so row 0 is background and is skipped when the CSV is
    # read back - same layout the desktop wrappers write. They index
    # range(0, num_features), which silently drops the final label; this range
    # is inclusive so the last lesion is not lost.
    index = list(range(0, num_features + 1))
    if index:
        coms = ndimage.center_of_mass(thresholded_img, labeled_array, index)
        volumes = ndimage.sum_labels(thresholded_img, labeled_array, index)
    else:
        coms, volumes = [], []

    rows = []
    for i in range(len(index)):
        com = coms[i]
        rows.append(
            {
                "COMs_x": float(com[0]) if not np.isnan(com[0]) else 0.0,
                "COMs_y": float(com[1]) if not np.isnan(com[1]) else 0.0,
                "COMs_z": float(com[2]) if not np.isnan(com[2]) else 0.0,
                "volumes": float(volumes[i]),
            }
        )

    csv_path = os.path.join(workdir, "{0}.csv".format(output_csv))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["COMs_x", "COMs_y", "COMs_z", "volumes"])
        writer.writeheader()
        writer.writerows(rows)
    outputs["csv"] = csv_path

    return {
        "labeled_array": labeled_array,
        "num_features": int(num_features),
        "rows": rows,
        "csv": csv_path,
        "outputs": outputs,
        "cutoff": cutoff,
        "voxel_volume": voxel_volume,
        "shape": tuple(int(s) for s in image_data.shape),
    }


class LesionCatalog:
    """Candidate lesions from a threshold run, ordered the way the desktop
    wrapper orders them: volume-filtered, then largest first."""

    def __init__(self, labeled_array, rows, axis=2,
                 vmin=VOLUME_THRESHOLD_MIN, vmax=VOLUME_THRESHOLD_MAX):
        self.labeled_array = labeled_array
        self.rows = rows
        self.axis = int(axis)
        self._boxes = ndimage.find_objects(labeled_array)
        entries = []
        # Row 0 is background (skipped by thresholdSAM_01062025.py as well).
        for label_id in range(1, len(rows)):
            volume = rows[label_id]["volumes"]
            if not (vmin < volume < vmax):
                continue
            entries.append(
                {
                    "label": label_id,
                    "volume": float(volume),
                    "com": [rows[label_id]["COMs_x"], rows[label_id]["COMs_y"],
                            rows[label_id]["COMs_z"]],
                }
            )
        entries.sort(key=lambda e: e["volume"], reverse=True)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def voxels(self, label_id):
        """Voxel coordinates of one label, as an (N, 3) int array."""
        box = self._boxes[label_id - 1]
        if box is None:
            return np.zeros((0, 3), dtype=int)
        sub = self.labeled_array[box] == label_id
        local = np.argwhere(sub)
        offset = np.array([s.start for s in box])
        return local + offset

    def anchor_frame(self, label_id):
        """The slice along ``axis`` holding most of the lesion - the desktop
        wrapper's "this lesion is largest at ..." choice."""
        voxels = self.voxels(label_id)
        if not len(voxels):
            return 0, voxels
        column = voxels[:, self.axis]
        values, counts = np.unique(column, return_counts=True)
        frame = int(values[np.argmax(counts)])
        return frame, voxels[column == frame]

    def prompt_points(self, label_id, eta=0.2):
        """Foreground prompts for SAM2: the anchor slice's voxels, sparsified.

        Returns ``(frame_idx, points_xy, all_slice_points_xy)``; the coordinate
        swap per axis matches the desktop wrapper's scatter/prompt code.
        """
        frame, slice_voxels = self.anchor_frame(label_id)
        if not len(slice_voxels):
            return frame, np.zeros((0, 2)), np.zeros((0, 2))
        sparse = sparsify(slice_voxels, eta)
        if self.axis == 0:
            to_xy = lambda v: np.stack([v[:, 2], v[:, 1]], axis=1)
        elif self.axis == 1:
            to_xy = lambda v: np.stack([v[:, 2], v[:, 0]], axis=1)
        else:
            to_xy = lambda v: np.stack([v[:, 1], v[:, 0]], axis=1)
        return frame, to_xy(sparse).astype(float), to_xy(slice_voxels).astype(float)
