# NIfTI handling for the SAMURAI web portal
# Headless port of the array plumbing in
# scripts/Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py

import os

import numpy as np
import nibabel as nib
from PIL import Image

# Axis names as used by MultiViewMaskEditor in the desktop scripts. Index into
# the *display* array (see Volume.display).
AXIS_NAMES = ("Axial", "Coronal", "Sagittal")


class Volume:
    """A loaded NIfTI image plus the two orientations SAMURAI works in.

    ``native``  - the array exactly as ``nib.load(...).get_fdata()`` returns it.
                  Mode (ii) (threshold lesion finding) operates here, as the
                  desktop RAW/QSM wrappers do.
    ``display`` - ``np.flip(np.transpose(native), 0)``, the orientation the
                  manual point-selection GUI uses. Axis 0/1/2 are
                  axial/coronal/sagittal.
    """

    def __init__(self, path, use_dwi=False):
        self.path = str(path)
        self.image = nib.load(self.path)
        data = self.image.get_fdata()
        if use_dwi:
            # temporary patch carried over from the desktop script: 4-D DWI
            # series are reduced to their first volume
            data = data[:, :, :, 0]
        if data.ndim != 3:
            raise ValueError(
                "Expected a 3-D volume, got shape {0}. Tick 'DWI / 4-D input' "
                "if this is a 4-D series.".format(data.shape)
            )
        self.native = data
        self.use_dwi = bool(use_dwi)
        self._display = None

    # -- orientation ------------------------------------------------------
    @property
    def display(self):
        if self._display is None:
            self._display = np.flip(np.transpose(self.native), 0)
        return self._display

    @staticmethod
    def to_native(display_array):
        """Inverse of the display transform, ready for saving as NIfTI."""
        return np.transpose(np.flip(display_array, 0))

    @property
    def affine(self):
        return self.image.affine

    @property
    def zooms(self):
        return self.image.header.get_zooms()[:3]

    @property
    def display_shape(self):
        return self.display.shape

    @property
    def native_shape(self):
        return self.native.shape

    # -- windowing --------------------------------------------------------
    @property
    def vmin(self):
        return float(np.min(self.display))

    @property
    def vmax(self):
        return float(np.max(self.display))

    def slice(self, axis, index):
        """Return one 2-D slice of the display volume."""
        index = int(np.clip(index, 0, self.display.shape[axis] - 1))
        if axis == 0:
            return self.display[index, :, :]
        if axis == 1:
            return self.display[:, index, :]
        if axis == 2:
            return self.display[:, :, index]
        raise ValueError("axis must be 0, 1 or 2")

    def n_slices(self, axis):
        return int(self.display.shape[axis])

    # -- SAM2 input -------------------------------------------------------
    def uint8_stack(self):
        """Whole-volume 0-255 normalisation, exactly as the desktop scripts."""
        data = self.display
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        return data.astype(np.uint8)

    def write_jpeg_stack(self, axis, output_dir):
        """Write one JPEG per slice along ``axis``; SAM2's video predictor
        consumes the directory. Returns the list of frame file names."""
        return write_jpeg_stack(self.uint8_stack(), axis, output_dir)


def normalize_uint8(data):
    """Whole-volume 0-255 normalisation, as the desktop scripts do it."""
    data = np.asarray(data, dtype=np.float64)
    lo, hi = np.min(data), np.max(data)
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return ((data - lo) / (hi - lo) * 255).astype(np.uint8)


def write_jpeg_stack(data_uint8, axis, output_dir):
    """Write a JPEG per slice of an already-normalised uint8 volume."""
    clear_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    frame_names = []
    for i in range(data_uint8.shape[axis]):
        if axis == 0:
            slice_data = data_uint8[i, :, :]
        elif axis == 1:
            slice_data = data_uint8[:, i, :]
        else:
            slice_data = data_uint8[:, :, i]
        Image.fromarray(slice_data).save(os.path.join(output_dir, "{0:04d}.jpg".format(i)))
        frame_names.append("{0:04d}.jpg".format(i))
    return frame_names


def clear_directory(path):
    """Remove a directory tree without shelling out (the desktop scripts use
    ``rm -rf`` / ``Remove-Item``; this is the portable equivalent)."""
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def save_mask(display_mask, affine, out_path, threshold=None, binarize=True):
    """Save a mask given in *display* orientation to ``out_path``.

    Reproduces the tail of the manual script: flip, transpose, threshold, then
    write a binary NIfTI with the source affine.
    """
    arr = Volume.to_native(np.asarray(display_mask))
    if threshold is not None and binarize:
        out = np.zeros(arr.shape)
        out[np.where(arr > threshold)] = 1
    elif binarize:
        out = (arr > 0).astype(np.uint8)
    else:
        out = arr
    nib.save(nib.Nifti1Image(out, affine=affine), out_path)
    return out_path


def save_native(native_array, affine, out_path):
    """Save an array that is already in native orientation (mode ii output)."""
    nib.save(nib.Nifti1Image(native_array, affine=affine), out_path)
    return out_path
