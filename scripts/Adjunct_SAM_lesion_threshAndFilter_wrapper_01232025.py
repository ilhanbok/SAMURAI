# Take existing SAM / Mirror pipeline and use SAM2 video parsing capabilties
# Dec 17, 2024 | Ilhan Bok, Patrick Belton

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.measure import regionprops, label
import os
import sys
import platform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
import json

import os.path

import dialogues

if __name__ == '__main__':
    print('Usage: python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py NIFTI_PATH STOP')
    print('e.g. python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py ../niivue-images/chris_t1.nii.gz 46')
    nifti_path = str(sys.argv[1])
    if len(sys.argv) >= 3:
        STOP = int(str(sys.argv[2]))

import pickle

def compute_asymmetry_index(slice,i):
    F = np.flipud(slice)
    asymmetry_index_matrix = np.abs(slice - F)
    return asymmetry_index_matrix

    I = slice

    # Get the dimensions of the image
    rows, columns = I.shape

    # Binary image creation (assuming already binary or thresholded)
    binaryImage = I > 0  # Convert to boolean for binary image
    # Label the image and measure properties
    labeledImage = label(binaryImage)
    props = regionprops(labeledImage)

    # Check if any region is detected
    if not props:
        return(np.zeros(I.shape))
    print(props[0].orientation)
    #quit()
    centroid = props[0].centroid
    xCentroid, yCentroid = centroid

    # Middle of the image
    middlex = columns / 2
    middley = rows / 2

    # Translate the image to the center
    deltax = 0#middlex - xCentroid
    deltay = 0#middley - yCentroid
    translatedImage = shift(I, [deltay, deltax])

    theta = -props[0].orientation
    # Rotate the image (adjust angle if needed) # This is the default which should be adjusted in the future
    rotatedImage = rotate(translatedImage, theta, reshape=False)

    # Compute the asymmetry by reflecting the image and calculating absolute differences
    F = np.flipud(rotatedImage)
    asymmetry_index_matrix = np.abs(rotatedImage - F)

    # Extra step: transform the image back


    # Rotate the image (adjust angle if needed) # This is the default which should be adjusted in the future
    rotated_asymmetry_index_matrix = rotate(asymmetry_index_matrix, -theta, reshape=False)
    # Translate the image to the center
    translated_rotated_asymmetry_index_matrix = shift(rotated_asymmetry_index_matrix, [-deltay, -deltax])

    return translated_rotated_asymmetry_index_matrix

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

import numpy as np

def fill_bounded_regions(binary_array):
    """
    Fills all bounded regions of 0s in a binary 3D NumPy array with 1s.

    Parameters:
        binary_array (np.ndarray): A binary 3D NumPy array (containing only 0s and 1s).

    Returns:
        np.ndarray: A binary 3D NumPy array with bounded regions filled.
    """
    if not np.array_equal(binary_array, binary_array.astype(bool)):
        raise ValueError("Input array must be binary (contain only 0s and 1s).")

    # Create a copy of the array to work on
    filled_array = binary_array.copy()

    # Get the shape of the array
    shape = binary_array.shape

    # Iterate until no more changes occur
    changed = True
    while changed:
        changed = False
        # Iterate over each element in the array
        for x in range(1, shape[0] - 1):
            for y in range(1, shape[1] - 1):
                if filled_array[x, y] == 0:
                    # Check if all directly adjacent entries are 1
                    if (filled_array[x - 1, y] == 1 and
                        filled_array[x + 1, y] == 1 and
                        filled_array[x, y - 1] == 1 and
                        filled_array[x, y + 1] == 1):
                        changed = True

    return filled_array


import nibabel as nib
import numpy as np
import torch

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Qt5agg")

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scipy.ndimage import gaussian_filter

from PIL import *
import subprocess



# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")


sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Load the NIfTI file
#nifti_path = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/01_1007/2_MP_RAGE.nii.gz"
#nifti_path="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/07_1064/16_3D_T2_TSE.nii.gz"  # Replace with your NIfTI file path
#nifti_path ="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/02_1200/Mag_Images.nii.gz"
#nifti_path="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/06_1005/3D_T2_TSE.nii.gz"
#nifti_path="/Users/bok/Documents/03_1097/SESSIONS/20150406/ACQUISITIONS/4_3D_SAG_SWAN_T2star/FILES/4_3D_SAG_SWAN_T2star.nii.gz"
#nifti_path="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Segmentation_Results/03_1097_M.nii.gz"
#nifti_path="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/07_1064/16_3D_T2_TSE.nii.gz"
#nifti_path="/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/Tif_07_1064_TRANSPOSED.nii.gz"


# open config file
config_file='config.json'
with open(config_file,'r') as f:
    config = json.load(f)
    print(config) if 'debug' in config else 0
use_dwi = config['dwi'] if 'dwi' in config else 0

# Load NIfTI image
nifti_image = nib.load(nifti_path)

image_data = nifti_image.get_fdata()#[:,:,:,0])#temporary patch
if use_dwi:
    image_data = image_data[:,:,:,0]

image_data = np.transpose(image_data)
image_data = np.flip(image_data,0)

size = nifti_image.shape
print(size)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

import time



class MaskEditor:
    def __init__(self, volume, mask, toggle, axis=0, min=0, max=0):
        self.volume = volume
        self.mask = mask
        self.slice_idx = 0
        self.toggle = toggle
        self.axis = axis

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.3)  # Adjust layout for sliders
        if min == 0 and max == 0:
            self.min = np.min(self.volume[:,:,:])
            self.max = np.max(self.volume[:,:,:])
        else:
            self.min = min
            self.max = max
        self.update_image()
        self.ax.set_title("Adjust Brightness Range")

        def update_brightness(val):
            """Update image brightness range based on slider values."""
            vmin = slider_min.val
            self.min=slider_min.val
            vmax = slider_max.val
            self.max=slider_max.val
            self.img.set_clim(vmin, vmax)  # Adjust brightness range
            self.fig.canvas.draw_idle()

        # Create sliders
        ax_min = plt.axes([0.1, 0.15, 0.35, 0.03])
        ax_max = plt.axes([0.55, 0.15, 0.35, 0.03])

        slider_min = Slider(ax_min, "Min Brightness", self.min, self.max, valinit=self.min)
        slider_max = Slider(ax_max, "Max Brightness", self.min, self.max, valinit=self.max)

        # Attach sliders to update function
        slider_min.on_changed(update_brightness)
        slider_max.on_changed(update_brightness)

        if self.toggle:
            self.mask_overlay = self.ax.imshow(self.mask[:, :, self.slice_idx], clim=(0, 1), cmap='cool', alpha=0.5)

        self.ax.set_title(f'Slice {self.slice_idx}')
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        if self.toggle:
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.confirm_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.confirm_button = Button(self.confirm_ax, 'Yes?')
        self.confirm_button.on_clicked(self.confirm_output)

        self.directions_ax = plt.axes([0.65, 0.01, 0.1, 0.05])
        self.directions_button = Button(self.directions_ax, 'Directions')
        self.directions_button.on_clicked(self.show_directions_toggle)

        plt.show()

    def update_image(self):
        if self.axis == 0:
            self.img = self.ax.imshow(self.volume[self.slice_idx, :, :], cmap='gray', vmin=self.min, vmax=self.max)
        elif self.axis == 1:
            self.img = self.ax.imshow(self.volume[:, self.slice_idx, :], cmap='gray', vmin=self.min, vmax=self.max)
        elif self.axis == 2:
            self.img = self.ax.imshow(self.volume[:, :, self.slice_idx], cmap='gray', vmin=self.min, vmax=self.max)

    def get_mask_slice(self):
        if self.axis == 0:
            return self.mask[self.slice_idx, :, :]
        elif self.axis == 1:
            return self.mask[:, self.slice_idx, :]
        elif self.axis == 2:
            return self.mask[:, :, self.slice_idx]

    def update_display(self):
        self.img.set_array(self.get_image_slice())
        if self.toggle:
            self.mask_overlay.set_array(self.get_mask_slice())
        self.ax.set_title(f'Slice {self.slice_idx}')
        self.fig.canvas.draw()

    def get_image_slice(self):
        if self.axis == 0:
            return self.volume[self.slice_idx, :, :]
        elif self.axis == 1:
            return self.volume[:, self.slice_idx, :]
        elif self.axis == 2:
            return self.volume[:, :, self.slice_idx]

    def on_scroll(self, event):
        max_idx = self.volume.shape[self.axis] - 1
        if event.step > 0:
            self.slice_idx = min(self.slice_idx + 1, max_idx)
        else:
            self.slice_idx = max(self.slice_idx - 1, 0)
        self.update_display()

    def on_click(self, event):
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode:  # Ignore clicks when zoom or pan is active
            return
        if event.inaxes != self.ax:
            return
        x1, x2 = round(event.xdata), round(event.ydata)
        if self.axis == 0:
            self.mask[self.slice_idx, x2, x1] = 1 - self.mask[self.slice_idx, x2, x1]
        elif self.axis == 1:
            self.mask[x2, self.slice_idx, x1] = 1 - self.mask[x2, self.slice_idx, x1]
        elif self.axis == 2:
            self.mask[x2, x1, self.slice_idx] = 1 - self.mask[x2, x1, self.slice_idx]
        self.update_display()

    def confirm_output(self, event):
        print("Mask confirmed!")
        plt.close(self.fig)

    def show_directions_toggle(self, event):
        if self.toggle:
            directions_text = (
                "Use the scroll wheel to navigate slices and examine output mask.\n"
                "Click on a mask point to toggle it.\n"
                "Click 'Yes?' to confirm."
            )
        else:
            directions_text = (
                "Use the scroll wheel to navigate slices and select segmentation layer.\n"
                "Click 'Yes?' on the corresponding view (axial, coronal, or sagittal) to confirm."
            )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, directions_text, fontsize=12, ha='center', va='center', wrap=True)
        ax.set_axis_off()
        plt.show()

class MultiViewMaskEditor:
    def __init__(self, volume, mask, toggle, min_val=0, max_val=0,config_file_def=config_file):
        self.volume = volume
        self.mask = mask
        self.slices = [32, volume.shape[1] // 2, 43]#volume.shape[2] // 2]
        self.toggle = toggle
        self.orientations = ['Axial','Coronal','Sagittal']
        self.shape = volume.shape

        with open(config_file_def,'r') as f:
            config = json.load(f)

        if min_val == 0 and max_val == 0:
            self.min_val = np.min(self.volume)
            self.max_val = np.max(self.volume)
        else:
            self.min_val = min_val
            self.max_val = max_val

        self.thresh = config['threshold'] if 'threshold' in config else np.min(self.mask)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(left=0.1, bottom=0.3)  # Adjust layout for sliders
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.imgs = [
            self.axes[0].imshow(self.volume[self.slices[0], :, :], cmap='gray', vmin=self.min_val, vmax=self.max_val/2),
            self.axes[1].imshow(self.volume[:, self.slices[1], :], cmap='gray', vmin=self.min_val, vmax=self.max_val/2),
            self.axes[2].imshow(self.volume[:, :, self.slices[2]], cmap='gray', vmin=self.min_val, vmax=self.max_val/2)
        ]

        if self.toggle:
            self.masks = [
                self.axes[0].imshow(self.mask[self.slices[0], :, :] > self.thresh, clim=(0, 1), cmap='cool', alpha=0.5),
                self.axes[1].imshow(self.mask[:, self.slices[1], :] > self.thresh, clim=(0, 1), cmap='cool', alpha=0.5),
                self.axes[2].imshow(self.mask[:, :, self.slices[2]] > self.thresh, clim=(0, 1), cmap='cool', alpha=0.5)
            ]

        self.buttons = []
        for i, ax in enumerate(self.axes):
            ax.set_title(f'{self.orientations[i]} Axis ({i})')
            button_ax = self.fig.add_axes([0.1 + 0.3 * i, 0.01, 0.1, 0.05])
            button = Button(button_ax, 'Yes')
            button.on_clicked(lambda event, idx=i: self.confirm_selection(idx))
            self.buttons.append(button)

        # Brightness sliders
        ax_min = plt.axes([0.1, 0.2, 0.35, 0.03])
        ax_max = plt.axes([0.55, 0.2, 0.35, 0.03])
        ax_thr = plt.axes([0.1, 0.1, 0.6, 0.03])

        self.slider_min = Slider(ax_min, "Min Threshold", self.min_val, self.max_val, valinit=self.min_val)
        self.slider_max = Slider(ax_max, "Max Threshold", self.min_val, self.max_val, valinit=self.max_val)
        self.slider_thr = Slider(ax_thr, "Threshold", np.min(self.mask), np.max(self.mask), valinit=self.thresh)

        self.slider_min.on_changed(self.update_threshold)
        self.slider_max.on_changed(self.update_threshold)
        self.slider_thr.on_changed(self.update_display)

        self.directions_ax = plt.axes([0.9, 0.02, 0.1, 0.05])
        self.directions_button = Button(self.directions_ax, 'Directions')
        self.directions_button.on_clicked(self.show_directions_toggle)

        # Slice specification textbox & buttons
        ax_radio = plt.axes([0.9, 0.2, 0.1, 0.1])
        self.radio = RadioButtons(ax_radio, ('Axial (0)', 'Coronal (1)', 'Sagittal (2)'),
                             label_props={'color': 'cmy', 'fontsize': [12, 14, 16]},
                             radio_props={'s': [16, 32, 64]})

        ax_0 = plt.axes([0.1, 0.15, 0.1, 0.05])
        ax_1 = plt.axes([0.45, 0.15, 0.1, 0.05])
        ax_2 = plt.axes([0.80, 0.15, 0.1, 0.05])

        self.slice_0 = TextBox(ax_0,'Slice (axial):', initial=self.shape[0])
        self.slice_1 = TextBox(ax_1,'Slice (coronal):', initial=self.shape[1])
        self.slice_2 = TextBox(ax_2,'Slice (sagittal):', initial=self.shape[2])

        self.slice_0.on_text_change(lambda event, axis=0: self.on_text_change(event,axis))
        self.slice_1.on_text_change(lambda event, axis=1: self.on_text_change(event,axis))
        self.slice_2.on_text_change(lambda event, axis=2: self.on_text_change(event,axis))

        self.display_text = ax.text(0.5, 0.5, f'Enter a number (0-{(self.shape[0],self.shape[1],self.shape[2])})',
                       ha='center', va='center',
                       fontsize=14, wrap=True)

        plt.show()

    def update_display(self, val=None):
        if val:
            self.thresh = val
        if self.toggle:
            self.masks[0].set_array(self.mask[self.slices[0], :, :] > self.thresh)
            self.masks[1].set_array(self.mask[:,self.slices[1], :] > self.thresh)
            self.masks[2].set_array(self.mask[:,:,self.slices[2]] > self.thresh)
        self.imgs[0].set_array(self.volume[self.slices[0], :, :])
        self.imgs[1].set_array(self.volume[:, self.slices[1], :])
        self.imgs[2].set_array(self.volume[:, :, self.slices[2]])
        for i, ax in enumerate(self.axes):
            ax.set_title(f'{self.orientations[i]} Axis ({i}) - Slice {self.slices[i]}')
        self.fig.canvas.draw()

    def on_scroll(self, event):
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                if event.step > 0:
                    self.slices[i] = min(self.slices[i] + 1, self.volume.shape[i] - 1)
                else:
                    self.slices[i] = max(self.slices[i] - 1, 0)
                self.update_display()
                break

    def confirm_selection(self, axis):
        print(f'Confirmed selection for axis {axis}')
        plt.close(self.fig)
        self.axis = axis
        return axis

    def update_threshold(self, val):
        vmin = self.slider_min.val
        self.min_val = self.slider_min.val
        vmax = self.slider_max.val
        self.max_val = self.slider_max.val
        for img in self.imgs:
            img.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def show_directions_toggle(self, event):
        if self.toggle:
            directions_text = (
                "Use the scroll wheel to navigate slices and examine output mask.\n"
                "Click on a mask point to toggle it.\n"
                "Click 'Yes?' to confirm."
            )
        else:
            directions_text = (
                "Use the scroll wheel to navigate slices and select segmentation layer.\n"
                "Click 'Yes?' on the corresponding view (axial, coronal, or sagittal) to confirm."
            )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, directions_text, fontsize=12, ha='center', va='center', wrap=True)
        ax.set_axis_off()
        plt.show()

    def on_text_change(self, text, axis):
        try:
        #"""Validate and update display as user types
            if axis == 0:
                self.slices[axis] = len(self.volume[:,0,0])-int(text)
            else:
                self.slices[axis] = int(text)
        except Exception as e:
            pass
        self.update_display()

skull_strip = False
if skull_strip:
    output_path = "./temp_sks.nii.gz"
    command = ["bet", nifti_path, output_path, "-m", "-f", str(0.1)]
    subprocess.run(command, check=True)

    nifti_image = nib.load(output_path)
    image_data = nifti_image.get_fdata()


import matplotlib.pyplot as plt

pickleIt = True
mode = 1
times_taken = np.array([])
yn = []
cbar = False

subplot_title_dictionary = {
    'foreground':'Selection: Foreground',
    'background':'Selection: Background',
    'prediction_likelihood_mask':'Prediction Likelihood Mask'
    }

out_mask_logits = []
redo_cache = []
yn_cache = []

LPFP = 'last_preloaded.txt'
if pickleIt:
    def undo(event, ax):
        global coordinates
        global redo_cache
        global yn_cache
        global yn
        print('undo:')
        print(coordinates)
        print(redo_cache)
        print(yn_cache)
        print(yn)
        print('undo out')
        if len(coordinates) > 0:
            redo_cache.append(coordinates.pop())
            yn_cache.append(yn[-1])
            yn = np.delete(yn,-1)
            on_click(event,ax,True,False,True,intentional=True)
    def redo(event, ax):
        global coordinates
        global redo_cache
        global yn_cache
        global yn
        print('redo:')
        print(coordinates)
        print(redo_cache)
        print(yn_cache)
        print(yn)
        print('redo out')
        if len(redo_cache) > 0:
            coordinates.append(redo_cache.pop())
            yn = np.append(yn,yn_cache.pop())
            on_click(event,ax,True,False,True,intentional=True)
    # Define a callback function for mouse clicks
    def on_click(event,ax1,init=False,refresh_view = True,ur = False,intentional=False):
            print('onclick called')
            global yn
            global mode
            global coordinates
            global cbar
            global out_mask_logits
            global redo_cache
            global title

            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar.mode:  # Ignore clicks when zoom or pan is active
                return
            # Store the coordinates of the click
            if event and event.inaxes is ax1:
                intentional = True
                redo_cache = []
                yn_cache = []
                coordinates.append((event.xdata, event.ydata))
                print(f"Mouse clicked at: {event.xdata}, {event.ydata}")
                yn = np.append(yn,mode)
            if intentional:
                ax1.clear()
                imshow_click = ax1.imshow(slice_data, vmin=editor.min_val*np.max(slice_data)/np.max(image_data), vmax=editor.max_val*np.max(slice_data)/np.max(image_data), cmap='gray')#,clim=(editor.min_val, editor.max_val),vmin=editor.min_val,vmax=editor.max_val,cmap='viridis')
                ax1.set_title(title)
                if len(coordinates) > 0:
                    print('1')
                    print(coordinates)
                    print(yn)
                    for pos in range(len(coordinates)):
                        print('done.')
                        print((coordinates[pos][0],coordinates[pos][1]))
                        if yn[pos] == 1:
                            # Optionally, plot the point where clicked
                            ax1.plot(coordinates[pos][0],coordinates[pos][1], 'ko',markersize=10)#markerfacecolor='w',markeredgecolor='k',markeredgewidth=1,markersize=5)
                            ax1.plot(coordinates[pos][0],coordinates[pos][1], 'wo', markersize=8)
                            ax1.plot(coordinates[pos][0],coordinates[pos][1], 'ko', markersize=6)
                            #ax1.plot(event.xdata, event.ydata, markerfacecolor='k',markersize=3.5)
                        else:
                            # Optionally, plot the point where clicked
                            ax1.plot(coordinates[pos][0],coordinates[pos][1], 'wo',markersize=10)#markerfacecolor='k',markeredgecolor='w',markeredgewidth=1,markersize=4)
                            ax1.plot(coordinates[pos][0],coordinates[pos][1], 'ro', markersize=8)
                plt.draw()
                if (len(coordinates) > 0 and ((event and event.inaxes is ax1) or init)) or ur:
                    if len(coordinates) > 0:
                        if not refresh_view or ur:
                            points = np.array(coordinates, dtype=np.float32)
                            scores, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=ann_frame_idx,
                                obj_id=ann_obj_id,
                                points=points,
                                labels=yn
                            )
                        if n == 0:
                            slice_2d = image_data[slices[n],:,:]
                        if n == 1:
                            slice_2d = image_data[:,slices[n],:]
                        if n == 2:
                            slice_2d = image_data[:,:,slices[n]]
                                    # Display the results
                        #fig, ax = plt.subplots(1, len(out_mask_logits), figsize=(20, 10))
                        for i in range(len(out_mask_logits)):


                            # Calculate gradient
                            #gradient_x, gradient_y = np.gradient(out_mask_logits[i].cpu())

                            # Calculate magnitude
                            #magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


                            print(out_mask_logits[i][0].shape)
                            print(i)
                            axlims = ax1.get_xlim()
                            aylims = ax1.get_ylim()
                            print(axlims)
                            print(aylims)
                            plt.close(2)
                            fig, ax0 = plt.subplots(num=2)
                            ax0.imshow(slice_2d, cmap="gray")
                            im = ax0.imshow((out_mask_logits[i][0].cpu().numpy()), alpha=0.5, cmap="viridis")#out_mask_logits[0][i].cpu().numpy()
                            ax0.set_title(subplot_title_dictionary['prediction_likelihood_mask'])
                            ax0.set_xlim(axlims[0],axlims[1])
                            ax0.set_ylim(aylims[0],aylims[1])
                            #ax[i].axis("off")
                            if cbar:
                                cbar.remove()
                            cbar = fig.colorbar(im, ax=ax0)
                plt.show()
                if len(coordinates) == 0:
                    plt.close(2)
                    fig, ax0 = plt.subplots(num=2)
                    plt.show()
    def on_flip(event, ax):
        # Reset x and y axis limits to autoscale
        global mode
        global title
        mode = 1-mode
        if mode == 1:
            title = subplot_title_dictionary['foreground']
        else:
            title = subplot_title_dictionary['background']
        on_click(event=event,ax1=ax,intentional=True)



    havearray = dialogues.call('Did you preload an array of points? (Yes/No)')
    #frame_names = np.array(list(range(data.shape[2]))).astype(str)
    coordinates = []
    preloaded_it = False
    failed_preload = False
    if havearray.lower() == 'yes':
        preloaded_it = True
        first = True
        if preload_path is None:
          if os.path.isfile(LPFP):
            with open(LPFP) as preload_file_file:
                preload_path = preload_file_file.readline().strip()
            try:
                with open(preload_path) as f:
                    index = 0
                    for line in f:
                        if line[0] == '#' or line.strip() == "":
                            continue
                        if first:
                            fl = line.split(',')
                            yn = [int(float(fl[0])),int(float(fl[1]))]
                            first = False
                        else:
                            fl = line.split(':')[1].strip().split(',')
                            fl = [s.strip() for s in fl]
                            tuple = (float(fl[0]),float(fl[1]))
                            if line.split(':')[0] == '1.0':
                                coordinates.insert(0,tuple)
                            else:
                                coordinates.append(tuple)

                    if index < int(yn[0]) + int(yn[1]):
                        print('WARNING: coordinate count lower than specified number (check your preloaded file)')
                #imshow_click = plt.imshow(slice_data, vmin=editor.min_val, vmax=editor.max_val/2)#clim=(editor.min_val, editor.max_val),vmin=editor.min_val,vmax=editor.max_val,cmap='viridis')

                #imshow_click.set_clim(editor.min_val, editor.max_val)
                #plt.show()
                print(coordinates)
                Approve = False
                yn = np.append(np.ones(yn[0]),np.zeros(yn[1]))
            except Exception as e:
                failed_preload = True
    if failed_preload or havearray.lower() != 'yes':
        xs = []
        ys = []
        Approve = False
    start_time = time.time()


    ''' Run the editor with the information given so far '''
    editor = MultiViewMaskEditor(image_data,np.zeros(image_data.shape),False)

    n = editor.axis
    if editor.axis == 0:
        slices=[editor.slices[editor.axis],0,0]
    if editor.axis == 1:
        slices=[0,editor.slices[editor.axis],0]
    if editor.axis == 2:
        slices=[0,0,editor.slices[editor.axis]]

    #slices = [45,80,115] #(0,1,2)
    # [58,82,153] #07_1073 (QSM)
    # [59,51,129] #05_1059 (QSM)
    # [34,89,140] #01_1007 (GRE)
    # [64,127,185] #01_1007 (MP_RAGE)
    # [47,144,110] #01_1011 (GRE)
    # [79,124,197] #05_1003 (MP_RAGE)
    # Load the NIfTI file
    output_dir = "temp_jpeg_stack"
    if platform.system() == 'Windows':
        #command = 'Test-Path {0} -PathType Container'.format(output_dir)
        #print(command)
        if subprocess.run(["powershell", "-Command", f"Test-Path '{output_dir}'"], capture_output=True, text=True):
            subprocess.run(["powershell", "-Command", f"Remove-Item -Path '{output_dir}' -Recurse -Force"],capture_output=True, text=True)
    else:
        subprocess.run(['rm','-rf',output_dir])
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = nib.load(nifti_path)
    data = image_data
    # Normalize data to 0-255 for saving as JPEG
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    frame_names = []
    # Iterate through slices and save as JPEG
    for i in range(data.shape[n]):  # Assuming 3D data
        if n == 0:
            slice_data = data[i, :, :]
        if n == 1:
            slice_data = data[:, i, :]
        if n == 2:
            slice_data = data[:, :, i]

        # Restore orientation
        slice_data = (slice_data)

        img = Image.fromarray(slice_data)
        img.save(os.path.join(output_dir, f"{i:04d}.jpg"))
        frame_names.append(f"{i:04d}.jpg")

    print(f"JPEG stack saved to {output_dir}")
    times_taken = np.append(times_taken, time.time() - start_time)
    start_time = time.time()
    print(times_taken)

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = video_predictor.init_state(video_path=output_dir)

    ann_frame_idx = slices[n]-1 # subtract 1 in order to match ITK-SNAP indices
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # List to store coordinates'

    ''' Perform manual point selection '''
    def confirm_close(event, fig):
        plt.close(fig)
        plt.close(fig(1))
        plt.close(fig(2))
    Final_Approve = False
    redux = False
    coordinates_orig = np.copy(coordinates)
    yn_orig = np.copy(yn)
    while Final_Approve == False:
        print(redux)
        if redux:
            plt.close()
            coordinates = coordinates_orig.tolist()
            yn = yn_orig
        fig, ax = plt.subplots(num=1)
        fig.canvas.mpl_connect('button_press_event',lambda event: on_click(event,ax,refresh_view=False))
        mode = 1
        flip_ax = plt.axes([0.65, 0.01, 0.3, 0.05])
        si_ax = plt.axes([0.2, 0.01, 0.1, 0.05])
        undo_ax = plt.axes([0.4, 0.01, 0.05, 0.05])
        redo_ax = plt.axes([0.45, 0.01, 0.05, 0.05])
        print(editor.min_val)
        print(editor.max_val)
        print('slice: {0}'.format(editor.slices[n]))
        if n == 0:
            slice_data = data[editor.slices[0], :, :]
        if n == 1:
            slice_data = data[:, editor.slices[1], :]
        if n == 2:
            slice_data = data[:, :, editor.slices[2]]
        out = 0
        for c in coordinates:
            if out < yn[0]:
                ax.scatter(c[0],c[1],c='r')
            else:
                ax.scatter(c[0],c[1],c='b')
            out += 1
        flip_button = Button(flip_ax, '(Click to switch fg/bg)')
        flip_button.on_clicked(lambda event: on_flip(event,ax))

        undo_button = Button(undo_ax, '<-')
        undo_button.on_clicked(lambda event: undo(event,ax))

        redo_button = Button(redo_ax, '->')
        redo_button.on_clicked(lambda event: redo(event,ax))

        title = 'Selection: Foreground'

        confirm_button = Button(si_ax, 'Yes')
        confirm_button.on_clicked(lambda event: confirm_close(event,fig))
        fig0, ax0 = plt.subplots(num=2)
        print(f'Confirmed selection.')
        #imshow_click.set_clim(editor.min_val, editor.max_val/2)
        on_click(None,ax,True,False,intentional=True)

        # Listen for all key press events
        #fig.canvas.mpl_connect('key_release_event', on_flip)



#         coordinates_final = [(55.85183633570725, 65.17225247870404),
# (57.7065004887585, 62.57572266443229),
# (51.3264558022622, 66.80435693338913),
# (53.733766233766175, 111.6385281385281),
# (109.14502164502159, 72.85064935064932),
# (97.02380952380946, 162.89393939393935),
# (49.25368314481211, 73.9151829353442)]
#         yn = [3,4]

        print(yn)
        # for labels, `1` means positive click and `0` means negative click
        #np.array(np.append(np.ones((1,int(yn[0]))),np.zeros((1,int(yn[1])))), np.int32)
        if preloaded_it:
            times_taken = np.append(times_taken, '0 (preloaded)')
        else:
            times_taken = np.append(times_taken, time.time() - start_time)
        print(times_taken)

        # fig, ax = plt.subplots(1, len(out_mask_logits[0]), figsize=(20, 10))
        # for i in range(len(out_mask_logits[0])):
        #
        #     # Calculate gradient
        #     gradient_x, gradient_y = np.gradient(out_mask_logits[0][i].cpu())
        #
        #     # Calculate magnitude
        #     magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        #
        #     plt.figure(figsize=(9, 6))
        #     plt.title(f"frame {ann_frame_idx}")
        #     plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
        #     show_points(points, labels, plt.gca())
        #     show_mask((out_mask_logits[0][i].cpu().numpy()), plt.gca(), obj_id=out_obj_ids[0])
        #
        plt.show()
        start_time = time.time()
        answer = dialogues.call('Approve? (Yes/No)')
        if answer == 'yes':
            Final_Approve = True
        redux = True
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not preloaded_it:
      last_fpath = 'points_SAMURAI_{0}.txt'.format(timestr)
      with open(last_fpath,'w') as wt:
        wt.write('{0}, {1}\n'.format(sum(yn),len(yn)-sum(yn)))
        e = 0
        for c in coordinates:
            wt.write('{0}: {1}, {2}\n'.format(yn[e],c[0],c[1]))
            e += 1
        wt.write('#slice: {0}\n'.format(editor.slices[n]))
        wt.write('#axis: {0}\n'.format(n))
      with open(LPFP,'w') as wt2:
        wt2.write(last_fpath)

    times_taken = np.append(times_taken, time.time() - start_time)
    print(times_taken)

    start_time = time.time()
    # Now do the whole video
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        if out_frame_idx < STOP:
            break
        print(out_frame_idx)
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Beta: now run the other way to fill the whole image
    # reverse video

    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=True):
        if out_frame_idx < STOP:
            break
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    times_taken = np.append(times_taken, time.time() - start_time)
    print(times_taken)


    def rot270(img):
        return np.rot90(np.rot90(np.rot90(img)))

    with open('01_1059_20241226.pkl','wb') as f:
        pickle.dump(video_segments, f)
else:
    data = image_data
    with open('01_1059_20241226.pkl','rb') as f:
        video_segments = pickle.load(f)

with open('01_1059_20241226.pkl','rb') as f:
    nifti_res = np.zeros(image_data.shape)
    video_segments = pickle.load(f)
    print(list(video_segments.keys()))
    for i in range(np.min(list(video_segments.keys())),np.max(list(video_segments.keys()))+1):
        for out_obj_id, out_mask in video_segments[i].items():
            if n == 0:
                nifti_res[i,:,:] = out_mask[0]
            if n == 1:
                nifti_res[:,i,:] = out_mask[0]
            if n == 2:
                nifti_res[:,:,i] = out_mask[0]
    start_time = time.time()

    editor_fin = MultiViewMaskEditor(image_data, nifti_res, True, editor.min_val, editor.max_val)

    nifti_res = np.flip(nifti_res,0)
    nifti_res = np.transpose(nifti_res)

    nrzeros = np.zeros(nifti_res.shape)
    included = np.where(nifti_res > editor_fin.thresh)
    nrzeros[included] = 1

    nifti_mask = nib.Nifti1Image(nrzeros, affine=nib.load(nifti_path).affine)

timestr = time.strftime("%Y%m%d-%H%M%S")
nib.save(nifti_mask,'output_SAMURAI_{0}.nii.gz'.format(timestr))
times_taken = np.append(times_taken, time.time() - start_time)
print(times_taken)
print('Slice selection time: {0}\nPoint selection time: {1}\nMask confirmation time: {2}\nMask propagation time: {3}\nOutput validation time: {4}\n(in seconds)'.format(times_taken[0],times_taken[1],times_taken[2],times_taken[3],times_taken[4]))
with open('times_SAMURAI_{0}.txt'.format(timestr),'w') as wt:
    wt.write('Slice selection time: {0}\nPoint selection time: {1}\nMask confirmation time: {2}\nMask propagation time: {3}\nOutput validation time: {4}\n (in seconds)'.format(times_taken[0],times_taken[1],times_taken[2],times_taken[3],times_taken[4]))
print('Program ')
