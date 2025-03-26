# Take existing SAM / Mirror pipeline and use SAM2 video parsing capabilties
# Dec 17, 2024 | Ilhan Bok, Patrick Belton

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.measure import regionprops, label
import os
import sys
import platform

if __name__ == '__main__':
    print('Usage: python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py NIFTI_PATH')
    print('e.g. python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py ../niivue-images/chris_t1.nii.gz')
    nifti_path = str(sys.argv[1])

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

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

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


# Load NIfTI image
nifti_image = nib.load(nifti_path)
image_data = np.transpose(nifti_image.get_fdata())
image_data = np.flip(image_data,0)

size = nifti_image.shape


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
        self.confirm_button = Button(self.confirm_ax, 'Si?')
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
                "Click 'Si?' to confirm."
            )
        else:
            directions_text = (
                "Use the scroll wheel to navigate slices and select segmentation layer.\n"
                "Click 'Si?' on the corresponding view (axial, coronal, or sagittal) to confirm."
            )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, directions_text, fontsize=12, ha='center', va='center', wrap=True)
        ax.set_axis_off()
        plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

class MultiViewMaskEditor:
    def __init__(self, volume, mask, toggle, min_val=0, max_val=0):
        self.volume = volume
        self.mask = mask
        self.slices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
        self.toggle = toggle
        self.orientations = ['Axial','Coronal','Sagittal']

        if min_val == 0 and max_val == 0:
            self.min_val = np.min(self.volume)
            self.max_val = np.max(self.volume)
        else:
            self.min_val = min_val
            self.max_val = max_val

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(left=0.1, bottom=0.3)  # Adjust layout for sliders
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.imgs = [
            self.axes[0].imshow(self.volume[self.slices[0], :, :], cmap='gray', vmin=self.min_val, vmax=self.max_val),
            self.axes[1].imshow(self.volume[:, self.slices[1], :], cmap='gray', vmin=self.min_val, vmax=self.max_val),
            self.axes[2].imshow(self.volume[:, :, self.slices[2]], cmap='gray', vmin=self.min_val, vmax=self.max_val)
        ]

        if self.toggle:
            self.masks = [
                self.axes[0].imshow(self.mask[self.slices[0], :, :], clim=(0, 1), cmap='cool', alpha=0.5),
                self.axes[1].imshow(self.mask[:, self.slices[1], :], clim=(0, 1), cmap='cool', alpha=0.5),
                self.axes[2].imshow(self.mask[:, :, self.slices[2]], clim=(0, 1), cmap='cool', alpha=0.5)
            ]

        self.buttons = []
        for i, ax in enumerate(self.axes):
            ax.set_title(f'{self.orientations[i]} Axis ({i})')
            button_ax = self.fig.add_axes([0.1 + 0.3 * i, 0.01, 0.1, 0.05])
            button = Button(button_ax, 'Si')
            button.on_clicked(lambda event, idx=i: self.confirm_selection(idx))
            self.buttons.append(button)

        # Brightness sliders
        ax_min = plt.axes([0.1, 0.15, 0.35, 0.03])
        ax_max = plt.axes([0.55, 0.15, 0.35, 0.03])

        self.slider_min = Slider(ax_min, "Min Brightness", self.min_val, self.max_val, valinit=self.min_val)
        self.slider_max = Slider(ax_max, "Max Brightness", self.min_val, self.max_val, valinit=self.max_val)

        self.slider_min.on_changed(self.update_brightness)
        self.slider_max.on_changed(self.update_brightness)

        self.directions_ax = plt.axes([0.9, 0.02, 0.1, 0.05])
        self.directions_button = Button(self.directions_ax, 'Directions')
        self.directions_button.on_clicked(self.show_directions_toggle)

        plt.show()

    def update_display(self):
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

    def update_brightness(self, val):
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
                "Click 'Si?' to confirm."
            )
        else:
            directions_text = (
                "Use the scroll wheel to navigate slices and select segmentation layer.\n"
                "Click 'Si?' on the corresponding view (axial, coronal, or sagittal) to confirm."
            )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, directions_text, fontsize=12, ha='center', va='center', wrap=True)
        ax.set_axis_off()
        plt.show()





skull_strip = False
if skull_strip:
    output_path = "./temp_sks.nii.gz"
    command = ["bet", nifti_path, output_path, "-m", "-f", str(0.1)]
    subprocess.run(command, check=True)

    nifti_image = nib.load(output_path)
    image_data = nifti_image.get_fdata()


import matplotlib.pyplot as plt

pickleIt = True

fwrite = open('data_{0}.txt'.format(sys.argv[1].split('/')[-1]),'w')

times_taken = np.array([])
if pickleIt:

    class Browser:
        def __init__(self):
            self.ff = False

        # Define a callback function for mouse clicks
        def on_click(self, event):
            if self.ff:
                toolbar = plt.get_current_fig_manager().toolbar
                if toolbar.mode:  # Ignore clicks when zoom or pan is active
                    return
                # Store the coordinates of the click
                if event.inaxes:
                    coordinates.append((event.xdata, event.ydata))
                    print(f"Mouse clicked at: {event.xdata}, {event.ydata}")

                    # Optionally, plot the point where clicked
                    plt.plot(event.xdata, event.ydata, 'ro')
                    plt.draw()
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar.mode:  # Ignore clicks when zoom or pan is active
                return
            global clicks, rect, box
            if event.xdata is None or event.ydata is None:
                return  # Ignore clicks outside the plot

            clicks.append((event.xdata, event.ydata))

            if len(clicks) == 1:
                # Set initial rectangle position
                rect.set_xy(clicks[0])
                rect.set_width(0)
                rect.set_height(0)
            elif len(clicks) == 2:
                # Set final rectangle dimensions
                x0, y0 = clicks[0]
                x1, y1 = clicks[1]
                rect.set_xy((min(x0, x1), min(y0, y1)))
                rect.set_width(abs(x1 - x0))
                rect.set_height(abs(y1 - y0))
                box[:] = [x0, y0, x1, y1]
                print("Selected Box:", box)
                fwrite.write(f'{box[0]}  {box[1]}  {box[2]}  {box[3]}')
                plt.draw()
                self.ff = True

                plt.draw()

    # Global variables
    box = np.zeros(4, dtype=np.float32)
    clicks = []
    rect = None



    start_time = time.time()
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
    print(times_taken)

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = video_predictor.init_state(video_path=output_dir)

    ann_frame_idx = slices[n]-1 # subtract 1 in order to match ITK-SNAP indices
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # List to store coordinates'


    havearray = input('Did you preload an array of points? (Si/No)')
    #frame_names = np.array(list(range(data.shape[2]))).astype(str)
    if havearray == 'Si':
        coordinates = []
        first = True
        with open('/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/Radiomics_project/Segmentation/01_1051_boundingbox') as f:
            ln = 0
            for line in f:
                if ln == 0:
                    pass
                if ln == 1:
                    box = np.array(line.split()).astype(float).astype(int)
                if ln >= 2:
                    fl = line.split(',')
                    fl = [s.strip() for s in fl]
                    if first:
                        yn = [fl[0],fl[1]]
                        first = False
                    else:
                        coordinates.append((fl[0],fl[1]))
                ln += 1
        gusta = True
        times_taken = np.append(times_taken,'0 (preload)')
    else:
        start_time = time.time()
        gusta = False
    while gusta == False:
        coordinates = []
        fig, ax=plt.subplots()
        print(editor.min_val)
        print(editor.max_val)
        print('slice: {0}'.format(editor.slices[n]))
        fwrite.write('slice: {0}'.format(editor.slices[n]))
        if n == 0:
            slice_data = data[editor.slices[0], :, :]
        if n == 1:
            slice_data = data[:, editor.slices[1], :]
        if n == 2:
            slice_data = data[:, :, editor.slices[2]]
        imshow_click = plt.imshow(slice_data)#clim=(editor.min_val, editor.max_val),vmin=editor.min_val,vmax=editor.max_val,cmap='viridis')
        #imshow_click.set_clim(editor.min_val, editor.max_val)


        # Set up the figure
        #fig, ax = plt.subplots()
        #ax.set_xlim(0, 500)
        #ax.set_ylim(0, 500)
        #ax.invert_yaxis()  # Match image coordinate system
        rect = plt.Rectangle((0,0), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        j = Browser()
        fig.canvas.mpl_connect('button_press_event', j.on_click)
        plt.show()
        g = input('Gusta? (Si/No)')
        confirmed = False
        while confirmed == False :
            try:
                if g == 'Si':
                    yn = input('Please enter fg (1) or bg (0) list: ')
                    yn = yn.split(' ')
                    print(yn)
                    fwrite.write(f'{yn[0]}, {yn[1]}')
                    gusta = True
                    confirmed = True
                else:
                    h = input('Are you sure you want to discard? (Si/No)')
                    if h == 'Si':
                        confirmed = True
                    else:
                        g = 'Si'
                    gusta = False
            except:
                print('Incorrect input')
            finally:
                pass

    times_taken = np.append(times_taken, time.time() - start_time)
    print(times_taken)

    # Let's add a positive click at (x, y)
    points = np.array(coordinates, dtype=np.float32)

    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(np.append(np.ones((1,int(yn[0]))),np.zeros((1,int(yn[1])))), np.int32)

    start_time = time.time()
    scores, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box
    )

    masks = out_mask_logits
    if n == 0:
        slice_2d = image_data[slices[n],:,:]
    if n == 1:
        slice_2d = image_data[:,slices[n],:]
    if n == 2:
        slice_2d = image_data[:,:,slices[n]]
                # Display the results
    fig, ax = plt.subplots(1, len(out_mask_logits), figsize=(20, 10))
    for i in range(len(out_mask_logits)):


        # Calculate gradient
        #gradient_x, gradient_y = np.gradient(out_mask_logits[i].cpu())

        # Calculate magnitude
        #magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


        print(out_mask_logits[i][0].shape)
        print(i)

        ax.imshow(slice_2d, cmap="gray")
        im = ax.imshow((out_mask_logits[i][0].cpu().numpy()), alpha=0.2, cmap="viridis")#out_mask_logits[0][i].cpu().numpy()
        ax.set_title(f"Mask {i}")
        #ax[i].axis("off")
        fig.colorbar(im, ax=ax)
    times_taken = np.append(times_taken, time.time() - start_time)
    print(times_taken)
    plt.show()


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
    # plt.show()



    gusta = False
    g = input('Gusta? (Si/No)')
    confirmed = False
    while confirmed == False:
        if g == 'Si':
            gusta = True
            confirmed = True
        else:
            h = input('Are you sure you want to discard? (Si/No)')
            if h == 'Si':
                confirmed = True

    start_time = time.time()
    # Now do the whole video
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        print(out_frame_idx)
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i].cpu().numpy() > 2.0)
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Beta: now run the other way to fill the whole image
    # reverse video

    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i].cpu().numpy() > 2.0)
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

    editor_fin = MaskEditor(image_data, nifti_res, True, editor.axis, editor.min_val, editor.max_val)

    nifti_res = np.flip(nifti_res,0)
    nifti_res = np.transpose(nifti_res)

    nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
nib.save(nifti_mask,'output_SAMURAI_{0}.nii.gz'.format(timestr))
times_taken = np.append(times_taken, time.time() - start_time)
print(times_taken)
print('Slice selection time: {0}\nPoint selection time: {1}\nMask confirmation time: {2}\nMask propagation time: {3}\nOutput validation time: {4}\n(in seconds)'.format(times_taken[0],times_taken[1],times_taken[2],times_taken[3],times_taken[4]))
with open('times_SAMURAI_{0}.txt'.format(timestr),'w') as wt:
    wt.write('Slice selection time: {0}\nPoint selection time: {1}\nMask confirmation time: {2}\nMask propagation time: {3}\nOutput validation time: {4}\n (in seconds)'.format(times_taken[0],times_taken[1],times_taken[2],times_taken[3],times_taken[4]))
quit()

def check_overlapping_entries(arr1, arr2):

    # Find the number of entries in common by position

    # Initialize counter
    counter = 0
    # Iterate through both arrs element by element
    for i in range(arr1.shape[0]):  # Rows
        for j in range(arr1.shape[1]):  # Columns
            if (arr1[i][j] <= arr2[i][j]) and arr1[i][j] == 1:
                counter += 1
    filt = arr2[np.where(arr1 == 1)]
    if counter > 0 and len(filt) > 0:
        max = np.mean(filt)#[np.where(filt<254)])
    else:
        max = 0
    #print(counter)
    return (counter, max)

#size = (192,256,80)
# (128,192,192) # QSM
# (192,128,192) # QSM
# (128,180,192) # GRE
# (176,240,256) # MP-RAGE
from scipy import ndimage
nifti_res = np.zeros(size)
maxes = []
for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            if n == 0:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[i,:,:]).astype(np.uint8)
                nifti_res[i,:,:] = (nifti_res[i,:,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 1:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[:,i,:]).astype(np.uint8)
                nifti_res[:,i,:] = (nifti_res[:,i,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 2:
                print(i)
                curr_mask_0_symmetrycheck = compute_asymmetry_index((data[:,:,i]),i).astype(np.uint8)
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck > 0)] = 1
                t = 180
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck < t)] = 0
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck >= t)] = 1
                #curr_mask_0_symmetrycheck = fill_bounded_regions(curr_mask_0_symmetrycheck)
                #curr_mask_0_symmetrycheck = ndimage.binary_fill_holes(curr_mask_0_symmetrycheck).astype(int)

                nifti_res[:,:,i] = curr_mask_0_symmetrycheck# + np.multiply(out_mask[0],curr_mask_0_symmetrycheck)
                #if np.any(out_mask[0] == curr_mask_0_symmetrycheck):
                counter,max = check_overlapping_entries(out_mask[0],curr_mask_0_symmetrycheck)#np.multiply(data[:,:,i],curr_mask_0_symmetrycheck))
                maxes.append(max)
                # if i == 110:
                #     print(max)
                #     plt.imshow(curr_mask_0_symmetrycheck)
                #     plt.show()
        x += 1

print(maxes)
max_lesion_cont = np.zeros((len(video_segments)))
prevmask = np.zeros((size[1],size[2]))#(11)
iprev = 0
currmax = 0
print('Ranking lesions while determining volume...')
lesion_dict = {}
# contains information on (dummy example)
# lesion num : 1
# mean intensity whole lesion : 210.42 a.u.
# total intensity : 288*210.42 = 60600.96 voxel a.u.
# total volume : 288 voxels
# voxel volume : 4.33 mm^3
lesion_num = 0
insideLesion = False
insideLesionPrev = False

# Compute all the final stats for the lesion dictionary
sx, sy, sz = nifti_image.header.get_zooms()
VV = sx*sy*sz
if n == 0:
    VA = sy*sz
if n == 1:
    VA = sx*sz
if n == 2:
    VA = sx*sy

for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            print(out_mask[0].shape)
            print(prevmask.shape)
            counter,_= check_overlapping_entries(out_mask[0],prevmask)
            #plt.imshow(prevmask)
            #plt.show()
            print(np.where(out_mask[0]==prevmask))
            max = maxes[i]
            insideLesionPrev = insideLesion
            if counter > 0:
                insideLesion = True
                if insideLesion == True and insideLesionPrev == False:
                    lesion_dict[lesion_num] = {
                        'meanIntensities' : [],
                        'totalMeanIntensity' : 0,
                        'sliceAreas' : [],
                        'sliceAreasVoxel' : [],
                        'totalVolume' : 0,
                        'totalVolumeVoxel' : 0
                    }
                print(i)
                print('yes')
                print(max)
                print(currmax)
                currmax += max
                lesion_dict[lesion_num]['meanIntensities'].append(max)
                lesion_dict[lesion_num]['sliceAreas'].append(np.sum(out_mask[0]))
            else:
                insideLesion = False
                if insideLesion == False and insideLesionPrev == True:
                    l = lesion_num
                    lesion_dict[l]['totalVolume'] = np.sum(lesion_dict[l]['sliceAreas'])
                    lesion_dict[l]['totalVolumeVoxel'] = np.multiply(lesion_dict[l]['totalVolume'],VV)
                    lesion_dict[l]['sliceAreasVoxel'] = np.multiply(lesion_dict[l]['sliceAreas'],VA)
                    lesion_dict[l]['totalMeanIntensity'] = np.sum(np.multiply(lesion_dict[l]['sliceAreasVoxel'],lesion_dict[l]['meanIntensities']))/lesion_dict[l]['totalVolumeVoxel']
                    print(lesion_dict[l]['totalMeanIntensity'])
                    max_lesion_cont[iprev:i] = lesion_dict[l]['totalMeanIntensity']
                    lesion_num += 1
                currmax = 0
                iprev = i
            prevmask = out_mask[0]
        x += 1

# Complete the whole loop by filling in the remaining items
max_lesion_cont[iprev:] = currmax
print(max_lesion_cont)

for i in range(len(lesion_dict)):
    print('Lesion #: {0}'.format(i))
    print('Total Volume (mm^3): {0}'.format(lesion_dict[i]['totalVolumeVoxel']))
    print('Mean Lesion Asymmetry Index: {0}'.format(lesion_dict[i]['totalMeanIntensity']))

nifti_res_SAMonly_lesionrank = np.zeros(size)
print('Applying rankings...')
for i in range(len(video_segments)):
    for out_obj_id, out_mask in video_segments[i].items():
        nifti_res_SAMonly_lesionrank[:,:,i] = nifti_res_SAMonly_lesionrank[:,:,i] + np.multiply(out_mask[0],max_lesion_cont[i])#curr_mask_0_symmetrycheck
print(np.where(nifti_res_SAMonly_lesionrank > 0))
nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1_overlap190_countermode_2_skullstripped.nii.gz'.format(n))

#nifti_mask = nib.Nifti1Image(nifti_res_SAMonly, affine=nib.load(nifti_path).affine)
#nib.save(nifti_mask, 'output_video_01_1007_20140402_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt_12262024-2.nii.gz'.format(n))

nifti_mask = nib.Nifti1Image(nifti_res_SAMonly_lesionrank, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt-2_lesionrank_skullstripped.nii.gz'.format(n))

video_dir = output_dir
# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


'''





# Implement later
##bias_corrected = nib.Nifti1Image(image_data, affine=input_image.affine)
##output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_bias_corrected.nii.gz"  # Replace with your desired output path
##nib.save(bias_corrected, output_path)

# Normalize the image data to 0-255 for SAM2 compatibility
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
image_data = image_data.astype(np.uint8)

mask_volume = np.zeros(image_data.shape, dtype=np.uint8)

predictor = SAM2ImagePredictor(sam2_model)

#for i in range(image_data.shape[-1]):  # Iterate through slices along the z-axis
# Select a slice to visualize and segment
start = 0#image_data.shape[2] // 2  # Middle slice as an example

# these masks represent each of 3 directions - when dot-multiplied with the index
# they allow for segmentation in 3 dimensions
# 0: axial
# 1: coronal
# 2: sagittal
for n in range(0,3):
    if n is 0:
        start = 128
    if n is 1:
        start = 128
    if n is 2:
        start = 128
    for slice_idx in range(start,start+5):#image_data.shape[n]):#image_data.shape[2]):
        print('{0} / {1}'.format(slice_idx,image_data.shape[n]))
        if n is 0:
            slice_2d = image_data[slice_idx,:,:]
        if n is 1:
            slice_2d = image_data[:,slice_idx,:]
        if n is 2:
            slice_2d = image_data[:,:,slice_idx]

        # Cap the maximum brightness for segmentation
        #slice_2d[np.where(slice_2d > 80)] = 80
        # Set dim regions to 0
        #slice_2d[np.where(slice_2d < 85)] = 0
        # Convert the slice to RGB format for SAM
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        predictor.set_image(slice_rgb)

        tolerance = 170 # threshold under which re-segmentation ceases
        iterations = 1 # the maximum number of segmentation attempts

        max_val = 1e100
        iteration = 0
        curr_slice_2d = slice_2d
        curr_mask_0 = np.zeros(np.shape(slice_2d))
        while max_val > tolerance and iteration < iterations:
            # Define a point (or points) to guide the segmentation
            max_val = np.max(curr_slice_2d)
            print('max: {0}'.format(max_val))

            # Gaussian blur the image (equivalent of taking the avg of 5x5 region)
            sigma = 3  # Standard deviation for Gaussian kernel
            blurred_image = gaussian_filter(curr_slice_2d, sigma=sigma)

            # Find the brightest region
            brightest_index = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)
            p =np.arr(np.unravel_index(np.argmax(curr_slice_2d), slice_2d.shape));
            #p=brightest_index
            input_point = np.array([[p[1],p[0]]])  # Example: one point at (100, 150)
            input_label = np.array([1])  # 1 indicates foreground
            print(input_point)

            # Perform segmentation
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # Return multiple masks for more options
            )

            # Convert the slice to RGB format for SAM
            slice_rgb = np.stack([curr_slice_2d] * 3, axis=-1)
            predictor.set_image(slice_rgb)

            # Update masks
            curr_slice_2d = (np.multiply(curr_slice_2d,np.invert(masks[0].astype(bool)))).astype(np.uint8)
            curr_mask_0 = np.array(curr_mask_0.astype(bool) | masks[0].astype(bool)).astype(np.uint8)
            # Update tracker
            iteration = iteration + 1
            print('iteration: {0} | tolerance: {1} / {2}'.format(iteration,max_val,tolerance))

        # Now take that output and determine symmetry of current slice to improve prediction quality
        curr_mask_0_symmetrycheck = curr_mask_0#np.multiply(curr_mask_0.astype(np.uint8),compute_asymmetry_index(slice_2d).astype(np.uint8)).astype(np.uint8)

        #Display the results
        fig, ax = plt.subplots(1, len(masks) + 1, figsize=(20, 10))
        ax[0].imshow(slice_2d, cmap="gray")
        ax[0].set_title("Original Slice")
        ax[0].axis("off")
        i=0
        ax[i + 1].imshow(slice_2d, cmap="gray")
        ax[i + 1].imshow(curr_mask_0_symmetrycheck, alpha=0.5, cmap="jet")
        ax[i + 1].set_title(f"Mask {i + 1} (Score: {scores[0]:.2f})")
        ax[i + 1].axis("off")

        plt.show()

        # Save the masks as binary images (optional)
        saveBinary = False;
        if saveBinary:
            for i, mask in enumerate(masks):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(f"segmentation_mask_{i + 1}_slice_{slice_idx}.png")

        # Save the masks as a NIfTI file (optional)
        if n is 0:
            mask_volume[slice_idx,:,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 1:
            mask_volume[:,slice_idx,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 2:
            mask_volume[:,:,slice_idx] = curr_mask_0_symmetrycheck.astype(np.uint8)

    # Save the resulting mask volume as a NIfTI file
    mask_nifti = nib.Nifti1Image(mask_volume, affine=nib.load(nifti_path).affine)
    output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_dir={0}_1.nii.gz".format(n)  # Replace with your desired output path
    ####nib.save(mask_nifti, output_path)

    print(f"Saved the segmented mask volume to {output_path}")
'''
# Take existing SAM / Mirror pipeline and use SAM2 video parsing capabilties
# Dec 17, 2024 | Ilhan Bok, Patrick Belton

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.measure import regionprops, label
import os
import sys

if __name__ == '__main__':
    print('Usage: python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py NIFTI_PATH')
    print('e.g. python Adjunct_SAM_lesion_threshAndFilter_wrapper_01232025.py ../niivue-images/chris_t1.nii.gz')
    nifti_path = str(sys.argv[1])

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


# Load NIfTI image
nifti_image = nib.load(nifti_path)
image_data = nifti_image.get_fdata()

size = nifti_image.shape

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class MaskEditor:
    def __init__(self, volume, mask, can_toggle):
        self.volume = volume
        self.mask = mask
        self.slice_idx = 0

        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.volume[self.slice_idx+1], cmap='gray')
        self.mask_overlay = self.ax.imshow(self.mask[self.slice_idx+1], cmap='Reds', alpha=0.5)

        self.ax.set_title(f'Slice {self.slice_idx}')
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        if can_toggle:
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.show()

    def update_display(self):
        self.img.set_array(self.volume[self.slice_idx])
        self.mask_overlay.set_array(self.mask[self.slice_idx])
        self.ax.set_title(f'Slice {self.slice_idx}')
        self.fig.canvas.draw()

    def on_scroll(self, event):
        if event.step > 0:
            self.slice_idx = min(self.slice_idx + 1, self.volume.shape[0] - 1)
        else:
            self.slice_idx = max(self.slice_idx - 1, 0)
        self.update_display()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.mask[self.slice_idx, y, x] = 1 - self.mask[self.slice_idx, y, x]  # Toggle mask value
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
if pickleIt:

    # Define a callback function for mouse clicks
    def on_click(event):
        # Store the coordinates of the click
        if event.inaxes:
            coordinates.append((event.xdata, event.ydata))
            print(f"Mouse clicked at: {event.xdata}, {event.ydata}")
            fwrite.write(f'{event.xdata}, {event.ydata}')
            # Optionally, plot the point where clicked
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
    slices = [45,80,115] #(0,1,2)
    # [58,82,153] #07_1073 (QSM)
    # [59,51,129] #05_1059 (QSM)
    # [34,89,140] #01_1007 (GRE)
    # [64,127,185] #01_1007 (MP_RAGE)
    # [47,144,110] #01_1011 (GRE)
    # [79,124,197] #05_1003 (MP_RAGE)
    for n in range(2,3):
        # Load the NIfTI file
        output_dir = "temp_jpeg_stack"
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
                slice_data = image_data[i, :, :]
            if n == 1:
                slice_data = image_data[:, i, :]
            if n == 2:
                slice_data = image_data[:, :, i]

            # Restore orientation
            slice_data = (slice_data)

            img = Image.fromarray(slice_data)
            img.save(os.path.join(output_dir, f"{i:04d}.jpg"))
            frame_names.append(f"{i:04d}.jpg")

        print(f"JPEG stack saved to {output_dir}")

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        inference_state = video_predictor.init_state(video_path=output_dir)

        ann_frame_idx = slices[n]-1 # subtract 1 in order to match ITK-SNAP indices
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # List to store coordinates'


        havearray = input('Did you preload an array of points? (Si/No)')
        #frame_names = np.array(list(range(data.shape[2]))).astype(str)
        if havearray == 'Si':
            coordinates = []
            first = True
            with open('preload_07_1064') as f:
                for line in f:
                    fl = line.split(',')
                    fl = [s.strip() for s in fl]
                    if first:
                        yn = [fl[0],fl[1]]
                        first = False
                    else:
                        coordinates.append((fl[0],fl[1]))
            gusta = True
        else:
            gusta = False
        while gusta == False:
            coordinates = []
            fig, ax=plt.subplots()
            plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])),interpolation='none')
            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            g = input('Gusta? (Si/No)')
            confirmed = False
            while confirmed == False:
                try:
                    if g == 'Si':
                        yn = input('Please enter fg (1) or bg (0) list: ')
                        yn = yn.split(' ')
                        print(yn)
                        gusta = True
                        confirmed = True
                    else:
                        h = input('Are you sure you want to discard? (Si/No)')
                        if h == 'Si':
                            confirmed = True
                        else:
                            g = 'Si'
                except:
                    print('Incorrect input')
                finally:
                    pass

        # Let's add a positive click at (x, y)
        points = np.array(coordinates, dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array(np.append(np.ones((1,int(yn[0]))),np.zeros((1,int(yn[1])))), np.int32)

        scores, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels
        )

        masks = out_mask_logits
        if n == 0:
            slice_2d = image_data[slices[n],:,:]
        if n == 1:
            slice_2d = image_data[:,slices[n],:]
        if n == 2:
            slice_2d = image_data[:,:,slices[n]]
                    # Display the results
        fig, ax = plt.subplots(1, len(out_mask_logits[0]), figsize=(20, 10))
        for i in range(len(out_mask_logits[0])):


            # Calculate gradient
            gradient_x, gradient_y = np.gradient(out_mask_logits[0][i].cpu())

            # Calculate magnitude
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


            print(out_mask_logits[0][i].shape)

            ax[i].imshow(slice_2d, cmap="gray")
            im = ax[i].imshow((out_mask_logits[0][i].cpu().numpy()), alpha=0.2, cmap="jet")#out_mask_logits[0][i].cpu().numpy()
            ax[i].set_title(f"Mask {i}")
            #ax[i].axis("off")
            fig.colorbar(im, ax=ax[i])

        plt.show()


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
        # plt.show()



        gusta = False
        g = input('Gusta? (Si/No)')
        confirmed = False
        while confirmed == False:
            if g == 'Si':
                gusta = True
                confirmed = True
            else:
                h = input('Are you sure you want to discard? (Si/No)')
                if h == 'Si':
                    confirmed = True

        # Now do the whole video
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            print(out_frame_idx)
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 2.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Beta: now run the other way to fill the whole image
        # reverse video

        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 2.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def rot270(img):
        return np.rot90(np.rot90(np.rot90(img)))

    with open('01_1059_20241226.pkl','wb') as f:
        pickle.dump(video_segments, f)
else:
    n = 2
    data = image_data
    with open('01_1059_20241226.pkl','rb') as f:
        video_segments = pickle.load(f)

with open('01_1059_20241226.pkl','rb') as f:
    nifti_res = np.zeros(size)
    video_segments = pickle.load(f)
    print(video_segments.keys())
    for i in range(np.min(list(video_segments.keys())),np.max(list(video_segments.keys()))+1):
        for out_obj_id, out_mask in video_segments[i].items():
            if n == 0:
                nifti_res[i,:,:] = out_mask[0]
            if n == 1:
                nifti_res[:,i,:] = out_mask[0]
            if n == 2:
                print(i)
                nifti_res[:,:,i] = out_mask[0]
    plt.hist(nifti_res)
    plt.show()
    editor = MaskEditor(image_data, nifti_res,True)

    nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask,'07_1096_NN.nii.gz')
quit()

def check_overlapping_entries(arr1, arr2):

    # Find the number of entries in common by position

    # Initialize counter
    counter = 0
    # Iterate through both arrs element by element
    for i in range(arr1.shape[0]):  # Rows
        for j in range(arr1.shape[1]):  # Columns
            if (arr1[i][j] <= arr2[i][j]) and arr1[i][j] == 1:
                counter += 1
    filt = arr2[np.where(arr1 == 1)]
    if counter > 0 and len(filt) > 0:
        max = np.mean(filt)#[np.where(filt<254)])
    else:
        max = 0
    #print(counter)
    return (counter, max)

#size = (192,256,80)
# (128,192,192) # QSM
# (192,128,192) # QSM
# (128,180,192) # GRE
# (176,240,256) # MP-RAGE
from scipy import ndimage
nifti_res = np.zeros(size)
maxes = []
for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            if n == 0:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[i,:,:]).astype(np.uint8)
                nifti_res[i,:,:] = (nifti_res[i,:,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 1:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[:,i,:]).astype(np.uint8)
                nifti_res[:,i,:] = (nifti_res[:,i,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 2:
                print(i)
                curr_mask_0_symmetrycheck = compute_asymmetry_index((data[:,:,i]),i).astype(np.uint8)
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck > 0)] = 1
                t = 180
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck < t)] = 0
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck >= t)] = 1
                #curr_mask_0_symmetrycheck = fill_bounded_regions(curr_mask_0_symmetrycheck)
                #curr_mask_0_symmetrycheck = ndimage.binary_fill_holes(curr_mask_0_symmetrycheck).astype(int)

                nifti_res[:,:,i] = curr_mask_0_symmetrycheck# + np.multiply(out_mask[0],curr_mask_0_symmetrycheck)
                #if np.any(out_mask[0] == curr_mask_0_symmetrycheck):
                counter,max = check_overlapping_entries(out_mask[0],curr_mask_0_symmetrycheck)#np.multiply(data[:,:,i],curr_mask_0_symmetrycheck))
                maxes.append(max)
                # if i == 110:
                #     print(max)
                #     plt.imshow(curr_mask_0_symmetrycheck)
                #     plt.show()
        x += 1

print(maxes)
max_lesion_cont = np.zeros((len(video_segments)))
prevmask = np.zeros((size[1],size[2]))#(11)
iprev = 0
currmax = 0
print('Ranking lesions while determining volume...')
lesion_dict = {}
# contains information on (dummy example)
# lesion num : 1
# mean intensity whole lesion : 210.42 a.u.
# total intensity : 288*210.42 = 60600.96 voxel a.u.
# total volume : 288 voxels
# voxel volume : 4.33 mm^3
lesion_num = 0
insideLesion = False
insideLesionPrev = False

# Compute all the final stats for the lesion dictionary
sx, sy, sz = nifti_image.header.get_zooms()
VV = sx*sy*sz
if n == 0:
    VA = sy*sz
if n == 1:
    VA = sx*sz
if n == 2:
    VA = sx*sy

for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            print(out_mask[0].shape)
            print(prevmask.shape)
            counter,_= check_overlapping_entries(out_mask[0],prevmask)
            #plt.imshow(prevmask)
            #plt.show()
            print(np.where(out_mask[0]==prevmask))
            max = maxes[i]
            insideLesionPrev = insideLesion
            if counter > 0:
                insideLesion = True
                if insideLesion == True and insideLesionPrev == False:
                    lesion_dict[lesion_num] = {
                        'meanIntensities' : [],
                        'totalMeanIntensity' : 0,
                        'sliceAreas' : [],
                        'sliceAreasVoxel' : [],
                        'totalVolume' : 0,
                        'totalVolumeVoxel' : 0
                    }
                print(i)
                print('yes')
                print(max)
                print(currmax)
                currmax += max
                lesion_dict[lesion_num]['meanIntensities'].append(max)
                lesion_dict[lesion_num]['sliceAreas'].append(np.sum(out_mask[0]))
            else:
                insideLesion = False
                if insideLesion == False and insideLesionPrev == True:
                    l = lesion_num
                    lesion_dict[l]['totalVolume'] = np.sum(lesion_dict[l]['sliceAreas'])
                    lesion_dict[l]['totalVolumeVoxel'] = np.multiply(lesion_dict[l]['totalVolume'],VV)
                    lesion_dict[l]['sliceAreasVoxel'] = np.multiply(lesion_dict[l]['sliceAreas'],VA)
                    lesion_dict[l]['totalMeanIntensity'] = np.sum(np.multiply(lesion_dict[l]['sliceAreasVoxel'],lesion_dict[l]['meanIntensities']))/lesion_dict[l]['totalVolumeVoxel']
                    print(lesion_dict[l]['totalMeanIntensity'])
                    max_lesion_cont[iprev:i] = lesion_dict[l]['totalMeanIntensity']
                    lesion_num += 1
                currmax = 0
                iprev = i
            prevmask = out_mask[0]
        x += 1

# Complete the whole loop by filling in the remaining items
max_lesion_cont[iprev:] = currmax
print(max_lesion_cont)

for i in range(len(lesion_dict)):
    print('Lesion #: {0}'.format(i))
    print('Total Volume (mm^3): {0}'.format(lesion_dict[i]['totalVolumeVoxel']))
    print('Mean Lesion Asymmetry Index: {0}'.format(lesion_dict[i]['totalMeanIntensity']))

nifti_res_SAMonly_lesionrank = np.zeros(size)
print('Applying rankings...')
for i in range(len(video_segments)):
    for out_obj_id, out_mask in video_segments[i].items():
        nifti_res_SAMonly_lesionrank[:,:,i] = nifti_res_SAMonly_lesionrank[:,:,i] + np.multiply(out_mask[0],max_lesion_cont[i])#curr_mask_0_symmetrycheck
print(np.where(nifti_res_SAMonly_lesionrank > 0))
nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1_overlap190_countermode_2_skullstripped.nii.gz'.format(n))

#nifti_mask = nib.Nifti1Image(nifti_res_SAMonly, affine=nib.load(nifti_path).affine)
#nib.save(nifti_mask, 'output_video_01_1007_20140402_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt_12262024-2.nii.gz'.format(n))

nifti_mask = nib.Nifti1Image(nifti_res_SAMonly_lesionrank, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt-2_lesionrank_skullstripped.nii.gz'.format(n))

video_dir = output_dir
# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


'''





# Implement later
##bias_corrected = nib.Nifti1Image(image_data, affine=input_image.affine)
##output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_bias_corrected.nii.gz"  # Replace with your desired output path
##nib.save(bias_corrected, output_path)

# Normalize the image data to 0-255 for SAM2 compatibility
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
image_data = image_data.astype(np.uint8)

mask_volume = np.zeros(image_data.shape, dtype=np.uint8)

predictor = SAM2ImagePredictor(sam2_model)

#for i in range(image_data.shape[-1]):  # Iterate through slices along the z-axis
# Select a slice to visualize and segment
start = 0#image_data.shape[2] // 2  # Middle slice as an example

# these masks represent each of 3 directions - when dot-multiplied with the index
# they allow for segmentation in 3 dimensions
# 0: axial
# 1: coronal
# 2: sagittal
for n in range(0,3):
    if n is 0:
        start = 128
    if n is 1:
        start = 128
    if n is 2:
        start = 128
    for slice_idx in range(start,start+5):#image_data.shape[n]):#image_data.shape[2]):
        print('{0} / {1}'.format(slice_idx,image_data.shape[n]))
        if n is 0:
            slice_2d = image_data[slice_idx,:,:]
        if n is 1:
            slice_2d = image_data[:,slice_idx,:]
        if n is 2:
            slice_2d = image_data[:,:,slice_idx]

        # Cap the maximum brightness for segmentation
        #slice_2d[np.where(slice_2d > 80)] = 80
        # Set dim regions to 0
        #slice_2d[np.where(slice_2d < 85)] = 0
        # Convert the slice to RGB format for SAM
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        predictor.set_image(slice_rgb)

        tolerance = 170 # threshold under which re-segmentation ceases
        iterations = 1 # the maximum number of segmentation attempts

        max_val = 1e100
        iteration = 0
        curr_slice_2d = slice_2d
        curr_mask_0 = np.zeros(np.shape(slice_2d))
        while max_val > tolerance and iteration < iterations:
            # Define a point (or points) to guide the segmentation
            max_val = np.max(curr_slice_2d)
            print('max: {0}'.format(max_val))

            # Gaussian blur the image (equivalent of taking the avg of 5x5 region)
            sigma = 3  # Standard deviation for Gaussian kernel
            blurred_image = gaussian_filter(curr_slice_2d, sigma=sigma)

            # Find the brightest region
            brightest_index = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)
            p =np.arr(np.unravel_index(np.argmax(curr_slice_2d), slice_2d.shape));
            #p=brightest_index
            input_point = np.array([[p[1],p[0]]])  # Example: one point at (100, 150)
            input_label = np.array([1])  # 1 indicates foreground
            print(input_point)

            # Perform segmentation
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # Return multiple masks for more options
            )

            # Convert the slice to RGB format for SAM
            slice_rgb = np.stack([curr_slice_2d] * 3, axis=-1)
            predictor.set_image(slice_rgb)

            # Update masks
            curr_slice_2d = (np.multiply(curr_slice_2d,np.invert(masks[0].astype(bool)))).astype(np.uint8)
            curr_mask_0 = np.array(curr_mask_0.astype(bool) | masks[0].astype(bool)).astype(np.uint8)
            # Update tracker
            iteration = iteration + 1
            print('iteration: {0} | tolerance: {1} / {2}'.format(iteration,max_val,tolerance))

        # Now take that output and determine symmetry of current slice to improve prediction quality
        curr_mask_0_symmetrycheck = curr_mask_0#np.multiply(curr_mask_0.astype(np.uint8),compute_asymmetry_index(slice_2d).astype(np.uint8)).astype(np.uint8)

        #Display the results
        fig, ax = plt.subplots(1, len(masks) + 1, figsize=(20, 10))
        ax[0].imshow(slice_2d, cmap="gray")
        ax[0].set_title("Original Slice")
        ax[0].axis("off")
        i=0
        ax[i + 1].imshow(slice_2d, cmap="gray")
        ax[i + 1].imshow(curr_mask_0_symmetrycheck, alpha=0.5, cmap="jet")
        ax[i + 1].set_title(f"Mask {i + 1} (Score: {scores[0]:.2f})")
        ax[i + 1].axis("off")

        plt.show()

        # Save the masks as binary images (optional)
        saveBinary = False;
        if saveBinary:
            for i, mask in enumerate(masks):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(f"segmentation_mask_{i + 1}_slice_{slice_idx}.png")

        # Save the masks as a NIfTI file (optional)
        if n is 0:
            mask_volume[slice_idx,:,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 1:
            mask_volume[:,slice_idx,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 2:
            mask_volume[:,:,slice_idx] = curr_mask_0_symmetrycheck.astype(np.uint8)

    # Save the resulting mask volume as a NIfTI file
    mask_nifti = nib.Nifti1Image(mask_volume, affine=nib.load(nifti_path).affine)
    output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_dir={0}_1.nii.gz".format(n)  # Replace with your desired output path
    ####nib.save(mask_nifti, output_path)

    print(f"Saved the segmented mask volume to {output_path}")
'''
