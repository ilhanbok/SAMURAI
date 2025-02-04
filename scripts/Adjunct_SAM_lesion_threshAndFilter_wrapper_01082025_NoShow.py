# Take existing SAM / Mirror pipeline and use SAM2 video parsing capabilties
# Dec 17, 2024 | Ilhan Bok, Patrick Belton

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.measure import regionprops, label
import os

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

def get_bounding_box(image, threshold):
    """
    Calculates the bounding box for image pixels greater than a threshold.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or single-channel).
        threshold (int or float): Pixel intensity threshold.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates of the bounding box.
    """
    y_coords, x_coords = np.where(image > threshold)

    if len(x_coords) == 0 or len(y_coords) == 0:
        # No pixels above the threshold
        return None

    # Calculate bounding box
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return x_min, y_min, x_max, y_max

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

from scipy.ndimage import gaussian_filter, label, generate_binary_structure

import SimpleITK as sitk

from PIL import *
import subprocess

from collections import Counter

import numpy as np

import pandas as pd
import cv2

def sparsify(array, ratio, seed=42):
    """
    Randomly sparsify an array by removing elements entirely, keeping only a given ratio of elements.

    Parameters:
    - array (numpy.ndarray): The input array to be sparsified.
    - ratio (float): The fraction of elements to retain (0 < ratio <= 1).
    - seed (int): Seed for the random number generator (default: 42).

    Returns:
    - numpy.ndarray: A sparsified array with fewer elements.
    """
    if len(array) <= 3:
        return array  # Return the full array if its length is 3 or less

    if not (0 < ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1 (exclusive).")

    np.random.seed(seed)  # Ensure reproducibility
    indices = np.arange(len(array))
    retained_count = int(np.ceil(len(array) * ratio))  # Number of elements to retain
    retained_indices = np.random.choice(indices, size=retained_count, replace=False)

    return array[np.sort(retained_indices)]  # Return the retained elements, sorted

def compute_gradient_magnitude(image):
    """
    Computes the magnitude of the gradient of a 2D image using the Sobel operator.

    Parameters:
        image (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Gradient magnitude image.
    """
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y

    # Compute magnitude of gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to range 0-255 for visualization
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return grad_magnitude

def main(nifti_path, output_csv,list_of_centroids,list_of_corresponding_indices,volumes=[]):
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")


    sam2_checkpoint = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Load the NIfTI file
    #nifti_path = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/01_1011/4_3D_T2star_GRE_ND.nii.gz"  # Replace with your NIfTI file path
    #nifti_path = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/01_1007/3_3D_T2star_GRE_ND.nii.gz"
    #nifti_path = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/02_1070/Mag_Images.nii.gz"
    # Load NIfTI image
    nifti_image = nib.load(nifti_path)
    image_data = nifti_image.get_fdata()

    skull_strip = False
    if skull_strip:
        output_path = "./temp_sks.nii.gz"
        command = ["bet", nifti_path, output_path, "-m", "-f", str(0.1)]
        subprocess.run(command, check=True)

        nifti_image = nib.load(output_path)
        image_data = nifti_image.get_fdata()

    gaussian_blur = False
    sigma = 1
    if gaussian_blur:
        image_data = gaussian_filter(image_data,sigma=sigma)

        nifti_mask = nib.Nifti1Image(image_data, affine=nib.load(nifti_path).affine)
        nib.save(nifti_mask,'{0}_gaub_sigma={1}.nii'.format(nifti_path,sigma));


    take_gradient = False
    if take_gradient:
        image_data = compute_gradient_magnitude(image_data)
    import matplotlib.pyplot as plt
    import sys

    import numpy as np

    pickleIt = True
    # Load the NIfTI file
    makeDir = True

    output_dir = "01_1011_001"
    if makeDir:
        subprocess.run(['rm','-rf',output_dir])
        os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = nib.load(nifti_path)
    data = image_data
    # Normalize data to 0-255 for saving as JPEG
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    frame_names = []
    n = 2
    # Iterate through slices and save as JPEG
    for i in range(data.shape[n]):  # Assuming 3D data
        if makeDir:
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
    if makeDir:
        print(f"JPEG stack saved to {output_dir}")

    def on_click(event):
        # Store the coordinates of the click
        if event.inaxes:
            coordinates.append((event.xdata, event.ydata))
            print(f"Mouse clicked at: {event.xdata}, {event.ydata}")

            # Optionally, plot the point where clicked
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
    Volume_threshold_min = 10
    Volume_threshold_max = 1e6
    list_of_centroids = np.array(list_of_centroids)
    list_of_corresponding_indices = np.array(list_of_corresponding_indices)
    if len(volumes) == 0:
        list_of_corresponding_indices = list_of_corresponding_indices[np.where(list_of_centroids[:,3] > Volume_threshold_min)]
        list_of_centroids = list_of_centroids[np.where(list_of_centroids[:,3] > Volume_threshold_min)]

        list_of_corresponding_indices = list_of_corresponding_indices[np.where(list_of_centroids[:,3] < Volume_threshold_max)]
        list_of_centroids = list_of_centroids[np.where(list_of_centroids[:,3] < Volume_threshold_max)]
    else:
        list_of_corresponding_indices_unsorted = list_of_corresponding_indices[np.where(volumes > Volume_threshold_min)]
        list_of_centroids_unsorted = list_of_centroids[np.where(volumes > Volume_threshold_min)]
        volumes_unsorted = volumes[np.where(volumes > Volume_threshold_min)]

        list_of_corresponding_indices_unsorted = list_of_corresponding_indices_unsorted[np.where(volumes_unsorted < Volume_threshold_max)]
        list_of_centroids_unsorted = list_of_centroids_unsorted[np.where(volumes_unsorted < Volume_threshold_max)]
        volumes_unsorted = volumes_unsorted[np.where(volumes_unsorted < Volume_threshold_max)]

        sortByVolume = True
        if sortByVolume:
            v_s = volumes_unsorted.argsort()
            list_of_corresponding_indices = list_of_corresponding_indices_unsorted[v_s[::-1]]
            list_of_centroids = list_of_centroids_unsorted[v_s[::-1]]
            volumes = volumes_unsorted[v_s[::-1]]
        else:
            list_of_corresponding_indices = list_of_corresponding_indices_unsorted
            list_of_centroids = list_of_centroids_unsorted
            volumes = volumes_unsorted
    #quit()
    # Generate threshold classifier
    thresholded_img_classifier = data.copy()
    RAW_cutoff = 250
    thresholded_img_classifier[np.where(thresholded_img_classifier < RAW_cutoff)] = 0

    s = generate_binary_structure(3,3)
    labeled_array_SAM, num_features_SAM = label(thresholded_img_classifier, structure=s)
    #print('has {0} unique features'.format(num_features_SAM))

    size = image_data.shape
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print('has {0} unique features'.format(len(pd.unique(list_of_corresponding_indices)[0:])))
    input('Press Enter to continue.')

    curr_e = 0
    unique_loop = pd.unique(list_of_corresponding_indices)
    unique_vols = pd.unique(volumes)

    while curr_e < len(unique_loop):
        print('lesion ID: {0}'.format(curr_e))
        e = unique_loop[curr_e]
        inference_state = video_predictor.init_state(video_path=output_dir)
        if pickleIt:
            #print(e)
            #print(list_of_corresponding_indices)
            #print(np.where(list_of_corresponding_indices == e))
            x = np.where(np.array(list_of_corresponding_indices) == e)[0]
            print(x)
            if len(x) == 0:
                print('blank index.')
                continue
            #if len(x) > 1:
            #    print('Multiple coordinates per lesion currently not supported.')
            #    continue
            num_centroids = 0
            #TODO (01/07/2025): accomodate for multiple slices
            slices = []
            for iiii in range(len(x)):
                a = list_of_centroids[x[iiii]]
                slices.append([round(a[0]),round(a[1]),round(a[2])]) #(0,1,2)
            slices = np.array(slices)
            # [58,82,153] #07_1073 (QSM)
            # [59,51,129] #05_1059 (QSM)
            # [34,89,140] #01_1007 (GRE)
            # [64,127,185] #01_1007 (MP_RAGE)
            # [47,144,110] #01_1011 (GRE)
            # [79,124,197] #05_1003 (MP_RAGE)
            print(slices)
            # Determine the most frequent slice
            ir = slices[:,2]
            counter = Counter(ir)
            most_common_element = counter.most_common(1)
            print(most_common_element)
            print('This lesion is largest at {0}'.format(most_common_element))
            print(np.where(ir == most_common_element[0][0]))
            slices = slices[np.where(ir == most_common_element[0][0])][:]
            num_centroids = len(slices)
            ann_frame_idx = slices[0][n]
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


            # 01 08 2025: Pseudocode
            '''1. Only grab SAM segments which have continguous lesion
            Grab X% of all intralesion points (n if n ≤3) and Y% of all extralesion points
            (iterate over all lesions) -> stop SAM once
            Save all output (take top 5 for now) to new folder
            Grab the folder and take logical AND'''


            fig, ax=plt.subplots()
            im = plt.imread(os.path.join(output_dir, frame_names[ann_frame_idx]))
            implot = plt.imshow(im,cmap='gray')

            threshold = 10
            #box_prev = get_bounding_box(im,threshold)
            #box = [box_prev[1],box_prev[0],box_prev[3],box_prev[2]]
            print('Volume:')
            print(unique_vols[curr_e])
            plt.scatter(slices[:,1],slices[:,0],color='#FF0000',alpha=0.2)
            #fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()

            g = input('Is this a lesion? (Si/No)')
            confirmed = False
            passCurr = False
            if (g == 'Si') or (g == ''):
                print('Alright! Lesion confirmed. Moving forward with segmentation.')
                pass
            else:
                print('Skipping current possible lesion!')
                passCurr = True
            if passCurr:
                curr_e = curr_e + 1
                continue
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
                            gusta = True
                            confirmed = True
                        else:
                            h = input('Are you sure you want to discard? (Si/No)')
                            if h == 'Si':
                                confirmed = True
                    except:
                        print('Incorrect input')
                    finally:
                        pass
                # Let's add a positive click at (x, y)
                ###coordinates = np.array(coordinates, dtype=np.float32)
                # for labels, `1` means positive click and `0` means negative click
                if yn:
                    labels = np.array(yn, np.int32)
                else:
                    continue

                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=coordinates,
                    labels=labels,
                )

            #quit()
            #print(slices)
            #quit()
            # 0.3 for mag images 02_1070
            # 0.05 01_1007 GRE
            eta = 0.2 # sparsity index
            slices_permuted = sparsify(slices, eta)
            eta_bg = 0.007
            # Create a mask to identify elements in arr1 that are also in arr2

            fullslice = []
            d = 5
            for x in range(size[0]):#range(box[0],box[2]):#
                for y in range(size[1]):#range(box[1],box[3]):#
                    for z in range(size[2]):
                        #if not [x,y] in slices[:,0:2].tolist():
                        if not np.any(np.linalg.norm(slices - np.array([x,y,z]), axis=1) <= d):
                            fullslice.append([x,y,z])
            fullslice = np.array(fullslice)
            bg_permuted = sparsify(fullslice, eta_bg)
            #coordinates = []
            #labels = [] #np.array(np.zeros((len(bg_permuted))), np.int32)
            #bg_permuted = []
            #for j in range(len(bg_permuted)):
            #    coordinates.append((bg_permuted[j][1],bg_permuted[j][0]))
            #    labels.append(0)#labeled_array_SAM[bg_permuted[j][1],bg_permuted[j][0],bg_permuted[j][2]])
            #slices_permuted = slices
            num_centroids = len(slices_permuted)
            for j in range(num_centroids):
                coordinates.append((slices_permuted[j][1],slices_permuted[j][0]))
                labels = np.append(labels,1)
            points = np.array(coordinates, dtype=np.float32)

            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels
                #box=(box)
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


            for m in out_mask_logits:
                m = m[0] # temporary code to choose the "best" mask
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
                show_points(points, labels, plt.gca())
                show_mask((m > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                plt.show()

            happy = input('Are you happy with this final segmentation result which will be propagated through the whole volume? (Si = yes, No = redo selection)')
            if not ((happy == 'Si') or (happy == 'Yes') or (happy =='')):
                continue

            #plt.figure(figsize=(9, 6))
            #plt.title(f"frame {ann_frame_idx}")
            #plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
            #show_points(points, labels, plt.gca())
            #show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            #plt.show()
            # Now do the whole video
            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Beta: now run the other way to fill the whole image
            # reverse video

            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            with open('{0}_240_lesionnum_{1}.pkl'.format(output_csv,e),'wb') as f:
                pickle.dump(video_segments, f)
            nifti_res = np.zeros(size)
            print('Applying rankings...')
            for i in video_segments.keys():#range(len(video_segments)):
                for out_obj_id, out_mask in video_segments[i].items():
                    nifti_res[:,:,i] = out_mask[0]#nifti_res_SAMonly_lesionrank[:,:,i] + np.multiply(out_mask[0],max_lesion_cont[i])#curr_mask_0_symmetrycheck
            print(np.where(nifti_res > 0))
            nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
            nib.save(nifti_mask, '{0}_240_lesionnum_{1}.nii.gz'.format(output_csv,e))

            curr_e = curr_e + 1
        else:
            n = 2
            data = image_data
            with open('{0}_240_lesionnum_{1}.pkl'.format(output_csv,e),'rb') as f:
                video_segments = pickle.load(f)

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


    # (128,192,192) # QSM
    # (192,128,192) # QSM
    # (128,180,192) # GRE
    # (176,240,256) # MP-RAGE
    from scipy import ndimage
    nifti_res = np.zeros(size)
    maxes = []
    for i in video_segments.keys():#range(len(video_segments)):
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
    prevmask = np.zeros((size[0],size[1]))
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
