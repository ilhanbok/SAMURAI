# Compares volume estimates using SAM(thresholding-based), QSM, and raw thresholding
# Ilhan Bok, Patrick Belton
# 2025

import Adjunct_SAM_lesion_threshAndFilter_wrapper_01082025_NoShow as SAM_segment
import RAW_lesion_threshAndFilter_wrapper as RAW_segment
import QSM_lesion_threshAndFilter_wrapper as QSM_segment
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print('Usage: python thresholdSAM.py NIFTI_PATH output_csv(.csv) mode(QSM or RAW) threshold(e.g. 400) flip(lesions are dark?) blur(gaussian blur amount to hide artifacts?) skullstrip(strip skull or not?)')
    nifti_path = str(sys.argv[1])
    output_csv = sys.argv[2]
    image_mode = sys.argv[3]
    thresh = float(sys.argv[4])
    flip = bool(sys.argv[5])
    bluramt = float(sys.argv[6])
    skullstrip = bool(sys.argv[7])
    print(sys.argv[1])
    # Generate CSV
    if image_mode == "QSM":
        print('Mode QSM')
        input('Press Enter to continue.')
        labeled_array = QSM_segment.main(output_csv)
    else:
        print('Mode RAW')
        input('Press Enter to continue.')
        labeled_array = RAW_segment.main(nifti_path, output_csv, thresh, flip, bluramt, skullstrip)
    # Read CSV
    list_of_centroids = []
    list_of_corresponding_indices = []

    df = pd.read_csv('{0}.csv'.format(output_csv))
    for i in range(len(df)):
        if i == 0:
            continue
        list_of_centroids.append([df.loc[i]['COMs_x'],df.loc[i]['COMs_y'],df.loc[i]['COMs_z'],df.loc[i]['volumes']])
        list_of_corresponding_indices.append(i-1)
    # Apply CSV
    ##SAM_segment.main(list_of_centroids,list_of_corresponding_indices) # centroid based
    list_of_centroids = []
    list_of_corresponding_indices = np.array([])
    volumes = np.array([])
    for i in range(0,np.max(labeled_array)):
        print('{0}/{1} Labels'.format(i,np.max(labeled_array)))
        matches = np.where(labeled_array == i)
        #if len(matches[0]) > 10:
        #    list_of_centroids = []
        #    list_of_corresponding_indices = np.array([])
        #    print(matches)
        for k in range(len(matches[0])):
            list_of_centroids.append([matches[0][k],matches[1][k],matches[2][k]])
        #print(list_of_centroids)
        list_of_corresponding_indices = np.append(list_of_corresponding_indices,np.multiply(np.ones((len(matches[0]))),i))
        volumes = np.append(volumes,np.multiply(np.ones((len(matches[0]))),df.loc[i]['volumes']))#np.multiply(np.ones((df.loc[i]['volumes'])),i))
        if len(matches[0]) > 100:
            #print(list_of_centroids)
            #print(list_of_corresponding_indices)
            #print(volumes)
            print('Large lesion.')
    list_of_centroids = np.array(list_of_centroids)
    #print(list_of_centroids)
    #print(list_of_corresponding_indices)
    #quit()
    SAM_segment.main(nifti_path, output_csv,list_of_centroids,list_of_corresponding_indices.astype(int),volumes) # full threshold based
