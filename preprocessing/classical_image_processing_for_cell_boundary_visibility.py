import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.fft
from shapely.geometry import Polygon

import json

from PIL import Image, ImageDraw
import cv2 as cv 

from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings('ignore')
import os
from os import listdir
from os.path import isfile
from os.path import join 
from pathlib import Path

####################### Classical Image Processing #################################
def band_pass_filter(gray_image, threshold, median_filter_size, cutoff_radius):
    """
        Filter image to get better quality image
    """

    # Adjust contrast and Brightness
    adjusted_gray_image = cv.convertScaleAbs(gray_image, alpha=1.0, beta=-50.0) # adjust brightness
    #blurred = cv.GaussianBlur(adjusted_gray_image, (3,3), 0) # reduce noise
    #gradient = cv.morphologyEx(blurred, cv.MORPH_GRADIENT, np.ones((3,3), np.uint8)) # highlight boundary

    #binary = cv.adaptiveThreshold(adjusted_gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 3)
    #opening = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
   

    # Fourier Transform
    fourier_transform = np.fft.fft2(adjusted_gray_image) # output is complex value array
    fourier_shift = np.fft.fftshift(fourier_transform) # shift DC component to center
    magnitude_spectrum = 20 * np.log10(np.abs(fourier_shift)) # extract magnitude and visualize better
    phase_spectrum = np.angle(fourier_shift)

    
    # Calculate the threshold dynamically based on the given percentage
    total_pixels = magnitude_spectrum.size 
    sorted_magnitude_spectrum = np.sort(magnitude_spectrum, axis=None)
    cumulative_sum = np.cumsum(sorted_magnitude_spectrum)
    threshold_index = np.argmax(cumulative_sum >= cumulative_sum[-1] * threshold)
    threshold_value = sorted_magnitude_spectrum[threshold_index]
    # Thresholding
    threshold_spectrum = np.where(magnitude_spectrum > threshold_value, 1, 0)
    
    # Median filter
    if median_filter_size % 2 == 0:
        median_filter_size += 1
    median_filter_mask = cv.medianBlur(threshold_spectrum.astype(np.uint8), median_filter_size)
   
    #cut off low frequencies (circle with radius cutoff_radius) - reduce brightness and illumination changes
    rows, cols = median_filter_mask.shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    cutoff_mask = x*x + y*y >= cutoff_radius**2
    cutoff_mask = np.uint8(cutoff_mask)
    final_mask = np.multiply(median_filter_mask, cutoff_mask)

    # Inverse fourier Transform
    spectrum = np.multiply(fourier_shift, cutoff_mask)
    f_ishift = np.fft.ifftshift(spectrum) #filtered_spectrum
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back_cutoff = np.uint8(img_back)

    # Apply final mask
    #filtered_magnitude_spectrum = np.where(filtered_spectrum_mask==1, np.abs(fourier_shift), 0) 
    #spectrum = np.multiply(filtered_magnitude_spectrum, np.exp(1j * phase_spectrum))
    spectrum = np.multiply(fourier_shift, final_mask)

    # Inverse fourier Transform
    f_ishift = np.fft.ifftshift(spectrum) #filtered_spectrum
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(img_back)

    # Visualize
    # plt.figure(figsize=(15, 10))
    # plots = [
    #     (gray_image, "original image"),
    #     (magnitude_spectrum, "Magnitude spectrum"),
    #     (phase_spectrum, "Phase spectrum"),
    #     (threshold_spectrum, f"After thresholding (Threshold = {threshold})"),
    #     (median_filter_mask, f"After median filter (size= {median_filter_size})"),
    #     (final_mask, f"After cutting off low frequencies (radius = {cutoff_radius})"),
    #     (img_back, "Filtered image")
    # ]

    # for i, (img, title) in enumerate(plots, 1):
    #     plt.subplot(3, 3, i)
    #     plt.imshow(img, cmap='gray')
    #     plt.title(title)
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()


    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Original calculated image")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(adjusted_gray_image, cmap='gray')
    plt.title("adjusted brightness")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(img_back_cutoff, cmap='gray')
    plt.title("Only cufoff mask applied")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(img_back, cmap='gray')
    plt.title("Final Filtered calculated Image")
    plt.axis('off')
    plt.show()


    return img_back, img_back_cutoff

def marker_based_watershed_segmentation(cutoff_image, original_image):
    
    # basic image processing (blur, noise removal, binary conversion)
    #equalized = cv.equalizeHist(image)
    blurred = cv.GaussianBlur(cutoff_image, (3,3), 0)
    gradient = cv.morphologyEx(blurred, cv.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    #_, binary = cv.threshold(gradient, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    binary = cv.adaptiveThreshold(gradient, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 3)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3,3), np.uint8)) # noise removal

    # Compute foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, foreground = cv.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    
    #foreground = cv.erode(opening, np.ones((2,2), np.uint8), iterations=1) # cells not touching eachoter?

    # Compute Background
    background = cv.dilate(opening, np.ones((2,2), np.uint8), iterations=3)

    # Compute unknown region
    unknown = cv.subtract(background, foreground)

    # Markers
    _, markers = cv.connectedComponents(cutoff_image)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed Algo
    img_colored = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(img_colored, markers)
    img_colored[markers == -1] = [255,0,0]

    # Visualize
    plt.figure(figsize=(10,8))
    plots = [(cutoff_image, "Gray Image"),
             #(equalized, "Equalized"),
            (binary, "Binary Image"),
            (opening, "Opening"),
            (background, "Sure Background (Dilation)"),
            #(dist_transform, "Distance Transform"),
            (foreground, "Sure Foreground (DT + Thresholding)"),
            (unknown, "Unknown area (Bg - Fg)"),
            (markers, "markers"),
            (img_colored, "segmentation result")

            ]

    for i, (img, title) in enumerate(plots, 1):
        plt.subplot(3,3,i)
        cmap = 'gray' if i != 7 else 'jet'
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.suptitle("Image segmentation with marker-based watershed algorithm")
    plt.figure(figsize=(10,8))

    plt.imshow(img_colored, cmap)
    plt.show()

    return img_colored

def blob_detection(image):
    
    #image = cv.convertScaleAbs(image, alpha=1.0, beta=-50.0) # adjust brightness
     
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 2
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Keypoints", im_with_keypoints)
    cv.waitKey(0)

    return im_with_keypoints    





###################### Create crop and mask for segmentation model ###############################################
parser = argparse.ArgumentParser()
parser.add_argument('--aoslo_folder', type=str, default='/home/codebind/thesis-data-preprocessing/data', help='path to AOSLO Images')
parser.add_argument('--image_name', type=str, default='2538_right_retina_Confocal_RPE_2023-09-10', help='AOSLO image name')
parser.add_argument('--output_folder', type=str, default='./AOSLO_calculated_dataset', help='path to output folder')

def crop_coord(x_center_coords, y_center_coords, delta):
    crop_x = []
    crop_y = []   
    x_max = max(x_center_coords)
    y_max = max(y_center_coords)   
    x_min = min(x_center_coords)
    y_min = min(y_center_coords)   
    xt = int((x_max - x_min) / delta)
    yt = int((y_max - y_min) / delta)
    crop_x.append(x_min)
    crop_y.append(y_min)
    
    #1. 
    if xt <= 1:
        crop_x.append(x_max)
    else:
        for i in range(1, xt + 1):
            crop_x.append(x_min + i * delta) 
    
    #2.
    if yt <= 1:
        crop_y.append(y_max)
    else:
        for i in range(1, yt + 1):
            crop_y.append(y_min + i * delta)
    return crop_x, crop_y
    

def main():
    args = parser.parse_args()
    aoslo_folder_path = Path(args.aoslo_folder)
    aoslo_name = Path(args.image_name)
    output_folder = Path(args.output_folder)

    # Create new folders 
    Path(os.path.join(output_folder, aoslo_name, 'crop')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder, aoslo_name, 'mask')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder, aoslo_name, 'seg')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder, aoslo_name, 'crop_masked')).mkdir(parents=True, exist_ok=True)

    # Read csv file for specific patient image filename 
    cone_csv = pd.read_csv(os.path.join(aoslo_folder_path, aoslo_name, 'cone_coords.csv'), skipinitialspace=True)
    image_ids = cone_csv['image_id'].unique()

    #
    #image_ids = [1,2,3]
    #

    for image_id in tqdm(image_ids):
        # Retrive filename
        cell_data = cone_csv[(cone_csv['image_id'] == image_id)].reset_index(drop=True)
        filename = str(cell_data['filename'].to_numpy()[0] + 'f')
        filename = filename.replace('confocal', 'calculated')

        ## For testing
        #test = "4064_od_170_11273_-3_-3_Small_calculated_ref_112_lps_8_lbss_8_sr_n_50_cropped_5"
        #test = "4064_od_170_11212_-4_0_Small_calculated_ref_122_lps_8_lbss_8_sr_n_50_cropped_5"
        #test = "4064_od_170_11214_-5_0_Small_calculated_ref_93_lps_8_lbss_8_sr_n_50_cropped_5" 
        #test = "4064_od_170_11217_-6_0_Small_calculated_ref_95_lps_8_lbss_8_sr_n_50_cropped_5"
        #filename = test + '.tiff'
        #json_filename = test + '.json'
        
        ##

        

        # Read image
        sample_image = cv.imread(os.path.join(aoslo_folder_path, aoslo_name, 'AOSLO',filename), cv.IMREAD_GRAYSCALE)
        assert sample_image is not None, "Unable to read image."
        sample_pixels = np.array(sample_image)
        samp_h, samp_w = sample_pixels.shape[0], sample_pixels.shape[1]
        
        # Filter image
        filtered_sample_image, cutoff_image = band_pass_filter(sample_pixels, threshold=0.90, median_filter_size=17, cutoff_radius=3)
        seg_image = marker_based_watershed_segmentation(cutoff_image, sample_image)
        blob_img = blob_detection(sample_image)
        
        # Create crop and mask
        json_filename = filename.replace('.tiff', '.json')
        json_file = os.path.join(aoslo_folder_path, aoslo_name, 'AOSLO', json_filename)
        # if not os.path.isfile(json_file):
        #     continue
        with open(json_file, 'r') as f:
            data = json.load(f)

        #
        #samp_h, samp_w = data["imageHeight"], data["imageWidth"]
        #print(samp_h,samp_w)
        #

        poly_coords = []
        cone_center_coords = []
        for cell in data["shapes"]:
            # compute polygon vertices
            temp = cell["points"]
            temp.append(temp[0])
            poly_coords.append(np.array([temp], dtype=np.int32))
            
            # Compute center coordinates
            polygon = Polygon(cell["points"])
            centroid = polygon.centroid
            cone_center_coords.append([int(centroid.x), int(centroid.y)])
            
        mask_binary = np.zeros(shape = (samp_h, samp_w), dtype=np.uint8)
        for i in range(len(poly_coords)):
            cv.fillPoly(mask_binary, poly_coords[i], 1)
        
        sample_pixels_masked = (sample_pixels * mask_binary).astype(np.uint8)
        sample_pixels = (sample_pixels).astype(np.uint8)

        colors = []
        mask = np.zeros(shape = (samp_h, samp_w), dtype=np.uint8)
        outline = np.zeros(shape = (samp_h, samp_w), dtype= np.uint8)
        for i in range(len(poly_coords)):
            color = list(np.random.random(size=3) * 256)
            cv.fillPoly(mask, poly_coords[i], color)
            cv.polylines(outline, poly_coords[i], True, color, 1)
            #cv.imshow('out', outline)
            #cv.waitKey(0)
            colors.append(color)
        
        #plt.imshow(mask_binary, cmap='gray')
        #plt.title('Filled Polygon Mask')
        #plt.show()
        #print(cone_center_coords)

        ## Crop
        x_center_coords = []
        y_center_coords = []
        for coord in cone_center_coords:
            x_center_coords.append(coord[0])
            y_center_coords.append(coord[1])
        
        crop_x, crop_y = crop_coord(x_center_coords=x_center_coords, y_center_coords=y_center_coords, delta=120)

        #print(crop_x, crop_y)

        for i in range(0, len(crop_y)-1):
            for j in range(0, len(crop_x)-1):
                x_1 = crop_x[j]
                y_1 = crop_y[i]

                x_2 = crop_x[j+1]
                y_2 = crop_y[i+1]

                # mask, sample_pixel and sample_pixels_masked processed for specific crop region
                mask_crop = mask[y_1 : y_2, x_1:x_2]
                sample_pixels_masked_crop = sample_pixels_masked[y_1:y_2, x_1:x_2]
                sample_pixels_crop = sample_pixels[y_1:y_2, x_1:x_2]
                
                outline_crop = outline[min(y_center_coords):max(y_center_coords), min(x_center_coords):max(x_center_coords)]

                # Relative cone center coordinate for specific crop region
                cone_center_coords_new = []
                for coords in cone_center_coords:
                    if (coords[0] > x_1) & (coords[0] < x_2) & (coords[1] > y_1) & (coords[1] < y_2):
                        cone_center_coords_new.append([coords[1] - y_1, coords[0] -x_1])
                coords_array = np.array(cone_center_coords_new)

                seg = {'outlines' : np.array(outline_crop),
                       'colors' : np.array(colors),
                       'masks' : np.array(mask_crop)
                       }
                seg = np.asarray(seg)

                cv.imwrite(os.path.join(output_folder, aoslo_name, 'crop_masked', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '.tif')), sample_pixels_masked_crop)
                cv.imwrite(os.path.join(output_folder, aoslo_name, 'mask', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_masks.tif')), mask_crop)
                cv.imwrite(os.path.join(output_folder, aoslo_name, 'crop', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '.tif')), sample_pixels_crop)

                np.save(os.path.join(output_folder, aoslo_name, 'seg', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_seg')), seg)
                np.save(os.path.join(output_folder, aoslo_name, 'seg', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_coord')), coords_array)



        
        #break

# Testing
def test_band_pass_filter():
    #a simple 3x3 grayscale image
    np.random.seed(0)  
    image = np.random.randint(0, 256, size=(3, 3), dtype=np.uint8)

    # Test the band_pass_filter function
    result_image, cutoff_image = band_pass_filter(image, threshold=0.4, median_filter_size=1, cutoff_radius=1)



if __name__ == "__main__":
    main()

    #test_band_pass_filter()
    #test_marker_based_watershed_segmentation()


