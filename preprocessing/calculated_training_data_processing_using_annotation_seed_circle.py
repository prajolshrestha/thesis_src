import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import ast
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw
import os
from os import listdir
from os.path import isfile
from os.path import join
from pathlib import Path
import argparse
from scipy.spatial import Voronoi
from shapely import Polygon 


parser = argparse.ArgumentParser()
parser.add_argument('--aoslo_folder', type=str, default='/home/codebind/thesis-data-preprocessing/data', help='path to AOSLO images')
parser.add_argument('--image_name', type=str, default='2538_right_retina_Confocal_RPE_2023-09-10', help='AOSLO image name')
parser.add_argument('--output_folder', type=str, default='./AOSLO_dataset_using_annotation_seed_circle', help='path to output folder')


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
    
    # Read csv file for the specific patient
    cone_csv = pd.read_csv(os.path.join(aoslo_folder_path, aoslo_name, 'cone_coords.csv'), skipinitialspace = True)
    image_ids = cone_csv['image_id'].unique() #different image_id of same Specific patient  
    
    for image_id in tqdm(image_ids):
        cell_data = cone_csv[(cone_csv['image_id'] == image_id)].reset_index(drop=True)
        filename = str(cell_data['filename'].to_numpy()[0] + 'f')
        filename = filename.replace('confocal', 'calculated') # changed
        sample_image = cv.imread(os.path.join(aoslo_folder_path, aoslo_name, 'AOSLO', filename), cv.IMREAD_GRAYSCALE)
        sample_pixels = np.array(sample_image)
        samp_h = sample_pixels.shape[0]
        samp_w = sample_pixels.shape[1]
        
        #1. cone center coordinates are populated
        cone_center_coords = []
        for cell in range(0, cell_data.shape[0]):
            cone_center_coords.append([cell_data['cone_x_local_pix'][cell], cell_data['cone_y_local_pix'][cell]])
        
        #2. cone Polygon vertices are populated
        poly_coords = []
        circumference_points = []
        for poly_coord in range(0, cell_data.shape[0]):
            temp = list(ast.literal_eval(cell_data['poly_verts_local_pix'][poly_coord]))
            temp.append(temp[0])
            poly_coords.append(np.array([temp]))  

        circumference_points = []
        for center, poly_vertices in zip(cone_center_coords, poly_coords):
            distances = np.linalg.norm(poly_vertices[0] - center, axis = 1)
            min_distance_index = np.argmin(distances)
            closest_vertex = np.array(poly_vertices[0][min_distance_index])
            circumference_points.append(closest_vertex)

    

        #3. fill Circle with 1. else 0
        mask_binary = np.zeros(shape = (samp_h, samp_w), dtype=np.uint8)
        for center, circumference_point in zip(cone_center_coords, circumference_points):
            radius = int(np.linalg.norm(center - circumference_point))
            center = tuple(map(int, center))
            cv.circle(mask_binary, center, radius, 1, thickness=-1)

        #A. original image is masked such that only values inside cell remains unchanged, else 0
        sample_pixels_masked = (sample_pixels*mask_binary).astype(np.uint8) 
        sample_pixels = (sample_pixels).astype(np.uint8) 
        
        #B. different colors for each instance cell, mask and outline
        colors = []    
        mask = np.zeros(shape = (samp_h, samp_w), dtype=np.uint8) 
        outline = np.zeros(shape = (samp_h, samp_w), dtype=np.uint8)
        for center, circumference_point in zip(cone_center_coords, circumference_points):
            radius = int(np.linalg.norm(center - circumference_point))  # substract some value to reduce overlap
            center = tuple(map(int, center))
            color = tuple((np.random.random(size=3) * 256).tolist())
            cv.circle(mask, center, radius, color, thickness=-1)  # Fill the circle
            cv.circle(outline, center, radius, color, thickness=1)  # Draw the outline
            colors.append(color)

        #4. Cropping
        x_center_coords = []
        y_center_coords = []
        for coord in cone_center_coords:
            x_center_coords.append(coord[0])
            y_center_coords.append(coord[1])

        crop_x, crop_y = crop_coord(x_center_coords=x_center_coords, y_center_coords=y_center_coords, delta=100)     
    
        for i in range(0, len(crop_y)-1):
            for j in range(0, len(crop_x)-1):
                x_1 = crop_x[j]
                y_1 = crop_y[i]
                
                x_2 = crop_x[j+1]
                y_2 = crop_y[i+1]
                # mask, sample_pixel and sample_pixels_masked processed for specific crop region
                mask_crop = mask[y_1 : y_2, x_1 : x_2]
                sample_pixels_masked_crop = sample_pixels_masked[y_1 : y_2, x_1 : x_2]
                sample_pixels_crop = sample_pixels[y_1 : y_2, x_1 : x_2]
                
                outline_crop = outline[min(y_center_coords):max(y_center_coords),min(x_center_coords):max(x_center_coords)]
                
                # Relative cone center coordinate for specific crop region
                cone_center_coords_new = []
                for coords in cone_center_coords:
                    if (coords[0] > x_1) & (coords[0] < x_2) & (coords[1] > y_1) & (coords[1] < y_2):
                        cone_center_coords_new.append([coords[1]-y_1, coords[0]-x_1])
                coords_array = np.array(cone_center_coords_new)

                seg = {'outlines' : np.array(outline_crop), 
                       'colors' : np.array(colors), 
                       'masks' : np.array(mask_crop)} 
                seg = np.asarray(seg)

                cv.imwrite(os.path.join(output_folder, aoslo_name, 'crop_masked', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '.tif')), sample_pixels_masked_crop)
                cv.imwrite(os.path.join(output_folder, aoslo_name, 'mask', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_masks.tif')), mask_crop)
                cv.imwrite(os.path.join(output_folder, aoslo_name, 'crop', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '.tif')), sample_pixels_crop)
                
                
                np.save(os.path.join(output_folder, aoslo_name, 'seg', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_seg')), seg)
                np.save(os.path.join(output_folder, aoslo_name, 'seg', str(filename[:-5] + '_' + str(i) + '_' + str(j) + '_coord')), coords_array)
    
       
if __name__ == "__main__":
    main()

