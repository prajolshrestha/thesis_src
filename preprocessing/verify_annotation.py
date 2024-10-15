import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

# Closest ridge of voronoi polygon
crop = np.asarray(PIL.Image.open('/home/codebind/thesis-data-preprocessing/AOSLO_dataset_using_voronoi_seed_ridge_circle/4571_left_retina_Confocal_RPE_IS_2023_09_09/crop/4571_os_129_7533_-6_0_calculated_ref_102_lps_8_lbss_8_sr_n_50_cropped_5_2_0.tif')) 
mask = np.asarray(PIL.Image.open("/home/codebind/thesis-data-preprocessing/AOSLO_dataset_using_voronoi_seed_ridge_circle/4571_left_retina_Confocal_RPE_IS_2023_09_09/mask/4571_os_129_7533_-6_0_calculated_ref_102_lps_8_lbss_8_sr_n_50_cropped_5_2_0_masks.tif"))

# Closest midpoint vertex of polygon from csv file
# crop = np.asarray(PIL.Image.open('/home/codebind/thesis-data-preprocessing/AOSLO_dataset_using_annotation_seed_ridge_circle/4571_left_retina_Confocal_RPE_IS_2023_09_09/crop/4571_os_129_7533_-6_0_calculated_ref_102_lps_8_lbss_8_sr_n_50_cropped_5_2_0.tif')) 
# mask = np.asarray(PIL.Image.open("/home/codebind/thesis-data-preprocessing/AOSLO_dataset_using_annotation_seed_ridge_circle/4571_left_retina_Confocal_RPE_IS_2023_09_09/mask/4571_os_129_7533_-6_0_calculated_ref_102_lps_8_lbss_8_sr_n_50_cropped_5_2_0_masks.tif"))

plt.title("crop with mask Overlay")
plt.imshow(mask, cmap='gray')
plt.imshow(crop, alpha=0.6, cmap='gray')  
plt.show()