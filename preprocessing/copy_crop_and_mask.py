import os 
import shutil
from tqdm import tqdm

src_dir = '/home/hpc/iwi5/iwi5171h/data_processing/confocal_dataset/crop_and_mask_for_all_degree/closest_vertex_120'
dest_dir = '/home/hpc/iwi5/iwi5171h/data_processing/confocal_dataset/train_test_dataset/final/train'

folders_to_copy = ['crop', 'mask']

# Test
# aoslo_names = ['2538_right_retina_Confocal_RPE_2023-09-10',
#                '4017_left_retina_Confocal_RPE_2023-09-10',
#                '4078_right_retina_Confocal_RPE_2023-09-09',
#                #'4064_od_retina_montaged_2023',
#             ]


# Train
aoslo_names = ['4571_left_retina_Confocal_RPE_IS_2023_09_09',
               '5160_right_retina_2024',
                '5160_right_retina_Confocal_RPE_2023-09-10',
                '5163_right_retina_RPE_2023-06-08',
                '5165_right_retina_RPE_2023-06-08',
                '5181_left_retina_Confocal_RPE_IS_2023-09-10',
                '5188_right_retina_Confocal_RPE_2023-09-10',
                '8323_right_retina_confocal_RPE_2023-09-13']

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for aoslo in aoslo_names:

    for folder in folders_to_copy:
        src_folder = os.path.join(src_dir, aoslo, folder)

        if not os.path.exists(dest_dir) or not os.listdir(src_folder):
            continue

        # files = os.listdir(src_folder)
        # for filename in tqdm(files, desc=f'Processing {aoslo}/{folder}'):
        #     src_file = os.path.join(src_folder, filename)

        #     # Copy only low degree or high degree images
        #     try:
        #         parts = filename.split('_')
        #         # Locate the position of 'Small_calculated' and extract the relevant values
        #         #idx0 = parts[4] #.index('Small') #- 1
        #         #idx1 = parts[5] #.index('Small') #- 2
        #         value1 = int(parts[4])
        #         value2 = int(parts[5])
                
        #         if (-4 < value1 < 4 and -4 < value2 < 4): # for low degree only
        #         #if not (-4 < value1 < 4 and -4 < value2 < 4): # for high degree only
        #             dest_file = os.path.join(dest_dir, filename)
        #             shutil.copyfile(src_file, dest_file)
        #     except (ValueError, IndexError):
        #         continue
            
        # Copy all degree images
        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copyfile(src_file, dest_file)

            




