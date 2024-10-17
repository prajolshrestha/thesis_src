# Thesis: Automated Detection and Analysis of Photoreceptors in Retinal Imaging

This repository includes the **Automated Detection and Analysis of Photoreceptors in Retinal Imaging** project, for precise detection and analysis of cone density and area in AOSLO (Adaptive Optics Scanning Laser Ophthalmoscopy) images. The project utilizes advanced deep learning techniques such as **StarDist** and **CellPose 2.0** to process images from confocal and calculated modalities, respectively.

## Source Directory Structure
```
thesis_src/
    │
    ├── calculated_modality/
    │         └── analysis
    │         |      └── csv_files # Saves computed density and area here
    │         └── calculate_cell_density_and_area.ipynb # Calculates cone density and area for each calculated image 
    │ 
    │
    ├── confocal_modality/
    │         └── analysis
    │         |      └── csv_files # Saves computed density and area here
    │         └── calculate_cell_density_and_area.ipynb # Calculates cone density and area for each confocal image
    │
    ├── trained_models/    # Directory to store all trained deep learning models.
    │         └── cellpose_models/
    │         |      
    │         └── stardist_models/
    ├── plots/     # Folder to save all generated plots and figures.
    │
    ├── plot_all.ipynb # Jupyter Notebook to generate all diagrams after calculating cone density and area using both confocal and calculated modalities.
    │
    ├── multimodal_AOSLO_data/
    │         └── # Add your original multimodal AOSLO data here.
    |
    |
    ├── preprocessing/ # Includes Classical Image Processing files (bandpass filter, watershed algo, blob detection, hough circle, ..)
    |                  # Includes annotation processing to convert polygon to circle or ellipse
    |                  #                     └──> # Using both center and boundary information from Confocal AOSLO data
    |                  #                                └──> Closest vertex: annotation_seed_circle
    |                  #                                └──> Closest ridge midpoint vertex: annotation_seed_ridge_circle
    |                  #                                └──> Closest ridge midpoint vertex: annotation_seed_ridge_ellipse (Note: Not used in this work: Incorrect annotation))
    |                  #                     └──> # Using cone information from Confocal AOSLO data
    |                  #                                └──> Voronoi method: # Note: We do not use this technique in our work as it is not accurate!
    |                  #                                        └──> # We use center info. but later we extract the center and boundary from the Voronoi regions            
    |                  #                                                  └──> voronoi_circle
    |                  #                                                  └──> voronoi_ridge_circle
    |                  #                                                  └──> voronoi_ridge_ellipse
    |                  #                                        └──> # We use center info. and later we extract the  boundary from the Voronoi regions            
    |                  #                                                  └──> voronoi_seed_circle
    |                  #                                                  └──> voronoi_seed_ridge_circle
    |                  #                                                  └──> voronoi_seed_ridge_ellipse
    |                  #                     └──> verify_annotation.py # To do a visual inspection by overlaying extracted mask and original crop
    |                  #  
    ├── utils/ # training notebook for stardist and evaluation notebook for cellpose 
    |


Note: You may need to update the file path as required!            
```
## In the plot_all.ipynb file, there are 2D plots. The parameters of the asymmetric power law can still be refined. Adjust as required!
```
# Initial parameters (p0)
p0 = [
    5.0,    # kappa: baseline density (log scale) - starting higher as it's peak density
    0.0,    # k_s: random intercept - start at neutral
    -0.5,   # pi_n: nasal slope - steeper initial decay
    -0.3,   # pi_t: temporal slope - slightly less steep
    0.0,    # p_s: participant effect - start at neutral
    0.0,    # p_ns: nasal random effect - start at neutral
    0.0,    # p_ts: temporal random effect - start at neutral
    0.1,    # rho: offset - small positive value
    0.0,    # r_s: random offset - start at neutral
    0.0     # e_i: error term - start at neutral
]

# Bounds
bounds = (
    # Lower bounds
    [
        4.0,    # kappa: minimum reasonable log density
        -0.2,   # k_s: small negative random adjustment
        -2.0,   # pi_n: steeper negative slope possible for nasal
        -1.5,   # pi_t: less steep negative slope for temporal
        -0.2,   # p_s: small negative participant effect
        -0.2,   # p_ns: small negative nasal random effect
        -0.2,   # p_ts: small negative temporal random effect
        0.05,   # rho: small positive offset (avoid zero)
        -0.2,   # r_s: small negative random offset
        -0.2    # e_i: small negative error term
    ],
    # Upper bounds
    [
        6.0,    # kappa: maximum reasonable log density
        0.2,    # k_s: small positive random adjustment
        -0.1,   # pi_n: shallow negative slope
        -0.1,   # pi_t: shallow negative slope
        0.2,    # p_s: small positive participant effect
        0.2,    # p_ns: small positive nasal random effect
        0.2,    # p_ts: small positive temporal random effect
        0.5,    # rho: larger offset allowed
        0.2,    # r_s: small positive random offset
        0.2     # e_i: small positive error term
    ]
)

```

## Follow the steps below to set up the necessary environments and check GPU configurations:

## 1. Setup StarDist
```
$ conda create -n stardist python=3.8 -y
$ cd stardist
$ conda activate stardist

$ pip install tensorflow
$ pip install git+https://github.com/stardist/stardist.git
```

### Request a GPU resource, load the required modules & check GPU compatibility
```
$ salloc.tinygpu --gres=gpu:1 --time=01:00:00
$ module load python/3.8-anaconda cuda/11.8.0 cudnn/8.8.0.121-11.8 tensorrt/8.5.3.1-cuda11.8-cudnn8.6
$ conda activate stardist

$ python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
$ python3 -c 'import tensorflow as tf; print(tf.random.uniform((2,3)))'
```


## 2. Setup Cellpose 2.0:
```
$ conda create --name cellpose python=3.8
$ conda activate cellpose
$ python -m pip install cellpose[gui]
$ pip uninstall torch
$ conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

$ python -m pip install notebook 
$ python -m pip install matplotlib
```

### Request a GPU resource, load the required modules & check GPU compatibility
```
$ salloc.tinygpu --gres=gpu:1 --time=01:00:00
$ module load python/3.8-anaconda cuda/11.8.0 cudnn/8.8.0.121-11.8 
$ cd cellpose
$ conda activate cellpose

$ python -c 'import torch; print(torch.rand(2,3).cuda())'
```



