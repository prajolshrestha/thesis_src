import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv 

import warnings
warnings.filterwarnings('ignore')


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

if __name__ == '__main__':
    file = "/home/codebind/thesis-data-preprocessing/data/4571_left_retina_Confocal_RPE_IS_2023_09_09/AOSLO/4571_os_129_7531_-5_0_calculated_1p0_ref_124_lps_8_sr_n_40_cropped_5.tiff"
    
    img = cv.imread(file, cv.IMREAD_GRAYSCALE) 
    blob_detection(img)