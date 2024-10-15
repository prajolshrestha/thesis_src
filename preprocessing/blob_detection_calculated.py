import cv2 as cv
import numpy as np

def blob_detection(image):
    
    
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

if __name__ == "__main__":
    file = "/home/codebind/thesis-data-preprocessing/data/2538_right_retina_Confocal_RPE_2023-09-10/AOSLO/2538_od_24_654_1_0_calculated_1p0_ref_44_lps_8_sr_n_25_cropped_5.tiff"
    #file = "/home/codebind/thesis-data-preprocessing/data/4571_left_retina_Confocal_RPE_IS_2023_09_09/AOSLO/4571_od_129_7537_-10_0_calculated_ref_149_lps_8_lbss_8_sr_n_40_cropped_5.tiff"
    file = '/home/codebind/thesis-data-preprocessing/data/4571_left_retina_Confocal_RPE_IS_2023_09_09/AOSLO/4571_os_129_7548_-5_-1_calculated_ref_83_lps_8_lbss_8_sr_n_50_cropped_5.tiff'
    file = "/home/codebind/thesis-data-preprocessing/data/4571_left_retina_Confocal_RPE_IS_2023_09_09/AOSLO/4571_os_129_7531_-5_0_calculated_1p0_ref_124_lps_8_sr_n_40_cropped_5.tiff"
    
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    blobs = blob_detection(image)