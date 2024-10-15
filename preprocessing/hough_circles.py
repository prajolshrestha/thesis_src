import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/home/codebind/thesis-data-preprocessing/data/2538_right_retina_Confocal_RPE_2023-09-10/AOSLO/2538_od_24_654_1_0_calculated_1p0_ref_44_lps_8_sr_n_25_cropped_5.tiff'  # Change this to the path of your image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=50, param2=30, minRadius=1, maxRadius=30)

# Ensure at least some circles were found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    # Draw circles on the image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.axis('off')
plt.show()
