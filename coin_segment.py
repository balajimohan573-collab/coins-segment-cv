import cv2
import numpy as np
import os

# Input and output folders
input_folder = "images"
output_folder = "outputs"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in the input folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Read image
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Otsu Thresholding
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert (if coins are dark on light background)
        binary = cv2.bitwise_not(binary)

        # Optional: Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on a copy of the image
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # Count and display coins
        print(f"{file_name}: {len(contours)} coins detected")

        # Save outputs
        base_name = os.path.splitext(file_name)[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), binary)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_segmented.png"), result)

print("\n✅ All images processed! Check the 'outputs' folder for results.")
