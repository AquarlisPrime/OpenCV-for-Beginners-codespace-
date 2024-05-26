import cv2
import numpy as np

# Load an image
image = cv2.imread(r'C:\Users\Dell\Downloads\c675c20b9ba8eb49ef4145d5bdca2e88.jpg')

# Specify the rotation angle
angle = 45

# Get the image center coordinates
center = (image.shape[1] // 2, image.shape[0] // 2)

# Create the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Apply the rotation to the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Display the rotated image
cv2.namedWindow('Rotated Image', cv2.WINDOW_NORMAL)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
