import cv2

# Load an image
image = cv2.imread(r'C:\Users\Dell\Downloads\c675c20b9ba8eb49ef4145d5bdca2e88.jpg')

# Resize the image to a specific width and height
new_width, new_height = 300, 200
resized_image = cv2.resize(image, (new_width, new_height))

# Display the resized image
cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
