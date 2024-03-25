import cv2


try:
    image = cv2.imread(r'C:\Users\Dell\Downloads\Wallpaper\New folder (2)\9405c5b3b7bfcf8a62369ee0c1d4d3b0.jpg')


    if image is not None:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

         # Resize the image to a specific width and height
        width = 300  # specify your desired width
        height = 450  # specify your desired height
        resized_image = cv2.resize(image, (width, height))

        # Display the resized image
        cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error loading the image at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
