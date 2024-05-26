import cv2

# img shower for any pre-determined img loc from which user chooses which img to view
def display_menu():
    print("Select an image:")
    print("1. Img of hands")
    print("2. Img of city")
    print("3. Img of bike")

def load_selected_image(choice):
    if choice == 1:
        return cv2.imread(r'2C:\Users\Dell\Downloads\Wallpaper\New folder (2)\18b831c2b9ad673903d67f54c4c8061e.jpg')
    elif choice == 2:
        return cv2.imread(r'C:\Users\Dell\Downloads\Wallpaper\New folder (2)\c48d6e6f76739158e97ea5533cf6f748.jpg')
    elif choice == 3:
        return cv2.imread(r'C:\Users\Dell\Downloads\Wallpaper\New folder (2)\b3de56d9742c42770c8e09777230fa7f.jpg')
    else:
        return None
    
    image = cv2.imread(file_path)
    if image is not None:
        # image size
        height, width, channels = image.shape
        print(f"Loaded image: {file_path}")
        print(f"Image size: {width} x {height} pixels")
        return image
    else:
        print(f"Error loading the image at path: {file_path}")
        return None

def main():
    display_menu()

    try:
        # user input
        user_choice = int(input("Enter the number of the selected image (1, 2, or 3): "))

        # Load selected image
        selected_image = load_selected_image(user_choice)

        if selected_image is not None:
            # Display 
            cv2.imshow('Selected Image', selected_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
