"""
TASK:
    Simple Image Processing CLI Tool

    - Takes an image path from the user
    - Loads and resizes the image (600x600)
    - Converts it to grayscale
    - Lets user:
        1. Save color image
        2. Display image
        3. Save grayscale image
    - Runs in loop until user exits
"""

import cv2
import pathlib

BASE_DIR = pathlib.Path(
    "/home/nirzar-diwan/Desktop/computer_vision_learning/images/resized"
)

print(" Welcome to the image processing program")

while True:
    user_input = input("Please enter the image path: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Exiting the program. Goodbye!")
        break
    image_path = pathlib.Path(user_input)
    if not image_path.is_file():
        print("Invalid file path. Please try again.")
        continue
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load the image. Please check the file and try again.")
        continue
    image = cv2.resize(image, (600, 600))
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("What would you like to do with the image?")
    print("1. Save the image")
    print("2. Display the image")
    print("3. Save the grey image")
    selected_option = input("Enter your choice (1 or 2 or 3): ")
    if selected_option not in ["1", "2", "3"]:
        print("Invalid option. Please try again.")
        continue
    if selected_option == "1":
        print("Enter the new image name (without extension): ")
        new_image_name = input().strip()
        new_image_path = BASE_DIR / f"{new_image_name}.png"
        cv2.imwrite(str(new_image_path), image)
        print(f"Image saved successfully at {new_image_path}")
    elif selected_option == "3":
        print("Enter the new grey image name (without extension): ")
        new_image_name = input().strip()
        new_image_path = BASE_DIR / f"{new_image_name}_grey.png"
        cv2.imwrite(str(new_image_path), grey_image)
        print(f"Grey image saved successfully at {new_image_path}")
    else:
        print("Displaying the image. Close the window to continue.")
        cv2.imshow(f"{image_path.name}", image)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
