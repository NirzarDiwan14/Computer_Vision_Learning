"""
TASK:
    ask user what thet want in image
    line, rectangle,circle,text
    ask user for the coordinates and other details
    draw the shape or text on the image and show it to the user
    also give option to save the image
"""

import cv2
import pathlib

BASE_DIR = pathlib.Path(
    "/home/nirzar-diwan/Desktop/computer_vision_learning/images/resized"
)

print(" Welcome to the image processing program")
user_input = input("Please enter the image path: ")
image_path = pathlib.Path(user_input)
if not image_path.is_file():
    print("Invalid file path. Please try again.")
    exit(0)
image = cv2.imread(str(image_path))
if image is None:
    print("Failed to load the image. Please check the file and try again.")
    exit(0)
image = cv2.resize(image, (600, 600))

while True:
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Exiting the program. Goodbye!")
        break

    print("What would you like to do with the image?")
    print("1. Draw Line")
    print("2. Draw Rectangle")
    print("3. Draw Circle")
    print("4. Draw Text")
    print("5. Save the Image and Exit")
    selected_option = input("Enter your choice (1,2,3 or 4): ")

    if selected_option not in ["1", "2", "3", "4","5"]:
        print("Invalid option. Please try again.")
        continue
    if selected_option == "1":
        # Drawing line
        print("Enter the co-ordinates for the Drawing the line.")

        print("Enter Starting Point of the line (like (0,0)): ")
        starting_point_x = int(input("Enter X: "))
        starting_point_y = int(input("Enter Y: "))
        print("Enter Ending Point of the line (like (600,600)): ")
        ending_point_x = int(input("Enter X: "))
        ending_point_y = int(input("Enter Y: "))

        cv2.line(
            image,
            (starting_point_x, starting_point_y),
            (ending_point_x, ending_point_y),
            (0, 0, 255),
            3,
        )
        if image is not None:
            cv2.imshow("Line", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif selected_option == "2":
        # Drawing Rectangle
        print("Enter the co-ordinates for the Drawing the Rectangle.")

        print("Enter Starting Point of the Rectangle (like (0,0)): ")
        starting_point_x = int(input("Enter X: "))
        starting_point_y = int(input("Enter Y: "))
        print("Enter Ending Point of the Rectangle (like (600,600)): ")
        ending_point_x = int(input("Enter X: "))
        ending_point_y = int(input("Enter Y: "))

        cv2.rectangle(
            image,
            (starting_point_x, starting_point_y),
            (ending_point_x, ending_point_y),
            (255, 0, 0),
            3,
        )
        if image is not None:
            cv2.imshow("Rectangle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif selected_option == "3":
        # Drawing Circle
        print("Enter the co-ordinates for the Drawing the Circle.")

        print("Enter Center Points of the Circle (like (0,0)): ")
        center_point_x = int(input("Enter X: "))
        center_point_y = int(input("Enter Y: "))
        print("Enter Radius of the circle(like 50): ")
        radius = int(input("Enter Radius: "))

        cv2.circle(image, (center_point_x, center_point_y), radius, (0, 255, 0), 3)
        if image is not None:
            cv2.imshow("Circle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif selected_option == "4":
        # Putting Text
        print("Enter the co-ordinates for putting text.")

        print("Enter Origin Point of the Rectangle (like (10,50)): ")
        origin_point_x = int(input("Enter X: "))
        origin_point_y = int(input("Enter Y: "))
        text = input("Enter Text: ")

        cv2.putText(
            image,
            text,
            (origin_point_x, origin_point_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=4,
        )
        if image is not None:
            cv2.imshow(f"added Text: {text}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif selected_option == "5":
        print("Enter the new image name (without extension): ")
        new_image_name = input().strip()
        new_image_path = BASE_DIR / f"{new_image_name}_edited.png"
        cv2.imwrite(str(new_image_path), image)
        print(f"Image saved successfully at {new_image_path}")
        break
