import cv2
import os
import uuid
import numpy as np

labels = ['No detection', 'hello', 'rock', 'peace', 'thumbs up', 'ok']
number_images = 40
base_dir = "C:/Users/rajpa/OneDrive/Desktop/Project 3301/collected images"


# Creating directories for each label
for label in labels:
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)

cap = cv2.VideoCapture(0)

for label in labels:
    print('Collecting images for {}'.format(label))
    imgnum = 0
    while imgnum < number_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (35, 35), 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
        if np.mean(thresh1) > 127:
            thresh1 = cv2.bitwise_not(thresh1)
            # Defining kernel size
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply erosion
            thresh1 = cv2.erode(thresh1, kernel, iterations=1)

            # Defining kernel size
            kernel_size_dilation = 3
            kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)

            #Applying dilation
            thresh1 = cv2.dilate(thresh1, kernel_dilation, iterations=1)


        cv2.imshow('frame', frame)
        cv2.imshow('Thresholded', thresh1)

            # Capture image on 'c' key press
        if cv2.waitKey(1) & 0xFF == ord('c'):
            imagename = os.path.join(base_dir, label, f"{label}.{uuid.uuid1()}.jpg")
            cv2.imwrite(imagename, thresh1)
            print("Image saved: ", imgnum + 1)
            imgnum += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()
