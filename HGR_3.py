import cv2
import numpy as np
import os
import tensorflow as tf


# Setting the path to the py file and later using it to load ml model and labels
HGR_path = os.path.dirname(__file__)

# Load TensorFlow SavedModel
model_path = os.path.join(HGR_path, 'converted_savedmodel', 'model.savedmodel')
model = tf.saved_model.load(model_path)

labels_path = os.path.join(HGR_path, 'converted_savedmodel', 'labels.txt')

# Loading labels
with open(labels_path, "r") as label_file:
    class_names = [line.strip() for line in label_file.readlines()]

cap = cv2.VideoCapture(0)
while(cap.isOpened()):

    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Getting hand image from rectangle window on screen
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    crop_img = img[100:300, 100:300]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholding using Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Checking avg color of  thresholded image, if hand is black then apply bitwise not
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

    # Preparing image for ml model
    processed_img = cv2.resize(thresh1, (224, 224))
    processed_img = np.expand_dims(processed_img, axis=-1)
    processed_img = np.repeat(processed_img, 3, axis=-1)
    processed_img = processed_img / 255.0

    processed_img = tf.image.convert_image_dtype(processed_img, tf.float32)
    processed_img = tf.expand_dims(processed_img, axis=0)

    # Performing prediction
    prediction = model(processed_img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Showing prediction on frame
    cv2.putText(img, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  
    cv2.imshow('Gesture', img)

    cv2.imshow('Thresholded', thresh1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



