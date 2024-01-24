# Hand Gesture Recognition System

HGR_3.py is a code for Hand Gesture recognition made using Tensorflow and OpenCV library. It records the video, does some image pre-processing and the hand gesture is predicted using the web-based tool named Teachable Machine that creates downloadable simple-to-use Machine Learning Models.

## Requirements to run the code

- Python 3.11 (Anything above 3.11 doesn't work)
- NumPy
- OpenCV (cv2)
- Tensorflow

## Installation

Ensure that the above libraries are installed, if not then use the following command in your terminal.

``` 
pip install opencv-python numpy tensorflow
```


## Usage

MAKE SURE THAT YOU DOWNLOAD ALL THE CONTENT OF THE REPOSITORY IN ONE DIRECTORY OR FOLDER. ESPECIALLY THE HGR_3.PY REQUIRES THE FOLDER 'converted_savedmodel' AND ITS CONTENTS TO BE IN THE SAME DIRECTORY.

Run the script HGR_3.py in a Python environment. (Runs successfully in VS CODE by selecting the option run without debugging)

The script then opens two frames. The first one is for live video feed and the second one is for displaying the thresholded image. (Thresholded image makes it easier to check if the image processing works properly)

Make sure to perform hand gestures within the rectangle in the gesture window provided on the display. THE FOLLOWING GESTURES ARE RECOGNIZED ['hello', 'rock', 'peace', 'thumbs up', 'ok']

For better results, try to keep the background simple (eg. white or green wall) and also keep the hand till the wrist in the gesture window.

## Notes

The performance of the hand recognition system depends on several factors like the environment the user is testing, lightning conditions, quality of code, gesture detection model and the resolution of the user's front camera

## Support

READ THE DOCUMENTATION FOR HELP AND IF THE USER STILL REQUIRES ASSISTANCE WITH CODE, CONTACT US IMMEDIATELY.

Raj - bhaveshkumap@mun.ca
Devang - dkdonda@mun.ca




