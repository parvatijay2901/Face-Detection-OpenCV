# Face detection/OpenCV

This is a Haarcascade face detector in OpenCV, we have used inbuilt haarcascade classifier to train the data.
The Haar cascade classifier uses the AdaBoost algorithm to detect multiple facial organs including the eye, nose, and mouth. First, it reads the image to be detected and converts it into gray image, then loads Haar cascade classifier to judge whether it contains human face.

The model follows LBPH algorithm.
The LBP operator is used to describe the contrast information of a pixel to its neighborhood pixels. The original LBP operator is defined in the window of 3*3. Using the center pixel value as the threshold of window, it compares with the gray value of the adjacent 8 pixels. If the neighborhood pixel value is greater than or equal to the center pixel value, the value of pixel position is marked as 1, otherwise marked as 0. 
The LBPH algorithm uses the histogram of LBP characteristic spectrum as the feature vector for classification. It divides a picture into several sub regions, then extracts LBP feature from each pixel of sub region, establishing a statistical histogram of LBP characteristic spectrum in each sub region, so that each sub region can using a statistical histogram to describe the whole picture by a number of statistical histogram components.

The data set is taken directly through webcam (run One_TrainData.py).

After creating a data set, run Three_FaceLocker.py
We will get the training data we previously made, and the Haar Cascade is further trained by superimposing the positive image over a set of negative images (after converting to gray scaled image).

The webcam then opens automatically, and classify whether the face is Locked or Unlocked (depending on the data set we create).

Also, there is another python file detect.py, which detects the face and eyes (using the same algorithm)
