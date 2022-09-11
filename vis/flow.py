import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("/home/pep/drive/PCLOUD/Dataset/RWF2000_Dataset/RWF-2000/train/Fight/3kpviz7lAMY_4.avi")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while True:
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if not ret: break
    # Opens a new window and displays the input frame
    # cv.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow[..., 0] -= np.mean(flow[..., 0])
    flow[..., 1] -= np.mean(flow[..., 1])
    # normalize each component in optical flow
    # flow[..., 0] = cv.normalize(flow[..., 0],None,0,255, cv.NORM_MINMAX)
    # flow[..., 1] = cv.normalize(flow[..., 1],None,0,255, cv.NORM_MINMAX)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    # mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    thresh = np.mean(magnitude)
    magnitude[magnitude < thresh] = 0
    heatmap = cv.applyColorMap(magnitude.astype(np.uint8), cv.COLORMAP_TWILIGHT_SHIFTED)
    merge_img = cv.addWeighted(heatmap, 0.5, frame, 0.5, 0)
    cv.imshow('Merge', merge_img)
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == 27:
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()