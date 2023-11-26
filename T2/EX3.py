import numpy as np
import cv2 as cv
vid = cv.VideoCapture(0)
if not vid.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #hsv = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
    #bound_lower = np.array([10, 100, 100]) 
    bound_lower = np.array([5, 100, 100], np.uint8)
    #bound_lower = np.array([5, 150, 150])
    #bound_upper = np.array([25, 255, 255])
    bound_upper = np.array([15, 255, 255], np.uint8)
    mask_orange = cv.inRange(hsv, bound_lower, bound_upper)
    kernel = np.ones((5, 5), np.uint8)
    dilate_mask = cv.dilate(mask_orange, kernel, iterations=1)
    img_orange = cv.bitwise_and(frame,frame,mask=dilate_mask)
    # Display the resulting frame
    cv.imshow('frame', img_orange)
    if cv.waitKey(1) == 27:
        break
# When everything done, release the capture
vid.release()
cv.destroyAllWindows()

