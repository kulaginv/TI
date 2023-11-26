import numpy as np
import cv2 as cv
import time

vid = cv.VideoCapture(0)
if not vid.isOpened():
    print("Cannot open camera")
    exit()
    
positions = []
to_file = []
start = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(gray,5)
    #tic = cv.getTickCount()
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,70,param1=30,param2=50,minRadius=20,maxRadius=100)
    #toc = cv.getTickCount()
    #print("HoughCircles exec time = ",(toc - tic) / cv.getTickFrequency() * 1000, "ms")
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            # Enregistrer la position
            positions.append((i[0], i[1]))
            # Enregistrer la position et le temps
            to_file.append((time.time()-start, (i[0], i[1])))
    for pos in positions:
        cv.circle(frame, pos, 2, (255, 0, 0), -1)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == 27:
        break
# When everything done, release the capture
vid.release()
cv.destroyAllWindows()

with open('ball_positions.txt', 'w') as file:
    for t, pos in to_file:
        file.write(f"{t}: {pos}\n")

print("Positions saved to ball_positions.txt")

