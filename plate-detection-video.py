import cv2
import numpy as np

###################################################################
# # Just for debugging - get video and display
# cap = cv2.VideoCapture('resources/pexels-taryn-elliott-5309422.mp4')
# try:
#     while True:
#         success, vid = cap.read()
#         cv2.imshow('Video', vid)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# except:
#     print("Reached end of video file")
######################################################################

nPlateCascade = cv2.CascadeClassifier('resources/haarcascade_russian_plate_number.xml')
videoPath = 'resources/Mercedes-Glk-1406.mp4'
cap = cv2.VideoCapture(videoPath)
heightImg = 560
widthImg = 960
minArea = 500
highlightColor = (255, 0, 255)
accentColor = (255, 0, 255)
font = cv2.QT_FONT_NORMAL
count = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Create grayscale image
    plates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)  # Detect all number plates in image

    for (x, y, w, h) in plates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), accentColor, 2) # Draw rectangle around plate
            cv2.putText(img, "Number Plate", (x, y-5), font, 1, highlightColor, 2) # Add name
            imgRoi = img[y:y+h, x:x+w] # Crop region of interest
            cv2.imshow("ROI", imgRoi) # Show cropped plate in separate window

    cv2.imshow("License Plate Detection", img) # Display video with superimposed boundary box

    if cv2.waitKey(1) & 0xFF == ord('s'): # Keep running until you press `s`
        cv2.imwrite("scanned/nPlate_"+str(count)+".jpg", imgRoi)
        count += 1














