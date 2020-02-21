import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture("eye_recording_video.flv")

detector = dlib.get_frontal_face_detector() # Object will be recognize face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Predict by using dlib's facial landmark picture

def midpoint(p1 ,p2): #To find the center of the eyes
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)



font = cv2.FONT_HERSHEY_PLAIN

def get_gaze_ratio(eye_points, facial_landmarks):
    
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
								(facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
								(facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
								(facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
								(facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
								(facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        
        
        #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)


    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape()
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio



while True:
    ret, frame = cap.read()
    if ret is False:
        break
    roi = frame[269: 795, 537: 1416] #from the video take 269th row to 795th row then 537th column to 1416. column values
                                       
    
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        
        (x, y, w, h) = cv2.boundingRect(cnt)

        cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()