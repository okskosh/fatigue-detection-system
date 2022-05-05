from scipy.spatial import distance
from imutils import face_utils 
import time
import dlib
import cv2
import matplotlib.pyplot as plt


def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])

	ear_aspect_ratio = (A + B) / (2.0 * C)
	return ear_aspect_ratio

def calculate_MAR(mouth): 
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])

    mouth_aspect_ratio = (A + B + C) / 3.0
    return mouth_aspect_ratio


# all eye and mouth aspect ratio
ear_list=[]
mar_list=[]

# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
EAR_THRESHOLD = 0.26
# Declare another costant to hold the consecutive number of frames to consider for a blink (20?)
CONSECUTIVE_FRAMES = 13 
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 1

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

## This is if you are using a Web Cam. If 0 does not work change it to 1.
cam = cv2.VideoCapture(0)

## This is if you are using a Raspberry Pi Camera V2.
# cam = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(width)+', height='+str(height)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

start = time.time()

## Timer while loop
while time.time() - start < 30:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

    # Detect faces 
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        # Convert it to a (68, 2) size numpy array 
        shape = face_utils.shape_to_np(face_landmarks)

        # Draw a rectangle over the detected face 
        (x, y, w, h) = face_utils.rect_to_bb(face) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

        # Put a number 
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ## Left eye detection
        leftEye = shape[lstart:lend]
        leftEyeHull = cv2.convexHull(shape[lstart:lend])
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

        ## Right eye detection
        rightEye = shape[rstart:rend]
        rightEyeHull = cv2.convexHull(shape[rstart:rend])
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 6)
        ear_list.append(EAR)

        mouth = shape[mstart:mend]
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1) 
        MAR = calculate_MAR(mouth)
        MAR = round(MAR / 10, 6)
        mar_list.append(MAR)

        ## Check Eye Aspect Ratio for blink
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                cv2.putText(frame, "Closed eyes ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        else:
            FRAME_COUNT = 0

        if MAR > MAR_THRESHOLD:
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
            cv2.putText(frame, "Yawn ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Fatigue Detection System", frame)

    key = cv2.waitKey(1)
    if key == 'q':
        break

print(ear_list)
print(mar_list) 

plt.figure()
plt.plot(ear_list)
# plt.subplots_adjust(bottom=0.30)
plt.title("EAR calculation")
plt.ylabel('EAR')
#plt.gca().axes.get_xaxis().set_visible(False)

plt.figure()
plt.plot(mar_list)
plt.title("MAR calculation")
plt.ylabel("MAR")

plt.show()

cam.release()
cv2.destroyAllWindows()
