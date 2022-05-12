from scipy.spatial import distance
from imutils import face_utils 
from fuzzy_logic import FatigueSys
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
    D = distance.euclidean(mouth[12], mouth[16])

    mouth_aspect_ratio = (A + B + C) / (2.0 * D)
    return mouth_aspect_ratio

def drive_process():
    process(alert=True)

def test_process():
    process(alert=False)

def process(alert): 
    # all eye and mouth aspect ratio
    ear_list=[]
    mar_list=[]

    # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
    EAR_THRESHOLD = 0.20
    # declare another costant to hold the consecutive number of frames to consider for a driver state (3 sec) (20?)
    CONSECUTIVE_FRAMES = 120 
    # Another constant which will work as a threshold for MAR value
    MAR_THRESHOLD = 0.9

    # initialize camera
    cam = cv2.VideoCapture(0)

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    fatigue_sys = FatigueSys()
    driver_state = []

    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(frame, "Press any key to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # detect faces 
        faces = hog_face_detector(gray)

        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)

            # convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(face_landmarks)

            # draw a rectangle over the detected face 
            (x, y, w, h) = face_utils.rect_to_bb(face) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            # put a description 
            cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # left eye detection
            leftEye = shape[42:48]
            leftEyeHull = cv2.convexHull(leftEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

            # right eye detection
            rightEye = shape[36:42]
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 6)
            ear_list.append(EAR)

            # mouth detection
            mouth = shape[48:68]
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1) 
            MAR = calculate_MAR(mouth)
            MAR = round(MAR, 6)
            mar_list.append(MAR)

            driver_state.append(fatigue_sys.compute_inference(EAR, MAR))
            interpreted_state = []
            common_state = []
            if len(driver_state) > CONSECUTIVE_FRAMES:
                for state in driver_state[-CONSECUTIVE_FRAMES:]:
                    interpreted_state.append(fatigue_sys.interpret_membership(state))

                common_state = max(set(interpreted_state), key=interpreted_state.count)

            # check Eye Aspect Ratio for closing the eye
            if EAR < EAR_THRESHOLD:
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                    cv2.putText(frame, "Closed eyes ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # check Mouth Aspect Ratio for closing the mouth
            if MAR > MAR_THRESHOLD:
                cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
                cv2.putText(frame, "Yawn ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("Fatigue Detection System", frame)

        key = cv2.waitKey(1)
        if key != -1:  
            cam.release()
            cv2.destroyAllWindows()
            break

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

    plt.figure()
    plt.plot(driver_state)
    plt.title("Driver State calculation")
    plt.ylabel("Driver State")

    plt.show()
